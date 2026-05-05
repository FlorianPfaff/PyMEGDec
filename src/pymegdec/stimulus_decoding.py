"""Time-resolved stimulus decoding analyses."""

from __future__ import annotations

import warnings
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from pymegdec.alpha_metrics import write_alpha_metrics_csv
from pymegdec.classifiers import (
    get_default_classifier_param,
    should_use_default_classifier_param,
    train_multiclass_classifier,
)
from pymegdec.data_config import resolve_data_folder
from pymegdec.preprocessing import (
    downsample_data,
    extract_windows,
    filter_features,
    reduce_features_pca,
)

DEFAULT_DECODING_TIME_WINDOW = (-0.2, 0.6)
DEFAULT_DECODING_STEP_S = 0.05
DEFAULT_STIMULUS_WINDOW_SIZE = 0.1
DEFAULT_CHANCE_CLASSES = 16
TRANSFER_DIRECTIONS = ("main-to-cue", "cue-to-main")
SUMMARY_GROUP_FIELDS = (
    "control",
    "control_label",
    "transfer_direction",
    "variant",
    "window_center_s",
    "classifier",
    "components_pca",
    "frequency_low_hz",
    "frequency_high_hz",
)
TEMPORAL_GENERALIZATION_SUMMARY_GROUP_FIELDS = (
    "transfer_direction",
    "variant",
    "train_window_center_s",
    "test_window_center_s",
    "classifier",
    "components_pca",
    "frequency_low_hz",
    "frequency_high_hz",
)
DEFAULT_WINDOW_CENTERS = tuple(
    float(value)
    for value in np.round(
        np.arange(
            DEFAULT_DECODING_TIME_WINDOW[0],
            DEFAULT_DECODING_TIME_WINDOW[1] + DEFAULT_DECODING_STEP_S / 2,
            DEFAULT_DECODING_STEP_S,
        ),
        10,
    )
)


@dataclass(frozen=True)
# pylint: disable-next=too-many-instance-attributes
class StimulusDecodingConfig:
    """Parameters for time-resolved stimulus decoding."""

    window_centers: tuple[float, ...] = DEFAULT_WINDOW_CENTERS
    window_size: float = DEFAULT_STIMULUS_WINDOW_SIZE
    null_window_center: float = float("nan")
    new_framerate: float = float("inf")
    classifier: str = "multiclass-svm"
    classifier_param: object = float("nan")
    components_pca: int | float = 100
    frequency_range: tuple[float, float] = (0.0, float("inf"))
    chance_classes: int = DEFAULT_CHANCE_CLASSES
    random_state: int | None = None
    permutations: int = 0
    permutation_seed: int | None = None
    transfer_direction: str = "main-to-cue"


def window_centers_from_range(time_window: tuple[float, float], step_s: float) -> tuple[float, ...]:
    """Build evenly spaced window centers from a start/stop range."""

    start, stop = time_window
    if step_s <= 0:
        raise ValueError("Window step must be positive.")
    if start > stop:
        raise ValueError("Time window start must be before stop.")
    return tuple(float(value) for value in np.round(np.arange(start, stop + step_s / 2, step_s), 10))


def evaluate_time_resolved_stimulus_transfer(
    data_folder,
    participants,
    *,
    config=None,
    progress=None,
):
    """Evaluate train-main/validate-cue stimulus decoding across time windows."""

    config = config or StimulusDecodingConfig()
    data_folder = resolve_data_folder(data_folder)
    rows = []
    for participant in participants:
        if progress is not None:
            progress(f"START participant={participant}")
        rows.extend(evaluate_participant_time_resolved_stimulus_transfer(data_folder, participant, config=config))
        if progress is not None:
            progress(f"DONE participant={participant}")
    return rows


def evaluate_participant_time_resolved_stimulus_transfer(
    data_folder,
    participant,
    *,
    config=None,
):
    """Evaluate one participant's stimulus transfer accuracy across window centers."""

    rows, _ = _evaluate_participant_time_resolved_stimulus_transfer(
        data_folder,
        participant,
        config=config,
        diagnostic_window_centers=(),
    )
    return rows


def evaluate_participant_stimulus_decoding_diagnostics(
    data_folder,
    participant,
    *,
    config=None,
    diagnostic_window_centers=None,
):
    """Evaluate one participant and return accuracy rows plus prediction diagnostics."""

    return _evaluate_participant_time_resolved_stimulus_transfer(
        data_folder,
        participant,
        config=config,
        diagnostic_window_centers=diagnostic_window_centers,
    )


# jscpd:ignore-start
def evaluate_participant_stimulus_temporal_generalization(
    data_folder,
    participant,
    *,
    config=None,
):
    """Evaluate train-time/test-time stimulus decoding for one participant."""

    config = config or StimulusDecodingConfig()
    classifier_param = config.classifier_param
    if should_use_default_classifier_param(classifier_param):
        classifier_param = get_default_classifier_param(config.classifier)

    train_cue, validation_cue = _transfer_direction_cue_flags(config.transfer_direction)
    train_data = _load_participant_data(data_folder, participant, cue=train_cue)
    validation_data = _load_participant_data(data_folder, participant, cue=validation_cue)
    _check_matching_sample_rate(train_data, validation_data)

    labels_train = np.asarray(train_data["trialinfo"][0][0], dtype=int).ravel()
    labels_validation = np.asarray(validation_data["trialinfo"][0][0], dtype=int).ravel()
    if np.isnan(config.null_window_center):
        labels_train = labels_train - 1
        labels_validation = labels_validation - 1

    if not np.array_equal(np.unique(labels_train), np.unique(labels_validation)):
        warnings.warn("There are labels in the training or validation experiment that are not in the other experiment.")

    train_data = _prepare_data(train_data, config)
    validation_data = _prepare_data(validation_data, config)
    validation_features_by_center = {
        _window_center_key(window_center): _validation_features_for_window(validation_data, float(window_center), config) for window_center in config.window_centers
    }

    rows = []
    for train_window_center in config.window_centers:
        model_bundle = _train_window_model(
            train_data,
            labels_train,
            float(train_window_center),
            classifier_param,
            config,
        )
        for test_window_center in config.window_centers:
            test_features = validation_features_by_center[_window_center_key(test_window_center)]
            rows.append(
                _temporal_generalization_row(
                    participant,
                    labels_train,
                    labels_validation,
                    test_features,
                    float(train_window_center),
                    float(test_window_center),
                    classifier_param,
                    model_bundle,
                    config,
                )
            )
    return rows


# jscpd:ignore-end
def _evaluate_participant_time_resolved_stimulus_transfer(
    data_folder,
    participant,
    *,
    config=None,
    diagnostic_window_centers=None,
):
    config = config or StimulusDecodingConfig()
    classifier_param = config.classifier_param
    if should_use_default_classifier_param(classifier_param):
        classifier_param = get_default_classifier_param(config.classifier)

    train_cue, validation_cue = _transfer_direction_cue_flags(config.transfer_direction)
    train_data = _load_participant_data(data_folder, participant, cue=train_cue)
    validation_data = _load_participant_data(data_folder, participant, cue=validation_cue)
    _check_matching_sample_rate(train_data, validation_data)

    labels_train = np.asarray(train_data["trialinfo"][0][0], dtype=int).ravel()
    labels_validation = np.asarray(validation_data["trialinfo"][0][0], dtype=int).ravel()
    if np.isnan(config.null_window_center):
        labels_train = labels_train - 1
        labels_validation = labels_validation - 1

    if not np.array_equal(np.unique(labels_train), np.unique(labels_validation)):
        warnings.warn("There are labels in the training or validation experiment " "that are not in the other experiment.")

    train_data = _prepare_data(train_data, config)
    validation_data = _prepare_data(validation_data, config)
    permutation_rng = np.random.default_rng(config.permutation_seed)
    diagnostic_centers = _window_center_set(diagnostic_window_centers or ())

    rows = []
    prediction_rows = []
    for window_center in config.window_centers:
        include_predictions = _window_center_key(window_center) in diagnostic_centers
        result = _evaluate_window(
            train_data,
            validation_data,
            labels_train,
            labels_validation,
            participant,
            float(window_center),
            classifier_param,
            config,
            permutation_rng=permutation_rng,
            include_predictions=include_predictions,
        )
        if include_predictions:
            row, window_prediction_rows = result
            rows.append(row)
            prediction_rows.extend(window_prediction_rows)
        else:
            rows.append(result)
    return rows, prediction_rows


def summarize_stimulus_decoding(rows):
    """Summarize decoding rows across participants for each window center."""

    group_fields = _present_group_fields(rows, SUMMARY_GROUP_FIELDS)
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(field, "") for field in group_fields)].append(row)

    summary_rows = []
    for key, group_rows in sorted(grouped.items(), key=lambda item: item[0]):
        accuracies = [_to_float(row["accuracy"]) for row in group_rows]
        mean, std, sem = _summary_stats(accuracies)
        percentages = [100.0 * value for value in accuracies if np.isfinite(value)]
        median = float(np.median(percentages)) if percentages else np.nan
        permutation_p = [_to_float(row.get("permutation_p_value")) for row in group_rows]
        n_with_permutation = sum(np.isfinite(permutation_p))
        significant_05 = sum(value < 0.05 for value in permutation_p if np.isfinite(value))
        significant_01 = sum(value < 0.01 for value in permutation_p if np.isfinite(value))
        chance_accuracy = _to_float(group_rows[0]["chance_accuracy"])
        summary_row = dict(zip(group_fields, key))
        summary_row.update(
            {
                "n_participants": len(group_rows),
                "accuracy_mean": mean,
                "accuracy_std": std,
                "accuracy_sem": sem,
                "percent_mean": 100.0 * mean,
                "percent_median": median,
                "percent_std": 100.0 * std,
                "percent_sem": 100.0 * sem,
                "chance_accuracy": chance_accuracy,
                "chance_percent": 100.0 * chance_accuracy,
                "above_chance_count": sum(value > chance_accuracy for value in accuracies),
                "n_with_permutation": int(n_with_permutation),
                "n_significant_p_0.05": int(significant_05),
                "n_significant_p_0.01": int(significant_01),
            }
        )
        summary_rows.append(summary_row)
    return summary_rows


def summarize_stimulus_decoding_peaks(rows):
    """Return the best decoding window per participant and variant."""

    group_fields = _present_group_fields(rows, ("control", "control_label", "transfer_direction", "variant", "participant"))
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(field, "") for field in group_fields)].append(row)

    peak_rows = []
    for key, group_rows in sorted(grouped.items(), key=lambda item: item[0]):
        peak = max(group_rows, key=lambda row: (_to_float(row["accuracy"]), -abs(_to_float(row["window_center_s"]))))
        peak_row = dict(zip(group_fields, key))
        peak_row.update(
            {
                "peak_window_center_s": peak["window_center_s"],
                "peak_window_start_s": peak["window_start_s"],
                "peak_window_stop_s": peak["window_stop_s"],
                "peak_accuracy": peak["accuracy"],
                "peak_percent": peak["percent"],
                "chance_accuracy": peak["chance_accuracy"],
                "chance_percent": peak["chance_percent"],
            }
        )
        peak_rows.append(peak_row)
    return peak_rows


def summarize_stimulus_prediction_diagnostics(prediction_rows):
    """Summarize trial-level prediction diagnostics."""

    group_fields = _present_group_fields(prediction_rows, ("control", "control_label", "transfer_direction", "variant", "window_center_s"))
    confusion: dict[tuple[object, ...], int] = defaultdict(int)
    per_stimulus_trials: dict[tuple[object, ...], int] = defaultdict(int)
    per_stimulus_correct: dict[tuple[object, ...], int] = defaultdict(int)
    per_stimulus_participants: dict[tuple[object, ...], set[object]] = defaultdict(set)
    for row in prediction_rows:
        base_key = tuple(row.get(field, "") for field in group_fields)
        confusion[base_key + (row["true_stimulus"], row["predicted_stimulus"])] += 1
        per_stimulus_key = base_key + (row["true_stimulus"],)
        per_stimulus_trials[per_stimulus_key] += 1
        per_stimulus_correct[per_stimulus_key] += int(bool(row["correct"]))
        per_stimulus_participants[per_stimulus_key].add(row["participant"])

    confusion_rows = []
    for key, count in sorted(confusion.items()):
        true_stimulus, predicted_stimulus = key[-2:]
        row = dict(zip(group_fields, key[:-2]))
        row.update(
            {
                "true_stimulus": true_stimulus,
                "predicted_stimulus": predicted_stimulus,
                "count": count,
            }
        )
        confusion_rows.append(row)
    per_stimulus_rows = []
    for key in sorted(per_stimulus_trials):
        true_stimulus = key[-1]
        n_trials = per_stimulus_trials[key]
        n_correct = per_stimulus_correct[key]
        accuracy = n_correct / n_trials if n_trials else np.nan
        row = dict(zip(group_fields, key[:-1]))
        row.update(
            {
                "true_stimulus": true_stimulus,
                "n_participants": len(per_stimulus_participants[key]),
                "n_trials": n_trials,
                "n_correct": n_correct,
                "accuracy": accuracy,
                "percent": 100.0 * accuracy,
            }
        )
        per_stimulus_rows.append(row)
    return confusion_rows, per_stimulus_rows


# jscpd:ignore-start
def summarize_stimulus_temporal_generalization(rows):
    """Summarize temporal-generalization rows across participants."""

    group_fields = _present_group_fields(rows, TEMPORAL_GENERALIZATION_SUMMARY_GROUP_FIELDS)
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(field, "") for field in group_fields)].append(row)

    summary_rows = []
    for key, group_rows in sorted(grouped.items(), key=lambda item: item[0]):
        accuracies = [_to_float(row["accuracy"]) for row in group_rows]
        mean, std, sem = _summary_stats(accuracies)
        percentages = [100.0 * value for value in accuracies if np.isfinite(value)]
        median = float(np.median(percentages)) if percentages else np.nan
        chance_accuracy = _to_float(group_rows[0]["chance_accuracy"])
        diagonal_values = {_window_center_key(row["train_window_center_s"]) == _window_center_key(row["test_window_center_s"]) for row in group_rows}
        summary_row = dict(zip(group_fields, key))
        summary_row.update(
            {
                "n_participants": len(group_rows),
                "accuracy_mean": mean,
                "accuracy_std": std,
                "accuracy_sem": sem,
                "percent_mean": 100.0 * mean,
                "percent_median": median,
                "percent_std": 100.0 * std,
                "percent_sem": 100.0 * sem,
                "chance_accuracy": chance_accuracy,
                "chance_percent": 100.0 * chance_accuracy,
                "above_chance_count": sum(value > chance_accuracy for value in accuracies),
                "is_diagonal": bool(diagonal_values == {True}),
            }
        )
        summary_rows.append(summary_row)
    return summary_rows


# jscpd:ignore-end
def write_stimulus_decoding_plots(summary_rows, output_dir):
    """Write group-level stimulus decoding plots."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _plot_group_accuracy(summary_rows, output_dir / "stimulus_decoding_accuracy.png")


# pylint: disable-next=too-many-arguments,too-many-positional-arguments
def export_time_resolved_stimulus_decoding(
    data_folder,
    participants,
    output_path,
    *,
    summary_output_path=None,
    predictions_output_path=None,
    confusion_output_path=None,
    per_stimulus_output_path=None,
    participant_peaks_output_path=None,
    diagnostic_window_centers=None,
    plots_dir=None,
    config=None,
    progress=None,
):
    """Run time-resolved stimulus decoding and write CSV/plot artifacts."""

    config = config or StimulusDecodingConfig()
    data_folder = resolve_data_folder(data_folder)
    rows = []
    prediction_rows = []
    for participant in participants:
        if progress is not None:
            progress(f"START participant={participant}")
        participant_rows, participant_prediction_rows = _evaluate_participant_time_resolved_stimulus_transfer(
            data_folder,
            participant,
            config=config,
            diagnostic_window_centers=diagnostic_window_centers,
        )
        rows.extend(participant_rows)
        prediction_rows.extend(participant_prediction_rows)
        if progress is not None:
            progress(f"DONE participant={participant}")
    write_alpha_metrics_csv(rows, output_path)
    summary_rows = summarize_stimulus_decoding(rows)
    if summary_output_path:
        write_alpha_metrics_csv(summary_rows, summary_output_path)
    if participant_peaks_output_path:
        write_alpha_metrics_csv(summarize_stimulus_decoding_peaks(rows), participant_peaks_output_path)
    if predictions_output_path and prediction_rows:
        write_alpha_metrics_csv(prediction_rows, predictions_output_path)
    if (confusion_output_path or per_stimulus_output_path) and prediction_rows:
        confusion_rows, per_stimulus_rows = summarize_stimulus_prediction_diagnostics(prediction_rows)
        if confusion_output_path:
            write_alpha_metrics_csv(confusion_rows, confusion_output_path)
        if per_stimulus_output_path:
            write_alpha_metrics_csv(per_stimulus_rows, per_stimulus_output_path)
    if plots_dir:
        write_stimulus_decoding_plots(summary_rows, plots_dir)
    return rows, summary_rows


# jscpd:ignore-start
def export_stimulus_temporal_generalization(
    data_folder,
    participants,
    output_path,
    *,
    summary_output_path=None,
    config=None,
    progress=None,
):
    """Run stimulus temporal generalization and write CSV artifacts."""

    config = config or StimulusDecodingConfig()
    data_folder = resolve_data_folder(data_folder)
    rows = []
    for participant in participants:
        if progress is not None:
            progress(f"START participant={participant}")
        rows.extend(evaluate_participant_stimulus_temporal_generalization(data_folder, participant, config=config))
        if progress is not None:
            progress(f"DONE participant={participant}")
    write_alpha_metrics_csv(rows, output_path)
    summary_rows = summarize_stimulus_temporal_generalization(rows)
    if summary_output_path:
        write_alpha_metrics_csv(summary_rows, summary_output_path)
    return rows, summary_rows


# jscpd:ignore-end
def _load_participant_data(data_folder, participant, *, cue):
    suffix = "CueData" if cue else "Data"
    path = Path(data_folder) / f"Part{participant}{suffix}.mat"
    return sio.loadmat(path)["data"][0]


def _transfer_direction_cue_flags(transfer_direction):
    if transfer_direction == "main-to-cue":
        return False, True
    if transfer_direction == "cue-to-main":
        return True, False
    supported = ", ".join(TRANSFER_DIRECTIONS)
    raise ValueError(f"Unsupported transfer direction: {transfer_direction}. Supported directions: {supported}")


def _check_matching_sample_rate(train_data, validation_data):
    train_sample_interval = np.diff(train_data["time"][0][0][0][0, :2])
    validation_sample_interval = np.diff(validation_data["time"][0][0][0][0, :2])
    if not np.allclose(train_sample_interval, validation_sample_interval):
        raise ValueError("Sampling rate of the two experiments must match.")


def _prepare_data(data, config):
    data = filter_features(data, config.frequency_range[0], config.frequency_range[1])
    if config.new_framerate != float("inf"):
        data = downsample_data(data, config.new_framerate)
    return data


# jscpd:ignore-start
@dataclass(frozen=True)
class _WindowModelBundle:
    model: object
    train_window: tuple[float, float]
    train_labels: np.ndarray
    pca_coeff: np.ndarray | None
    train_features_mean: np.ndarray | None
    explained_variance_percent: float
    actual_components_pca: int


def _train_window_model(train_data, labels_train, window_center, classifier_param, config):
    train_window = _centered_window(window_center, config.window_size)
    null_window = _null_window(config)
    train_stimuli_features, train_null_features = extract_windows(train_data, train_window, null_window)
    train_features = np.hstack(train_stimuli_features + train_null_features).T
    train_labels = labels_train
    if train_null_features:
        train_labels = np.concatenate((labels_train, np.zeros(len(train_null_features), dtype=int)))

    pca_components = _actual_pca_components(config.components_pca, train_features)
    pca_coeff = None
    train_features_mean = None
    explained_variance = np.nan
    if config.components_pca != float("inf"):
        train_features, pca_coeff, train_features_mean, explained_variance = reduce_features_pca(train_features, int(config.components_pca))

    model = train_multiclass_classifier(
        train_features,
        train_labels,
        config.classifier,
        classifier_param,
        random_state=config.random_state,
    )
    return _WindowModelBundle(
        model=model,
        train_window=train_window,
        train_labels=train_labels,
        pca_coeff=pca_coeff,
        train_features_mean=train_features_mean,
        explained_variance_percent=explained_variance,
        actual_components_pca=pca_components,
    )


def _validation_features_for_window(validation_data, window_center, config):
    test_window = _centered_window(window_center, config.window_size)
    validation_stimuli_features, _ = extract_windows(validation_data, test_window, (np.nan, np.nan))
    return np.hstack(validation_stimuli_features).T


def _temporal_generalization_row(participant, labels_train, labels_validation, test_features, train_window_center, test_window_center, classifier_param, model_bundle, config):
    if model_bundle.pca_coeff is not None:
        test_features = (test_features - model_bundle.train_features_mean) @ model_bundle.pca_coeff[:, : model_bundle.actual_components_pca]
    predictions = model_bundle.model.predict(test_features)
    accuracy = float(np.mean(predictions == labels_validation))
    chance_accuracy = 1.0 / config.chance_classes
    variant = "without_null" if np.isnan(config.null_window_center) else "with_null"
    test_window = _centered_window(test_window_center, config.window_size)
    return {
        "participant": participant,
        "variant": variant,
        "transfer_direction": config.transfer_direction,
        "train_window_center_s": train_window_center,
        "train_window_start_s": model_bundle.train_window[0],
        "train_window_stop_s": model_bundle.train_window[1],
        "test_window_center_s": test_window_center,
        "test_window_start_s": test_window[0],
        "test_window_stop_s": test_window[1],
        "is_diagonal": _window_center_key(train_window_center) == _window_center_key(test_window_center),
        "accuracy": accuracy,
        "percent": 100.0 * accuracy,
        "chance_accuracy": chance_accuracy,
        "chance_percent": 100.0 * chance_accuracy,
        "above_chance": accuracy > chance_accuracy,
        "n_train_trials": len(labels_train),
        "n_validation_trials": len(labels_validation),
        "n_train_classes": len(np.unique(labels_train)),
        "n_validation_classes": len(np.unique(labels_validation)),
        "classifier": config.classifier,
        "classifier_param": classifier_param,
        "components_pca": config.components_pca,
        "actual_components_pca": model_bundle.actual_components_pca,
        "pca_explained_variance_percent": model_bundle.explained_variance_percent,
        "frequency_low_hz": config.frequency_range[0],
        "frequency_high_hz": config.frequency_range[1],
    }


# jscpd:ignore-end
# pylint: disable-next=too-many-arguments,too-many-positional-arguments,too-many-locals
def _evaluate_window(
    train_data,
    validation_data,
    labels_train,
    labels_validation,
    participant,
    window_center,
    classifier_param,
    config,
    permutation_rng=None,
    include_predictions=False,
):
    train_window = _centered_window(window_center, config.window_size)
    null_window = _null_window(config)
    train_stimuli_features, train_null_features = extract_windows(train_data, train_window, null_window)
    validation_stimuli_features, _ = extract_windows(validation_data, train_window, (np.nan, np.nan))
    train_features = np.hstack(train_stimuli_features + train_null_features).T
    train_labels = labels_train
    if train_null_features:
        train_labels = np.concatenate((labels_train, np.zeros(len(train_null_features), dtype=int)))
    validation_features = np.hstack(validation_stimuli_features).T

    pca_components = _actual_pca_components(config.components_pca, train_features)
    explained_variance = np.nan
    if config.components_pca != float("inf"):
        train_features, coeff, train_features_mean, explained_variance = reduce_features_pca(train_features, int(config.components_pca))
        validation_features = (validation_features - train_features_mean) @ coeff[:, :pca_components]

    model = train_multiclass_classifier(
        train_features,
        train_labels,
        config.classifier,
        classifier_param,
        random_state=config.random_state,
    )
    predictions = model.predict(validation_features)
    accuracy = float(np.mean(predictions == labels_validation))
    permutation_accuracy = np.array([], dtype=float)
    permutation_p = np.nan
    if config.permutations > 0:
        permutation_accuracy = _permutation_accuracy_curve(
            train_features,
            validation_features,
            labels_validation,
            train_labels,
            config.classifier,
            classifier_param,
            config.random_state,
            config.permutations,
            permutation_rng,
        )
        permutation_p = float(np.mean(permutation_accuracy >= accuracy))
        if np.isfinite(permutation_p):
            permutation_p = (permutation_p * config.permutations + 1.0) / (config.permutations + 1.0)
    chance_accuracy = 1.0 / config.chance_classes
    variant = "without_null" if np.isnan(config.null_window_center) else "with_null"
    null_prediction_rate = float(np.mean(predictions == 0)) if variant == "with_null" else np.nan

    row = {
        "participant": participant,
        "variant": variant,
        "transfer_direction": config.transfer_direction,
        "window_center_s": window_center,
        "window_start_s": train_window[0],
        "window_stop_s": train_window[1],
        "accuracy": accuracy,
        "percent": 100.0 * accuracy,
        "chance_accuracy": chance_accuracy,
        "chance_percent": 100.0 * chance_accuracy,
        "above_chance": accuracy > chance_accuracy,
        "n_train_trials": len(labels_train),
        "n_validation_trials": len(labels_validation),
        "n_train_classes": len(np.unique(labels_train)),
        "n_validation_classes": len(np.unique(labels_validation)),
        "n_permutations": int(config.permutations),
        "permutation_seed": config.permutation_seed,
        "permutation_p_value": permutation_p,
        "permutation_accuracy_mean": (float(np.mean(permutation_accuracy)) if permutation_accuracy.size else np.nan),
        "permutation_accuracy_std": (float(np.std(permutation_accuracy, ddof=1)) if permutation_accuracy.size > 1 else np.nan),
        "null_window_center_s": config.null_window_center,
        "null_prediction_rate": null_prediction_rate,
        "classifier": config.classifier,
        "classifier_param": classifier_param,
        "components_pca": config.components_pca,
        "actual_components_pca": pca_components,
        "pca_explained_variance_percent": explained_variance,
        "frequency_low_hz": config.frequency_range[0],
        "frequency_high_hz": config.frequency_range[1],
    }

    if include_predictions:
        return row, _stimulus_prediction_rows(
            participant,
            variant,
            window_center,
            train_window[0],
            train_window[1],
            labels_validation,
            predictions,
            config,
            pca_components,
        )
    return row


def _stimulus_prediction_rows(
    participant,
    variant,
    window_center,
    window_start,
    window_stop,
    labels_validation,
    predictions,
    config,
    actual_components_pca,
):
    rows = []
    for trial_idx, (true_label, predicted_label) in enumerate(zip(labels_validation, predictions)):
        true_stimulus = _display_stimulus_label(true_label, variant)
        predicted_stimulus = _display_stimulus_label(predicted_label, variant)
        rows.append(
            {
                "participant": participant,
                "variant": variant,
                "transfer_direction": config.transfer_direction,
                "window_center_s": window_center,
                "window_start_s": window_start,
                "window_stop_s": window_stop,
                "trial": trial_idx,
                "validation_trial_index": trial_idx,
                "validation_trial_number": trial_idx + 1,
                "true_label": int(true_label),
                "predicted_label": int(predicted_label),
                "true_stimulus": true_stimulus,
                "predicted_stimulus": predicted_stimulus,
                "true_stimulus_id": true_stimulus,
                "predicted_stimulus_id": predicted_stimulus,
                "correct": bool(predicted_label == true_label),
                "classifier": config.classifier,
                "components_pca": config.components_pca,
                "actual_components_pca": actual_components_pca,
            }
        )
    return rows


def _display_stimulus_label(label, variant):
    label = int(label)
    if variant == "without_null":
        return label + 1
    return label


def _centered_window(center, size):
    return center - size / 2, center + size / 2


def _null_window(config):
    if np.isnan(config.null_window_center):
        return np.nan, np.nan
    return _centered_window(config.null_window_center, config.window_size)


def _actual_pca_components(components_pca, features):
    if components_pca == float("inf"):
        return features.shape[1]
    return min(int(components_pca), features.shape[0], features.shape[1])


def _window_center_key(value):
    return float(np.round(float(value), 10))


def _window_center_set(values):
    return {_window_center_key(value) for value in values}


def _to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _permutation_accuracy_curve(
    train_features,
    validation_features,
    labels_validation,
    train_labels,
    classifier,
    classifier_param,
    random_state,
    n_permutations,
    permutation_rng,
):
    if permutation_rng is None:
        permutation_rng = np.random.default_rng()

    permuted_scores = []
    for _ in range(int(n_permutations)):
        permuted_train_labels = np.array(train_labels, copy=True)
        permutation_rng.shuffle(permuted_train_labels)
        model = train_multiclass_classifier(
            train_features,
            permuted_train_labels,
            classifier,
            classifier_param,
            random_state=random_state,
        )
        predictions = model.predict(validation_features)
        permuted_scores.append(float(np.mean(predictions == labels_validation)))
    return np.asarray(permuted_scores, dtype=float)


def _summary_stats(values):
    values = np.asarray(list(values), dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, np.nan, np.nan
    std = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
    return float(np.mean(values)), std, float(std / np.sqrt(values.size))


def _present_group_fields(rows, fields):
    return tuple(field for field in fields if any(field in row for row in rows))


def _plot_group_accuracy(summary_rows, output_path):
    figure, axes = plt.subplots(figsize=(8, 5))
    grouped = defaultdict(list)
    for row in summary_rows:
        grouped[row["variant"]].append(row)

    chance_percent = None
    for variant, rows in sorted(grouped.items()):
        rows = sorted(rows, key=lambda row: _to_float(row["window_center_s"]))
        x = np.asarray([_to_float(row["window_center_s"]) for row in rows], dtype=float)
        y = np.asarray([_to_float(row["percent_mean"]) for row in rows], dtype=float)
        sem = np.asarray([_to_float(row["percent_sem"]) for row in rows], dtype=float)
        chance_percent = _to_float(rows[0]["chance_percent"])
        axes.plot(x, y, marker="o", label=variant.replace("_", " "))
        axes.fill_between(x, y - sem, y + sem, alpha=0.2)

    if chance_percent is not None:
        axes.axhline(chance_percent, color="black", linewidth=1, linestyle="--")
    axes.axvline(0, color="black", linewidth=1, linestyle=":")
    axes.set_xlabel("window center from stimulus (s)")
    axes.set_ylabel("stimulus decoding accuracy (%)")
    axes.grid(True, alpha=0.25)
    axes.legend(fontsize="small")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)

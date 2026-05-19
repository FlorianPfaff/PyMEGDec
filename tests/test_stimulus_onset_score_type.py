from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from pymegdec._stimulus_decoding_core import (
    ONSET_SCORE_TYPE_PREDICTED_CLASS,
    ONSET_SCORE_TYPE_TRUE_CLASS,
    _stimulus_onset_event_rows_from_reptrace,
    _stimulus_onset_scan_rows,
)


def _dummy_config(onset_score_type=ONSET_SCORE_TYPE_TRUE_CLASS):
    return SimpleNamespace(
        transfer_direction="main-to-cue",
        chance_classes=2,
        classifier="dummy",
        components_pca=2,
        frequency_range=(0.0, float("inf")),
        window_size=0.1,
        onset_score_type=onset_score_type,
    )


def _dummy_model_bundle():
    return SimpleNamespace(
        train_window=(0.1, 0.2),
        actual_components_pca=2,
        explained_variance_percent=np.nan,
    )


def test_onset_scan_rows_use_true_class_score_as_stimulus_score():
    rows = _stimulus_onset_scan_rows(
        participant=1,
        variant="without_null",
        train_window_center=0.175,
        scan_window_center=0.1,
        labels_validation=np.asarray([1]),
        predictions=np.asarray([2]),
        predicted_class_scores=np.asarray([10.0]),
        true_class_scores=np.asarray([0.25]),
        score_margins=np.asarray([9.75]),
        onset_scores=np.asarray([0.25]),
        onset_score_type=ONSET_SCORE_TYPE_TRUE_CLASS,
        classifier_param=1.0,
        model_bundle=_dummy_model_bundle(),
        config=_dummy_config(),
        threshold_window=(-0.35, -0.05),
        threshold_quantile=0.95,
    )

    assert len(rows) == 1
    assert rows[0]["stimulus_score"] == 0.25
    assert rows[0]["onset_score"] == 0.25
    assert rows[0]["onset_score_type"] == ONSET_SCORE_TYPE_TRUE_CLASS
    assert rows[0]["predicted_class_score"] == 10.0
    assert rows[0]["true_class_score"] == 0.25


def test_onset_scan_rows_can_preserve_predicted_class_score_mode():
    rows = _stimulus_onset_scan_rows(
        participant=1,
        variant="without_null",
        train_window_center=0.175,
        scan_window_center=0.1,
        labels_validation=np.asarray([1]),
        predictions=np.asarray([2]),
        predicted_class_scores=np.asarray([10.0]),
        true_class_scores=np.asarray([0.25]),
        score_margins=np.asarray([9.75]),
        onset_scores=np.asarray([10.0]),
        onset_score_type=ONSET_SCORE_TYPE_PREDICTED_CLASS,
        classifier_param=1.0,
        model_bundle=_dummy_model_bundle(),
        config=_dummy_config(onset_score_type=ONSET_SCORE_TYPE_PREDICTED_CLASS),
        threshold_window=(-0.35, -0.05),
        threshold_quantile=0.95,
    )

    assert rows[0]["stimulus_score"] == 10.0
    assert rows[0]["onset_score"] == 10.0
    assert rows[0]["onset_score_type"] == ONSET_SCORE_TYPE_PREDICTED_CLASS


def _event_scan_row(scan_window_center_s, *, onset_score, predicted_score, true_score):
    return {
        "participant": 1,
        "variant": "without_null",
        "transfer_direction": "main-to-cue",
        "train_window_center_s": 0.175,
        "train_window_start_s": 0.125,
        "train_window_stop_s": 0.225,
        "scan_window_center_s": scan_window_center_s,
        "scan_window_start_s": scan_window_center_s - 0.05,
        "scan_window_stop_s": scan_window_center_s + 0.05,
        "validation_trial_index": 0,
        "validation_trial_number": 1,
        "true_label": 1,
        "predicted_label": 2,
        "true_stimulus": 2,
        "true_stimulus_id": 2,
        "predicted_stimulus_id": 3,
        "correct": False,
        "stimulus_score": onset_score,
        "onset_score": onset_score,
        "onset_score_type": ONSET_SCORE_TYPE_TRUE_CLASS,
        "predicted_class_score": predicted_score,
        "true_class_score": true_score,
        "score_margin": predicted_score - true_score,
        "score_threshold": np.nan,
        "above_threshold": False,
        "threshold_quantile": 0.0,
        "threshold_window_start_s": -0.2,
        "threshold_window_stop_s": -0.05,
        "classifier": "dummy",
        "components_pca": 2,
        "actual_components_pca": 2,
        "frequency_low_hz": 0.0,
        "frequency_high_hz": float("inf"),
    }


def test_onset_event_rows_carry_detection_score_components():
    scan_rows = [
        _event_scan_row(-0.1, onset_score=0.2, predicted_score=0.8, true_score=0.2),
        _event_scan_row(0.1, onset_score=0.3, predicted_score=0.9, true_score=0.3),
    ]

    event_rows = _stimulus_onset_event_rows_from_reptrace(
        scan_rows,
        threshold_window=(-0.2, -0.05),
        threshold_quantile=0.0,
        detection_start_s=0.0,
    )

    assert len(event_rows) == 1
    assert event_rows[0]["detected"] is True
    assert event_rows[0]["onset_score_type"] == ONSET_SCORE_TYPE_TRUE_CLASS
    assert event_rows[0]["stimulus_score_at_detection"] == 0.3
    assert event_rows[0]["onset_score_at_detection"] == 0.3
    assert event_rows[0]["predicted_class_score_at_detection"] == 0.9
    assert event_rows[0]["true_class_score_at_detection"] == 0.3
    assert event_rows[0]["score_margin_at_detection"] == 0.6

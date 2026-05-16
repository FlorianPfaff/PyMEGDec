"""Composed cross-subject stimulus decoding implementation.

The historical implementation remains in ``_stimulus_cross_subject_legacy``.
This module installs the result-changing scoring and target-alignment behavior
inside that implementation module, so the public API no longer depends on
package ``__init__`` side effects.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from pymegdec import _stimulus_cross_subject_legacy as _impl

_ORIGINAL_SCORE_OUTER_FOLD_MODEL = _impl._score_outer_fold_model


def _ranked_label_metrics(true_labels, class_scores, score_classes):
    """Return rank metrics without dropping unscoreable true-label trials."""

    true_label_ranks = _impl._true_label_ranks(true_labels, class_scores, score_classes)
    finite_ranks = true_label_ranks[np.isfinite(true_label_ranks)]
    if true_label_ranks.size == 0 or class_scores.ndim != 2 or class_scores.shape[1] == 0:
        return {
            "true_label_ranks": true_label_ranks,
            "top2_accuracy": np.nan,
            "top3_accuracy": np.nan,
            "mean_true_label_rank": np.nan,
            "median_true_label_rank": np.nan,
        }
    return {
        "true_label_ranks": true_label_ranks,
        "top2_accuracy": float(np.mean(true_label_ranks <= 2)),
        "top3_accuracy": float(np.mean(true_label_ranks <= 3)),
        "mean_true_label_rank": float(np.mean(finite_ranks)) if finite_ranks.size else np.nan,
        "median_true_label_rank": float(np.median(finite_ranks)) if finite_ranks.size else np.nan,
    }


def _alignment_model(alignment, *, common_classes, aligned_participants, transforms=(), target_transform=None):
    return {
        "metadata": _impl._alignment_metadata(
            alignment,
            common_classes=common_classes,
            aligned_participants=aligned_participants,
        ),
        "transforms": tuple(transforms),
        "target_transform": target_transform,
    }


def _group_average_channel_procrustes_transform(transforms):
    transforms = tuple(transforms)
    if not transforms:
        return None

    rotations = np.stack([np.asarray(transform["rotation"], dtype=float) for transform in transforms], axis=0)
    mean_rotation = np.mean(rotations, axis=0)
    left, _singular_values, right_t = np.linalg.svd(mean_rotation, full_matrices=False)
    rotation = left @ right_t
    return {
        "source_center": np.mean(
            np.stack([np.asarray(transform["source_center"], dtype=float) for transform in transforms], axis=0),
            axis=0,
        ),
        "target_center": np.mean(
            np.stack([np.asarray(transform["target_center"], dtype=float) for transform in transforms], axis=0),
            axis=0,
        ),
        "rotation": rotation,
    }


def _fitted_alignment_model(fitted_model):
    alignment_metadata = fitted_model.get("alignment_metadata", {})
    if isinstance(alignment_metadata, dict) and "metadata" in alignment_metadata:
        return alignment_metadata
    return {
        "metadata": alignment_metadata,
        "transforms": tuple(),
        "target_transform": None,
    }


def _channel_feature_mean(features, feature_set):
    channel_features = _impl._features_as_trial_channel_matrix(features, feature_set)
    return np.mean(channel_features, axis=(0, 1))


def _target_centered_channel_procrustes_transform(target_transform, features, feature_set):
    return {
        "source_center": _channel_feature_mean(features, feature_set),
        "target_center": np.asarray(target_transform["target_center"], dtype=float),
        "rotation": np.asarray(target_transform["rotation"], dtype=float),
    }


def _test_alignment_metadata(test_transform, target_centering):
    return {"test_transform": test_transform, "target_centering": target_centering}


def _align_test_features_by_subject(test_features, test_set, config, alignment_model):
    if config.alignment == "none":
        return test_features, _test_alignment_metadata("none", "none")
    if config.alignment != "train_class_procrustes":
        raise ValueError(f"Unsupported alignment: {config.alignment}")

    target_transform = alignment_model.get("target_transform")
    if target_transform is None:
        return test_features, _test_alignment_metadata("none", "none")

    test_transform = _target_centered_channel_procrustes_transform(
        target_transform,
        test_features,
        test_set,
    )
    return (
        _impl._apply_channel_procrustes_transform(test_features, test_set, test_transform),
        _test_alignment_metadata("group_average_train_transform", "target_unsupervised"),
    )


def _prediction_group_columns_with_alignment():
    columns = tuple(_impl.CROSS_SUBJECT_PREDICTION_GROUP_COLUMNS)
    additions = ("alignment_test_transform", "alignment_target_centering")
    if all(column in columns for column in additions):
        return columns
    output = []
    for column in columns:
        output.append(column)
        if column == "alignment":
            output.extend(addition for addition in additions if addition not in output)
    return tuple(output)


def _align_training_features_by_subject(feature_sets, features_by_subject, labels_by_subject, config):
    if config.alignment == "none":
        return features_by_subject, _alignment_model(
            config.alignment,
            common_classes=(),
            aligned_participants=(),
        )
    if config.alignment != "train_class_procrustes":
        raise ValueError(f"Unsupported alignment: {config.alignment}")

    common_classes = _impl._common_label_values(labels_by_subject)
    if len(common_classes) < 2:
        return features_by_subject, _alignment_model(
            config.alignment,
            common_classes=common_classes,
            aligned_participants=(),
        )

    class_patterns = [
        _impl._participant_class_channel_patterns(features, labels, feature_set, common_classes)
        for feature_set, features, labels in zip(feature_sets, features_by_subject, labels_by_subject, strict=True)
    ]
    transforms = _impl._fit_channel_procrustes_transforms(class_patterns)
    aligned_features = [
        _impl._apply_channel_procrustes_transform(features, feature_set, transform)
        for feature_set, features, transform in zip(feature_sets, features_by_subject, transforms, strict=True)
    ]
    return aligned_features, _alignment_model(
        config.alignment,
        common_classes=common_classes,
        aligned_participants=(feature_set.participant for feature_set in feature_sets),
        transforms=transforms,
        target_transform=_group_average_channel_procrustes_transform(transforms),
    )


def _score_outer_fold_model(fitted_model, test_set, config, *, include_predictions=True):
    alignment_model = _fitted_alignment_model(fitted_model)
    test_features = _impl._normalized_subject_features(test_set, config)
    test_features, test_alignment_metadata = _align_test_features_by_subject(
        test_features,
        test_set,
        config,
        alignment_model,
    )
    scoring_set = replace(test_set, features=test_features, normalization=config.normalization)
    scoring_model = dict(fitted_model)
    scoring_model["alignment_metadata"] = alignment_model["metadata"]
    outer_row, prediction_rows = _ORIGINAL_SCORE_OUTER_FOLD_MODEL(
        scoring_model,
        scoring_set,
        config,
        include_predictions=include_predictions,
    )
    outer_row["alignment_test_transform"] = test_alignment_metadata["test_transform"]
    outer_row["alignment_target_centering"] = test_alignment_metadata["target_centering"]
    for row in prediction_rows:
        row["alignment_test_transform"] = test_alignment_metadata["test_transform"]
        row["alignment_target_centering"] = test_alignment_metadata["target_centering"]
    return outer_row, prediction_rows


def _install_module_fixes():
    _impl._ranked_label_metrics = _ranked_label_metrics
    _impl._align_training_features_by_subject = _align_training_features_by_subject
    _impl._score_outer_fold_model = _score_outer_fold_model
    _impl.CROSS_SUBJECT_PREDICTION_GROUP_COLUMNS = _prediction_group_columns_with_alignment()


_install_module_fixes()

globals().update(_impl.__dict__)

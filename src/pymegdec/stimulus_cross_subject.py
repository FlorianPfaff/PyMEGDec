"""Public compatibility facade for cross-subject stimulus decoding.

The implementation lives in :mod:`pymegdec._stimulus_cross_subject_core`.
This module is kept as the stable public import path used by PyMEGDec's
CLI, tests, and downstream modules.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from pymegdec import _stimulus_cross_subject_core as _core

if TYPE_CHECKING:
    from pymegdec._stimulus_cross_subject_core import (  # noqa: F401
        ALIGNMENT_MODES,
        CROSS_SUBJECT_PREDICTION_GROUP_COLUMNS,
        DEFAULT_CROSS_SUBJECT_ALIGNMENT,
        DEFAULT_CROSS_SUBJECT_BASELINE_WINDOW,
        DEFAULT_CROSS_SUBJECT_CHANCE_CLASSES,
        DEFAULT_CROSS_SUBJECT_CLASSIFIER,
        DEFAULT_CROSS_SUBJECT_COMPONENTS_PCA,
        DEFAULT_CROSS_SUBJECT_FEATURE_MODE,
        DEFAULT_CROSS_SUBJECT_NESTED_WINDOW_CENTERS,
        DEFAULT_CROSS_SUBJECT_NORMALIZATION,
        DEFAULT_CROSS_SUBJECT_PARTICIPANTS,
        DEFAULT_CROSS_SUBJECT_WINDOW_CENTER,
        DEFAULT_CROSS_SUBJECT_WINDOW_SIZE,
        FEATURE_MODES,
        NORMALIZATION_MODES,
        CrossSubjectStimulusConfig,
        ParticipantFeatureSet,
        _align_training_features_by_subject,
        _alignment_metadata,
        _apply_channel_pattern_transform,
        _apply_channel_procrustes_transform,
        _baseline_channel_whitening_matrix,
        _baseline_feature_statistics,
        _centered_window,
        _channel_procrustes_transform,
        _common_label_values,
        _extract_window_features,
        _fit_channel_procrustes_transforms,
        _fit_outer_fold_model,
        _normalize_feature_mode,
        _normalize_features,
        _normalize_normalization,
        _normalize_trial_cap,
        _normalized_config,
        _normalized_subject_features,
        _participant_class_channel_patterns,
        _ranked_label_metrics,
        _score_outer_fold_model,
        _selected_trial_indices,
        _training_labels,
        _trial_signal,
        _trialinfo_labels,
        _true_label_ranks,
        evaluate_cross_subject_stimulus_smoke,
        evaluate_nested_cross_subject_stimulus,
        export_cross_subject_stimulus_smoke,
        export_nested_cross_subject_stimulus,
        load_participant_stimulus_features,
        make_cross_subject_candidate_configs,
        summarize_cross_subject_confusion_category_enrichment,
        summarize_cross_subject_confusion_category_matrix,
        summarize_cross_subject_confusion_pairs,
        summarize_cross_subject_predictions,
        summarize_cross_subject_stimulus_smoke,
        summarize_nested_cross_subject_stimulus,
    )

# Make imports of ``pymegdec.stimulus_cross_subject`` resolve to the core module
# object.  This keeps private helper monkey-patches and existing direct imports
# operating on the implementation module rather than on a shallow re-export copy.
sys.modules[__name__] = _core
globals().update(_core.__dict__)

"""Utilities for MEG decoding experiments."""

from pymegdec.alpha_metrics import (
    AlphaMetricConfig,
    compute_alpha_metrics,
    export_participant_alpha_metrics,
)
from pymegdec.alpha_movement import (
    AlphaMovementConfig,
    compute_alpha_movement,
    export_alpha_movement,
)
from pymegdec.alpha_movement_analysis import (
    AlphaMovementAnalysisConfig,
    analyze_alpha_movement_windows,
    export_alpha_movement_analysis,
    summarize_alpha_movement_effects,
)
from pymegdec.alpha_signal import extract_phase, extract_time_basis
from pymegdec.cross_validation import cross_validate_single_dataset
from pymegdec.data_config import DATA_DIR_ENV_VAR, resolve_data_folder
from pymegdec.model_transfer import (
    evaluate_model_transfer,
    get_original_feature_importance,
)
from pymegdec.reaction_time_analysis import (
    AlphaReactionTimeExportConfig,
    ReactionTimeCsvConfig,
    ReactionTimeUnavailableError,
    analyze_alpha_reaction_times,
    join_alpha_reaction_times,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "DATA_DIR_ENV_VAR",
    "AlphaMetricConfig",
    "AlphaMovementConfig",
    "AlphaMovementAnalysisConfig",
    "AlphaReactionTimeExportConfig",
    "ReactionTimeCsvConfig",
    "ReactionTimeUnavailableError",
    "analyze_alpha_reaction_times",
    "analyze_alpha_movement_windows",
    "compute_alpha_movement",
    "compute_alpha_metrics",
    "cross_validate_single_dataset",
    "evaluate_model_transfer",
    "export_alpha_movement",
    "export_alpha_movement_analysis",
    "export_participant_alpha_metrics",
    "extract_phase",
    "extract_time_basis",
    "get_original_feature_importance",
    "join_alpha_reaction_times",
    "resolve_data_folder",
    "summarize_alpha_movement_effects",
]

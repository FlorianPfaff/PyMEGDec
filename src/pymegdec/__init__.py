"""Utilities for MEG decoding experiments."""

from pymegdec.alpha_signal import extract_phase, extract_time_basis
from pymegdec.cross_validation import cross_validate_single_dataset
from pymegdec.data_config import DATA_DIR_ENV_VAR, resolve_data_folder
from pymegdec.model_transfer import (
    evaluate_model_transfer,
    get_original_feature_importance,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "DATA_DIR_ENV_VAR",
    "cross_validate_single_dataset",
    "evaluate_model_transfer",
    "extract_phase",
    "extract_time_basis",
    "get_original_feature_importance",
    "resolve_data_folder",
]

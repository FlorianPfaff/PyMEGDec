"""Utilities for MEG decoding experiments."""

from pymegdec.cross_validation import cross_validate_single_dataset
from pymegdec.model_transfer import evaluate_model_transfer

__all__ = [
    "cross_validate_single_dataset",
    "evaluate_model_transfer",
]

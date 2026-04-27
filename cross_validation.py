"""Backward-compatible wrapper for :mod:`pymegdec.cross_validation`."""

from pathlib import Path
import sys

_SRC = Path(__file__).resolve().parent / "src"
if _SRC.exists():
    sys.path.insert(0, str(_SRC))

from pymegdec.classifiers import train_binary_svm, train_for_stimulus_lasso_glm, train_gradient_boosting
from pymegdec.cross_validation import cross_validate_single_dataset

__all__ = [
    "cross_validate_single_dataset",
    "train_binary_svm",
    "train_for_stimulus_lasso_glm",
    "train_gradient_boosting",
]


if __name__ == "__main__":
    acc = cross_validate_single_dataset(r".", 2, classifier="multiclass-svm", components_pca=100)
    print(acc)

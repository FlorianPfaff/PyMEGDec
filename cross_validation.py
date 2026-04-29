"""Backward-compatible wrapper for :mod:`pymegdec.cross_validation`."""

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent / "src"
if _SRC.exists():
    sys.path.insert(0, str(_SRC))

from pymegdec.classifiers import (  # noqa: E402
    train_binary_svm,
    train_for_stimulus_lasso_glm,
    train_gradient_boosting,
)
from pymegdec.cross_validation import cross_validate_single_dataset  # noqa: E402

__all__ = [
    "cross_validate_single_dataset",
    "train_binary_svm",
    "train_for_stimulus_lasso_glm",
    "train_gradient_boosting",
]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Cross-validate one participant dataset."
    )
    parser.add_argument(
        "--data-dir", default=None, help="Directory containing Part*Data.mat files."
    )
    parser.add_argument(
        "--participant", type=int, default=2, help="Participant id to evaluate."
    )
    args = parser.parse_args()

    acc = cross_validate_single_dataset(
        args.data_dir, args.participant, classifier="multiclass-svm", components_pca=100
    )
    print(acc)

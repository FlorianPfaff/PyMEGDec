"""Backward-compatible wrapper for :mod:`pymegdec.model_transfer`."""

from pathlib import Path
import sys

_SRC = Path(__file__).resolve().parent / "src"
if _SRC.exists():
    sys.path.insert(0, str(_SRC))

from pymegdec.classifiers import get_default_classifier_param, train_multiclass_classifier
from pymegdec.model_transfer import evaluate_model_transfer
from pymegdec.preprocessing import (
    downsample_data,
    extract_windows,
    filter_features,
    preprocess_features,
    reduce_features_pca,
)

__all__ = [
    "MLPClassifierTorch",
    "downsample_data",
    "evaluate_model_transfer",
    "extract_windows",
    "filter_features",
    "get_default_classifier_param",
    "preprocess_features",
    "reduce_features_pca",
    "train_multiclass_classifier",
]


def __getattr__(name):
    if name == "MLPClassifierTorch":
        from pymegdec.classifiers import MLPClassifierTorch

        return MLPClassifierTorch
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if __name__ == "__main__":
    acc = evaluate_model_transfer(r".", 2, classifier="multiclass-svm", components_pca=100)
    print(acc)

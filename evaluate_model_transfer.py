"""Backward-compatible wrapper for :mod:`pymegdec.model_transfer`."""

import sys
from pathlib import Path
from typing import TYPE_CHECKING

_SRC = Path(__file__).resolve().parent / "src"
if _SRC.exists():
    sys.path.insert(0, str(_SRC))

if TYPE_CHECKING:
    from pymegdec.classifiers import MLPClassifierTorch

from pymegdec.classifiers import (  # noqa: E402
    get_default_classifier_param,
    train_multiclass_classifier,
)
from pymegdec.model_transfer import evaluate_model_transfer  # noqa: E402
from pymegdec.preprocessing import (  # noqa: E402
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
        # pylint: disable-next=no-name-in-module
        from pymegdec.classifiers import MLPClassifierTorch

        return MLPClassifierTorch
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if __name__ == "__main__":
    acc = evaluate_model_transfer(
        r".", 2, classifier="multiclass-svm", components_pca=100
    )
    print(acc)

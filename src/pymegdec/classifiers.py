"""Classifier factories and wrappers used by the decoding routines."""

from dataclasses import dataclass
from typing import Callable

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

_DEFAULT_CLASSIFIER_PARAMS = {
    "lasso": 0.005,
    "multiclass-svm": 0.5,
    "multiclass-svm-weighted": 0.5,
    "svm-binary": 0.5,
    "binary-svm": 0.5,
    "random-forest": 100,
    "gradient-boosting": 100,
    "knn": 5,
    "mostFrequentDummy": None,
    "always1Dummy": None,
    "xgboost": 100,
    "scikit-mlp": (150, 1000),
    "pytorch-mlp": {
        "hidden_dim": 720,
        "max_epochs": 500,
        "learning_rate": 1e-3,
        "dropout_rate": 0.2,
    },
}


@dataclass(frozen=True)
class ClassifierSpec:
    builder: Callable
    fits_in_builder: bool = False


def __getattr__(name):
    if name == "MLPClassifierTorch":
        from pymegdec.torch_models import MLPClassifierTorch

        return MLPClassifierTorch
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def should_use_default_classifier_param(classifier_param):
    try:
        return np.all(np.isnan(classifier_param))
    except TypeError:
        return False


def _build_multiclass_svm(_features, _labels, classifier_param, random_state):
    return make_pipeline(
        StandardScaler(),
        SVC(C=classifier_param, kernel="linear", random_state=random_state),
    )


def _build_multiclass_svm_weighted(_features, _labels, classifier_param, random_state):
    return make_pipeline(
        StandardScaler(),
        SVC(
            C=classifier_param,
            kernel="linear",
            class_weight="balanced",
            random_state=random_state,
        ),
    )


def _build_random_forest(_features, _labels, classifier_param, random_state):
    return RandomForestClassifier(
        n_estimators=int(classifier_param),
        min_samples_leaf=5,
        random_state=random_state,
    )


def _build_gradient_boosting(_features, _labels, classifier_param, random_state):
    return GradientBoostingClassifier(
        n_estimators=int(classifier_param),
        random_state=random_state,
    )


def _build_knn(_features, _labels, classifier_param, _random_state):
    return KNeighborsClassifier(n_neighbors=int(classifier_param))


def _build_most_frequent_dummy(_features, _labels, _classifier_param, _random_state):
    return DummyClassifier(strategy="most_frequent")


def _build_always_one_dummy(_features, _labels, _classifier_param, _random_state):
    return DummyClassifier(strategy="constant", constant=1)


def _build_xgboost(_features, _labels, classifier_param, random_state):
    try:
        import xgboost as xgb
    except ImportError as exc:
        raise ImportError(
            "Install PyMEGDec with the xgboost extra to use classifier='xgboost'."
        ) from exc

    return xgb.XGBClassifier(
        n_estimators=int(classifier_param),
        eval_metric="mlogloss",
        random_state=random_state,
    )


def _build_scikit_mlp(_features, _labels, classifier_param, random_state):
    return MLPClassifier(
        hidden_layer_sizes=int(classifier_param[0]),
        max_iter=int(classifier_param[1]),
        random_state=random_state,
    )


def _build_pytorch_mlp_classifier(features, labels, classifier_param, random_state):
    return _train_pytorch_mlp(
        features,
        labels,
        classifier_param,
        random_state=random_state,
    )


CLASSIFIER_REGISTRY = {
    "multiclass-svm": ClassifierSpec(_build_multiclass_svm),
    "multiclass-svm-weighted": ClassifierSpec(_build_multiclass_svm_weighted),
    "random-forest": ClassifierSpec(_build_random_forest),
    "gradient-boosting": ClassifierSpec(_build_gradient_boosting),
    "knn": ClassifierSpec(_build_knn),
    "mostFrequentDummy": ClassifierSpec(_build_most_frequent_dummy),
    "always1Dummy": ClassifierSpec(_build_always_one_dummy),
    "xgboost": ClassifierSpec(_build_xgboost),
    "scikit-mlp": ClassifierSpec(_build_scikit_mlp),
    "pytorch-mlp": ClassifierSpec(_build_pytorch_mlp_classifier, fits_in_builder=True),
}


def train_multiclass_classifier(
    features,
    labels,
    classifier,
    classifier_param,
    random_state=None,
):
    try:
        classifier_spec = CLASSIFIER_REGISTRY[classifier]
    except KeyError as exc:
        supported_classifiers = ", ".join(sorted(CLASSIFIER_REGISTRY))
        raise ValueError(
            f"Unsupported classifier: {classifier}. Supported classifiers: {supported_classifiers}"
        ) from exc

    model = classifier_spec.builder(features, labels, classifier_param, random_state)
    if classifier_spec.fits_in_builder:
        return model
    model.fit(features, labels)
    return model


def _train_pytorch_mlp(features, labels, classifier_param, random_state=None):
    if random_state is not None:
        try:
            import pytorch_lightning as pl
        except ImportError as exc:
            raise ImportError(
                "Install PyMEGDec with the torch extra to use classifier='pytorch-mlp'."
            ) from exc

        pl.seed_everything(random_state, workers=True)

    model = _build_pytorch_mlp(features, labels, classifier_param)
    train_loader, val_loader = _build_pytorch_data_loaders(
        features,
        labels,
        random_state=random_state,
    )
    trainer = _build_pytorch_trainer(classifier_param)
    trainer.fit(model, train_loader, val_loader)
    return model


def _build_pytorch_mlp(features, labels, classifier_param):
    try:
        from pymegdec.torch_models import MLPClassifierTorch
    except ImportError as exc:
        raise ImportError(
            "Install PyMEGDec with the torch extra to use classifier='pytorch-mlp'."
        ) from exc

    return MLPClassifierTorch(
        features.shape[1],
        int(classifier_param["hidden_dim"]),
        len(np.unique(labels)),
        learning_rate=classifier_param["learning_rate"],
        dropout_rate=classifier_param["dropout_rate"],
    )


def _build_pytorch_data_loaders(features, labels, random_state=None):
    try:
        import torch
        from torch.utils.data import DataLoader, TensorDataset, random_split
    except ImportError as exc:
        raise ImportError(
            "Install PyMEGDec with the torch extra to use classifier='pytorch-mlp'."
        ) from exc

    full_dataset = TensorDataset(
        torch.tensor(features, dtype=torch.float32),
        torch.tensor(labels, dtype=torch.long),
    )
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    generator = None
    if random_state is not None:
        generator = torch.Generator().manual_seed(random_state)
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=generator,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        generator=generator,
    )
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    return train_loader, val_loader


def _build_pytorch_trainer(classifier_param):
    try:
        import pytorch_lightning as pl
    except ImportError as exc:
        raise ImportError(
            "Install PyMEGDec with the torch extra to use classifier='pytorch-mlp'."
        ) from exc

    return pl.Trainer(
        max_epochs=int(classifier_param["max_epochs"]),
        default_root_dir=r"lightning_logs",
        callbacks=[pl.callbacks.EarlyStopping(monitor="val_loss", patience=10)],
    )


def train_gradient_boosting(
    train_features,
    train_labels,
    classifier_param,
    random_state=None,
):
    model = GradientBoostingClassifier(
        n_estimators=int(classifier_param),
        max_leaf_nodes=21,
        learning_rate=0.1,
        random_state=random_state,
    )
    model.fit(train_features, train_labels)
    return model


def train_for_stimulus_lasso_glm(
    train_features,
    train_labels,
    lambda_,
    random_state=None,
):
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            penalty="l1",
            C=1 / lambda_,
            solver="liblinear",
            max_iter=1000,
            random_state=random_state,
        ),
    )
    model.fit(train_features, train_labels)
    return model


def train_binary_svm(
    train_features,
    train_labels,
    box_constraint,
    random_state=None,
):
    model = make_pipeline(
        StandardScaler(),
        SVC(C=box_constraint, kernel="linear", random_state=random_state),
    )
    model.fit(train_features, train_labels)
    return model


def get_default_classifier_param(classifier):
    if classifier in _DEFAULT_CLASSIFIER_PARAMS:
        classifier_param = _DEFAULT_CLASSIFIER_PARAMS[classifier]
        if isinstance(classifier_param, dict):
            return classifier_param.copy()
        return classifier_param
    raise ValueError(f"Unsupported classifier: {classifier}")

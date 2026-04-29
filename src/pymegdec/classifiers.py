"""Classifier factories and wrappers used by the decoding routines."""

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


def __getattr__(name):
    if name == "MLPClassifierTorch":
        from pymegdec.torch_models import MLPClassifierTorch

        return MLPClassifierTorch
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def train_multiclass_classifier(features, labels, classifier, classifier_param):
    if classifier == "multiclass-svm":
        model = make_pipeline(
            StandardScaler(), SVC(C=classifier_param, kernel="linear")
        )
    elif classifier == "multiclass-svm-weighted":
        model = make_pipeline(
            StandardScaler(),
            SVC(C=classifier_param, kernel="linear", class_weight="balanced"),
        )
    elif classifier == "random-forest":
        model = RandomForestClassifier(
            n_estimators=int(classifier_param), min_samples_leaf=5
        )
    elif classifier == "gradient-boosting":
        model = GradientBoostingClassifier(n_estimators=int(classifier_param))
    elif classifier == "knn":
        model = KNeighborsClassifier(n_neighbors=int(classifier_param))
    elif classifier == "mostFrequentDummy":
        model = DummyClassifier(strategy="most_frequent")
    elif classifier == "always1Dummy":
        model = DummyClassifier(strategy="constant", constant=1)
    elif classifier == "xgboost":
        try:
            import xgboost as xgb
        except ImportError as exc:
            raise ImportError(
                "Install PyMEGDec with the xgboost extra to use classifier='xgboost'."
            ) from exc

        model = xgb.XGBClassifier(
            n_estimators=int(classifier_param),
            use_label_encoder=False,
            eval_metric="mlogloss",
        )
    elif classifier == "scikit-mlp":
        model = MLPClassifier(
            hidden_layer_sizes=int(classifier_param[0]),
            max_iter=int(classifier_param[1]),
        )
    elif classifier == "pytorch-mlp":
        model = _train_pytorch_mlp(features, labels, classifier_param)
        return model
    else:
        raise ValueError(f"Unsupported classifier: {classifier}")

    model.fit(features, labels)
    return model


def _train_pytorch_mlp(features, labels, classifier_param):
    model = _build_pytorch_mlp(features, labels, classifier_param)
    train_loader, val_loader = _build_pytorch_data_loaders(features, labels)
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


def _build_pytorch_data_loaders(features, labels):
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
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
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


def train_gradient_boosting(train_features, train_labels, classifier_param):
    model = GradientBoostingClassifier(
        n_estimators=int(classifier_param), max_leaf_nodes=21, learning_rate=0.1
    )
    model.fit(train_features, train_labels)
    return model


def train_for_stimulus_lasso_glm(train_features, train_labels, lambda_):
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            penalty="l1", C=1 / lambda_, solver="liblinear", max_iter=1000
        ),
    )
    model.fit(train_features, train_labels)
    return model


def train_binary_svm(train_features, train_labels, box_constraint):
    model = make_pipeline(StandardScaler(), SVC(C=box_constraint, kernel="linear"))
    model.fit(train_features, train_labels)
    return model


def get_default_classifier_param(classifier):
    if classifier in _DEFAULT_CLASSIFIER_PARAMS:
        classifier_param = _DEFAULT_CLASSIFIER_PARAMS[classifier]
        if isinstance(classifier_param, dict):
            return classifier_param.copy()
        return classifier_param
    raise ValueError(f"Unsupported classifier: {classifier}")

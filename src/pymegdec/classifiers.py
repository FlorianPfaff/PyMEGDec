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


def __getattr__(name):
    if name == "MLPClassifierTorch":
        from pymegdec.torch_models import MLPClassifierTorch

        return MLPClassifierTorch
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def train_multiclass_classifier(features, labels, classifier, classifier_param):
    if classifier == "multiclass-svm":
        model = make_pipeline(StandardScaler(), SVC(C=classifier_param, kernel="linear"))
    elif classifier == "multiclass-svm-weighted":
        model = make_pipeline(StandardScaler(), SVC(C=classifier_param, kernel="linear", class_weight="balanced"))
    elif classifier == "random-forest":
        model = RandomForestClassifier(n_estimators=int(classifier_param), min_samples_leaf=5)
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
            raise ImportError("Install PyMEGDec with the xgboost extra to use classifier='xgboost'.") from exc

        model = xgb.XGBClassifier(n_estimators=int(classifier_param), use_label_encoder=False, eval_metric="mlogloss")
    elif classifier == "scikit-mlp":
        model = MLPClassifier(hidden_layer_sizes=int(classifier_param[0]), max_iter=int(classifier_param[1]))
    elif classifier == "pytorch-mlp":
        model = _train_pytorch_mlp(features, labels, classifier_param)
        return model
    else:
        raise ValueError(f"Unsupported classifier: {classifier}")

    model.fit(features, labels)
    return model


def _train_pytorch_mlp(features, labels, classifier_param):
    try:
        import pytorch_lightning as pl
        import torch
        from torch.utils.data import DataLoader, TensorDataset, random_split
    except ImportError as exc:
        raise ImportError("Install PyMEGDec with the torch extra to use classifier='pytorch-mlp'.") from exc

    from pymegdec.torch_models import MLPClassifierTorch

    input_dim = features.shape[1]
    output_dim = len(np.unique(labels))
    model = MLPClassifierTorch(
        input_dim,
        int(classifier_param["hidden_dim"]),
        output_dim,
        learning_rate=classifier_param["learning_rate"],
        dropout_rate=classifier_param["dropout_rate"],
    )

    full_dataset = TensorDataset(torch.tensor(features, dtype=torch.float32), torch.tensor(labels, dtype=torch.long))
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    trainer = pl.Trainer(
        max_epochs=int(classifier_param["max_epochs"]),
        default_root_dir=r"lightning_logs",
        callbacks=[pl.callbacks.EarlyStopping(monitor="val_loss", patience=10)],
    )
    trainer.fit(model, train_loader, val_loader)
    return model


def train_gradient_boosting(train_features, train_labels, classifier_param):
    model = GradientBoostingClassifier(n_estimators=int(classifier_param), max_leaf_nodes=21, learning_rate=0.1)
    model.fit(train_features, train_labels)
    return model


def train_for_stimulus_lasso_glm(train_features, train_labels, lambda_):
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(penalty="l1", C=1 / lambda_, solver="liblinear", max_iter=1000),
    )
    model.fit(train_features, train_labels)
    return model


def train_binary_svm(train_features, train_labels, box_constraint):
    model = make_pipeline(StandardScaler(), SVC(C=box_constraint, kernel="linear"))
    model.fit(train_features, train_labels)
    return model


def get_default_classifier_param(classifier):
    if classifier == "lasso":
        return 0.005
    if classifier == "multiclass-svm":
        return 0.5
    if classifier == "multiclass-svm-weighted":
        return 0.5
    if classifier in ["svm-binary", "binary-svm"]:
        return 0.5
    if classifier == "random-forest":
        return 100
    if classifier == "gradient-boosting":
        return 100
    if classifier == "knn":
        return 5
    if classifier in ["mostFrequentDummy", "always1Dummy"]:
        return None
    if classifier == "xgboost":
        return 100
    if classifier == "scikit-mlp":
        return (150, 1000)
    if classifier == "pytorch-mlp":
        return {"hidden_dim": 720, "max_epochs": 500, "learning_rate": 1e-3, "dropout_rate": 0.2}
    raise ValueError(f"Unsupported classifier: {classifier}")

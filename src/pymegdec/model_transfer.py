import warnings

import numpy as np
import scipy.io as sio

from pymegdec.classifiers import (
    get_default_classifier_param,
    train_multiclass_classifier,
)
from pymegdec.preprocessing import preprocess_features, reduce_features_pca


# pylint: disable-next=too-many-arguments,too-many-positional-arguments,too-many-locals
def evaluate_model_transfer(
    data_folder,
    parts,
    window_size=0.1,
    train_window_center=0.2,
    null_window_center=-0.2,
    new_framerate=float("inf"),
    classifier="multiclass-svm",
    classifier_param=np.nan,
    components_pca=100,
    frequency_range=(0, float("inf")),
    return_feature_importance=False,
):

    if not isinstance(classifier_param, dict) and np.all(np.isnan(classifier_param)):
        classifier_param = get_default_classifier_param(classifier)

    train_exp_data = sio.loadmat(f"{data_folder}/Part{parts}Data.mat")["data"][0]
    val_exp_data = sio.loadmat(f"{data_folder}/Part{parts}CueData.mat")["data"][0]

    labels_train_exp = train_exp_data["trialinfo"][0][0]
    labels_val_exp = val_exp_data["trialinfo"][0][0]
    if np.isnan(null_window_center):
        # There is no null data in the validation experiment, and some
        # classifiers do not support labels starting above 0.
        labels_train_exp -= 1
        labels_val_exp -= 1

    train_sample_interval = np.diff(train_exp_data["time"][0][0][0][0, :2])
    val_sample_interval = np.diff(val_exp_data["time"][0][0][0][0, :2])
    if not np.allclose(train_sample_interval, val_sample_interval):
        raise ValueError("Sampling rate of the two experiments must match.")

    if not np.array_equal(np.unique(labels_train_exp), np.unique(labels_val_exp)):
        warnings.warn(
            "There are labels in the training or validation experiment "
            "that are not in the other experiment."
        )

    stimuli_features_train_exp, null_features_train_exp = preprocess_features(
        train_exp_data,
        frequency_range,
        new_framerate,
        window_size,
        train_window_center,
        null_window_center,
    )
    stimuli_features_val_exp, _ = preprocess_features(
        val_exp_data,
        frequency_range,
        new_framerate,
        window_size,
        train_window_center,
        np.nan,
    )

    features_train_exp = np.hstack(
        stimuli_features_train_exp + null_features_train_exp
    ).T
    labels_train_exp = np.concatenate(
        (labels_train_exp, np.zeros(len(null_features_train_exp), dtype=int))
    )

    features_val_exp = np.hstack(stimuli_features_val_exp).T

    pca_components = None
    if components_pca != float("inf"):
        features_train_exp, coeff, features_train_exp_mean, explained_variance = (
            reduce_features_pca(
                features_train_exp,
                components_pca,
            )
        )
        pca_components = coeff[:, :components_pca]
        features_val_exp = (features_val_exp - features_train_exp_mean) @ pca_components
        print(
            "Explained Variance by "
            f"{components_pca} components: {explained_variance:.2f}%"
        )

    model = train_multiclass_classifier(
        features_train_exp, labels_train_exp, classifier, classifier_param
    )
    predictions_val_exp = model.predict(features_val_exp)

    accuracy = np.mean(predictions_val_exp == labels_val_exp)
    if return_feature_importance:
        return accuracy, get_original_feature_importance(model, pca_components)

    return accuracy


def get_original_feature_importance(model, pca_components=None):
    feature_importance = _get_classifier_coefficients(model)
    if pca_components is not None:
        pca_pseudoinverse = np.linalg.pinv(pca_components)
        feature_importance = (pca_pseudoinverse.T @ feature_importance.T).T

    return feature_importance


def _get_classifier_coefficients(model):
    if hasattr(model, "coef_"):
        return model.coef_

    if hasattr(model, "steps"):
        classifier = model.steps[-1][1]
        if hasattr(classifier, "coef_"):
            coefficients = classifier.coef_
            for _, transformer in model.steps[:-1]:
                if hasattr(transformer, "scale_"):
                    coefficients = coefficients / transformer.scale_
            return coefficients

    raise ValueError(
        "Feature importance is only available for linear classifiers with coefficients."
    )


if __name__ == "__main__":
    acc = evaluate_model_transfer(
        r".",
        2,
        classifier="multiclass-svm",
        components_pca=100,
    )
    print(acc)

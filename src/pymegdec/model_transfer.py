import numpy as np
import scipy.io as sio
import warnings

from pymegdec.classifiers import get_default_classifier_param, train_multiclass_classifier
from pymegdec.preprocessing import preprocess_features, reduce_features_pca


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

    assert np.allclose(
        np.diff(train_exp_data["time"][0][0][0][0, :2]),
        np.diff(val_exp_data["time"][0][0][0][0, :2]),
    ), "Sampling rate of the two experiments must match."

    if not np.array_equal(np.unique(labels_train_exp), np.unique(labels_val_exp)):
        warnings.warn("There are labels in the training or validation experiment that are not in the other experiment.")

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

    features_train_exp = np.hstack(stimuli_features_train_exp + null_features_train_exp).T
    labels_train_exp = np.concatenate((labels_train_exp, np.zeros(len(null_features_train_exp), dtype=int)))

    features_val_exp = np.hstack(stimuli_features_val_exp).T

    if components_pca != float("inf"):
        features_train_exp, coeff, features_train_exp_mean, explained_variance = reduce_features_pca(
            features_train_exp,
            components_pca,
        )
        print(f"Explained Variance by {components_pca} components: {explained_variance:.2f}%")
        features_val_exp = (features_val_exp - features_train_exp_mean) @ coeff[:, :components_pca]

    model = train_multiclass_classifier(features_train_exp, labels_train_exp, classifier, classifier_param)
    predictions_val_exp = model.predict(features_val_exp)

    accuracy = np.mean(predictions_val_exp == labels_val_exp)
    return accuracy


if __name__ == "__main__":
    acc = evaluate_model_transfer(r".", 2, classifier="multiclass-svm", components_pca=100)
    print(acc)

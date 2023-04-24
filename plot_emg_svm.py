"""
==================================================
Plot different SVM classifiers in the iris dataset
==================================================

Comparison of different linear SVM classifiers on a 2D projection of the iris
dataset. We only consider the first 2 features of this dataset:

- Sepal length
- Sepal width

This example shows how to plot the decision surface for four SVM classifiers
with different kernels.

The linear models ``LinearSVC()`` and ``SVC(kernel='linear')`` yield slightly
different decision boundaries. This can be a consequence of the following
differences:

- ``LinearSVC`` minimizes the squared hinge loss while ``SVC`` minimizes the
  regular hinge loss.

- ``LinearSVC`` uses the One-vs-All (also known as One-vs-Rest) multiclass
  reduction while ``SVC`` uses the One-vs-One multiclass reduction.

Both linear models have linear decision boundaries (intersecting hyperplanes)
while the non-linear kernel models (polynomial or Gaussian RBF) have more
flexible non-linear decision boundaries with shapes that depend on the kind of
kernel and its parameters.

.. NOTE:: while plotting the decision function of classifiers for toy 2D
   datasets can help get an intuitive understanding of their respective
   expressive power, be aware that those intuitions don't always generalize to
   more realistic high-dimensional problems.

"""

import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn import model_selection as ms, metrics
from sklearn.inspection import DecisionBoundaryDisplay
import numpy as np
import itertools
from sklearn import svm, datasets
from sklearn.model_selection import KFold


def plot_confusion_matrix(predicted_labels_list, y_test_list,
                          class_names=np.array(['Calf', 'Bicep', 'Thumb', 'Masseter', 'Eye'])):
    cnf_matrix = metrics.confusion_matrix(y_test_list, predicted_labels_list)
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plt.figure()
    generate_confusion_matrix(cnf_matrix, classes=class_names, title='Confusion matrix, without normalization')
    # plt.show()

    # Plot normalized confusion matrix
    plt.figure()
    generate_confusion_matrix(cnf_matrix, classes=class_names, normalize=True, title='Normalized confusion matrix')
    plt.show()


def generate_confusion_matrix(cnf_matrix, classes, normalize=False, title='Confusion matrix'):
    if normalize:
        cnf_matrix = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cnf_matrix, interpolation='nearest', cmap=plt.get_cmap('Blues'))
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cnf_matrix.max() / 2.

    for i, j in itertools.product(range(cnf_matrix.shape[0]), range(cnf_matrix.shape[1])):
        plt.text(j, i, format(cnf_matrix[i, j], fmt), horizontalalignment="center",
                 color="white" if cnf_matrix[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    return cnf_matrix


def evaluate_multi_emg_svm_model(emg_data, emg_data_label, C=1.0, is_feature=False):
    k_fold = KFold(4, shuffle=True, random_state=1)
    predicted_targets = np.array([])
    actual_targets = np.array([])
    accuracy_list = list()
    X = emg_data
    y = emg_data_label
    for train_ix, test_ix in k_fold.split(X):
        X_train, y_train, X_test, y_test = X[train_ix], y[train_ix], X[test_ix], y[test_ix]
        clf = svm.SVC(kernel="rbf", gamma=0.7, C=C)
        clf.fit(X_train, y_train)
        predicted_labels = clf.predict(X_test)
        accuracy = metrics.accuracy_score(y_test, predicted_labels)
        predicted_targets = np.append(predicted_targets, predicted_labels)
        actual_targets = np.append(actual_targets, y_test)
        accuracy_list.append(accuracy)
    return predicted_targets, actual_targets, accuracy_list



#
def multi_emg_svm(emg_data, emg_data_label, C=1.0, is_feature=False):
    X = emg_data
    y = emg_data_label
    X_train, X_test, y_train, y_test = ms.train_test_split(X, y, train_size=0.75)
    models = (
        svm.LinearSVC(C=C, max_iter=100000),
        svm.SVC(kernel="rbf", gamma=0.7, C=C),
    )
    titles = (
        "LinearSVC (linear kernel)",
        "SVC with RBF kernel",
    )
    predicts = []
    models_dup = tuple()
    for clf_m in models:
        models_dup = models_dup + (clf_m.fit(X_train, y_train),)
        predicts.append(clf_m.predict(X_test))
        if is_feature:
            print(f"No of pca components seen during fit: {clf_m.n_features_in_}")
        else:
            print(f"No of features seen during fit: {clf_m.n_features_in_}")


    for predict, title in zip(predicts, titles):
        print("\n--------------------------------------")
        print(f"Confusion Matrix ({title}):\n {metrics.confusion_matrix(y_test, predict)}")
        print(f"Accuracy Score ({title}): {metrics.accuracy_score(y_test, predict)}")

def emg_svm(emg_data, emg_data_label, C=1.0, is_feature=False):
    X = emg_data
    y = emg_data_label
    # C: SVM regularization parameter
    X_train, X_test, y_train, y_test = ms.train_test_split(X, y, train_size=0.8)
    models = (
        svm.SVC(kernel="linear", C=C),
        svm.LinearSVC(C=C, max_iter=100000),
        svm.SVC(kernel="rbf", gamma=0.7, C=C),
        svm.SVC(kernel="poly", degree=3, gamma="auto", C=C),
    )
    # models = (clf.fit(X, y) for clf in models)
    # models = (clf.fit(X_train, y_train) for clf in models)
    # models = ((clf.fit(X_train, y_train), clf.predict(X_test)) for clf in models)
    predicts = []
    models_dup = tuple()
    for clf_m in models:
        models_dup = models_dup + (clf_m.fit(X_train, y_train),)
        predicts.append(clf_m.predict(X_test))

    models = models_dup
    titles = (
        "SVC with linear kernel",
        "LinearSVC (linear kernel)",
        "SVC with RBF kernel",
        "SVC with polynomial (degree 3) kernel",
    )

    # Set-up 2x2 grid for plotting.
    # plt.figure()
    fig, sub = plt.subplots(2, 2)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    X0, X1 = X[:, 0], X[:, 1]
    if is_feature:
        xlabel, ylabel = "feature 1", "feature 2"
    else:
        xlabel, ylabel = "PCA component 1", "PCA component 2"
    for clf, title, ax in zip(models, titles, sub.flatten()):
        disp = DecisionBoundaryDisplay.from_estimator(
            # clf[0],
            clf,
            X_train,
            response_method="predict",
            cmap=plt.cm.coolwarm,
            alpha=0.8,
            ax=ax,
            xlabel=xlabel,
            ylabel=ylabel,
        )
        ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors="k")
        ax.set_xticks(())
        ax.set_yticks(())
        if is_feature:
            ax.set_xlim([0.2, 0.8])
            ax.set_ylim([0.05, 0.5])
        else:
            ax.set_xlim([-0.2, 0.4])
            ax.set_ylim([-0.2, 0.4])
        ax.set_title(title)

        # print(pred)
    # for clf_m in models:
    for predict, title in zip(predicts, titles):
        print("\n--------------------------------------")
        print(f"Confusion Matrix ({title}):\n { metrics.confusion_matrix(y_test, predict)}")
        print(f"Accuracy Score ({title}): {metrics.accuracy_score(y_test, predict)}")
        # print(f"Precision Score ({title}): {metrics.precision_score(y_test, predict)}")
        # print(f"f1 Score ({title}): {metrics.f1_score(y_test, predict)}")

    plt.show()

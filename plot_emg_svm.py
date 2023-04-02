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
from sklearn.inspection import DecisionBoundaryDisplay
#

def emg_svm(emg_data, emg_data_label, C=1.0):
    X = emg_data
    y = emg_data_label
    # C: SVM regularization parameter
    models = (
        svm.SVC(kernel="linear", C=C),
        svm.LinearSVC(C=C, max_iter=10000),
        svm.SVC(kernel="rbf", gamma=0.7, C=C),
        svm.SVC(kernel="poly", degree=3, gamma="auto", C=C),
    )
    models = (clf.fit(X, y) for clf in models)

    # title for the plots
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

    for clf, title, ax in zip(models, titles, sub.flatten()):
        disp = DecisionBoundaryDisplay.from_estimator(
            clf,
            X,
            response_method="predict",
            cmap=plt.cm.coolwarm,
            alpha=0.8,
            ax=ax,
            xlabel="PCA component 1",
            ylabel="PCA component 2",
        )
        ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors="k")
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_xlim([-0.2, 0.35])
        ax.set_ylim([-0.2, 0.35])
        ax.set_title(title)

    plt.show()

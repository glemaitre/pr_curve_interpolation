import numpy as np


def davis_goadrich_interpolation(precision, recall, positive, num=100):
    """Interpolation of the precision-recall curve according to
    Davis and Goadrich [1]_.

    Davis and Goadrich propose an hyperbolic interpolation to define the
    precision-recall curve. Values on the precision-recall curve are interpolated such
    as in the equation of section 4. of [1]_.

    Parameters
    ----------
    precision : ndarray of shape (n_samples,)
        Precision values as returned by :func:`~sklearn.metrics.precision_recall_curve`.

    recall : ndarray of shape (n_samples,)
        Recall values as returned by :func:`~sklearn.metrics.precision_recall_curve`.

    positive : int
        The total number of positive samples in the dataset used when calling
        :func:`~sklearn.metrics.precision_recall_curve`.

    num : int, default=100
        Number of points to interpolate between sub-integral true positive values.

    References
    ----------
    [1] Davis, Jesse, and Mark Goadrich.
        "The relationship between Precision-Recall and ROC curves."
        Proceedings of the 23rd international conference on Machine learning.
        2006.
    """
    # Remove the last value of precision and recall as it corresponds to the
    # point (0, 1) and will lead to a division by zero during interpolation
    # We add this value back later.
    precision, recall = precision[:-1], recall[:-1]
    # Invert also the order to have an increasing number of true positive later on
    precision, recall = precision[::-1], recall[::-1]
    true_positive = (recall * positive).round()
    false_positive = ((recall * positive) * (1 - precision) / precision).round()

    indices = np.arange(len(true_positive))

    def recall_interpolation(true_positive_start, true_positive_stop, positive, num):
        return np.linspace(true_positive_start, true_positive_stop, num=num) / positive

    def precision_interpolation(
        true_positive_start,
        true_positive_stop,
        false_positive_start,
        false_positive_stop,
        num,
    ):
        x = np.linspace(true_positive_start, true_positive_stop, num=num)
        increment_denominator = (
            (false_positive_stop - false_positive_start)
            / (true_positive_stop - true_positive_start)
            * (x - true_positive_start)
        )
        return x / (x + false_positive_start + increment_denominator)

    # We use a double list to concatenate all the element afterwards
    # Add back the PR point corresponding to the point (0, 1)
    recall_interp, precision_interp = [[0.0]], [[1.0]]
    for indices_start, indices_stop in zip(indices[:-1], indices[1:]):
        # Interpolate sub-integral values
        true_positive_start = true_positive[indices_start]
        true_positive_stop = true_positive[indices_stop]
        false_positive_start = false_positive[indices_start]
        false_positive_stop = false_positive[indices_stop]
        if np.isclose(true_positive_start, true_positive_stop):
            # Discontinuity in the PR curve where true positive is constant
            # and false positive is increasing
            continue

        recall_interp.append(
            recall_interpolation(
                true_positive_start,
                true_positive_stop,
                positive,
                num,
            )
        )
        precision_interp.append(
            precision_interpolation(
                true_positive_start,
                true_positive_stop,
                false_positive_start,
                false_positive_stop,
                num,
            )
        )

    # Revert the order to be consistent with the output of precision_recall_curve
    return np.hstack(precision_interp)[::-1], np.hstack(recall_interp)[::-1]

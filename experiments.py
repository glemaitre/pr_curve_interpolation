# %% [markdown]
#
# ## Synthetic example
#
# This example is a synthetic examples to check the potential benefit of the
# interpolation method proposed by Davis & Goadrich [1]_.

# %%
from collections import Counter

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve

from interpolation import davis_goadrich_interpolation

X, y = make_classification(n_samples=50, class_sep=0.1, random_state=42)
estimator = LogisticRegression().fit(X, y)
precision, recall, thresholds = precision_recall_curve(
    y, estimator.decision_function(X)
)

counter = Counter(y)
precision_interp, recall_interp = davis_goadrich_interpolation(
    precision, recall, counter[1], num=1000
)

# %%
import matplotlib.pyplot as plt
from sklearn.metrics import PrecisionRecallDisplay

fig, ax = plt.subplots(figsize=(8, 8))
PrecisionRecallDisplay.from_estimator(
    estimator, X, y, ax=ax, label="constant (step-post)"
)
PrecisionRecallDisplay.from_estimator(
    estimator, X, y, ax=ax, drawstyle="default", label="linear", linestyle="-."
)
ax.plot(recall_interp, precision_interp, label="davis goadrich", linestyle="--")
ax.legend(loc="lower left")
fig.savefig("comparison_toy.svg")

# %%
import numpy as np
from sklearn.metrics import average_precision_score

print(
    "Average precision score: \n"
    f"{average_precision_score(y, estimator.decision_function(X))}"
)

print(
    "Integral from the precision-recall with linear interpolation by step of 1 TP: \n"
    f"{-np.trapz(precision, recall)}"
)

print(
    "Integral from the precision-recall with Davis & Goadrich interpolation with "
    f"sub-integral precision: \n{-np.trapz(precision_interp, recall_interp)}"
)

# %% [markdown]
#
# ## Example from original paper
#
# This example reproduce the example of Fig. 6 from the original paper [1]_.

# %%
positive, negative = 433, 56_164
recall = np.array([0, 9 / positive, 1])[::-1]
precision = np.array([1, 1, positive / (positive + negative)])[::-1]

precision_interp, recall_interp = davis_goadrich_interpolation(
    precision, recall, positive, num=1000
)

# %%
fig, ax = plt.subplots()
ax.plot(
    recall, precision, label=f"linear - AU-PRC = {-np.trapz(precision, recall):.3f}"
)
ax.plot(
    recall_interp,
    precision_interp,
    label=(
        "Davis & Goadrich - AU-PRC = "
        f"{-np.trapz(precision_interp, recall_interp):.3f}"
    ),
)
ax.legend()
ax.set_title("Reproduction of Fig. 6 from Davis & Goadrich (2006)")
fig.savefig("reproduction_figure_6.svg")

# %% [markdown]
#
# ## References
#
# [1] Davis, Jesse, and Mark Goadrich.
#     "The relationship between Precision-Recall and ROC curves."
#     Proceedings of the 23rd international conference on Machine learning.
#     2006.

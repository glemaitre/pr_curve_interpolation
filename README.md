# Precision-recall curve interpolation

This is an experiment to check the value of sub-integral interpolation in when
computing the area under the PR curve.

The file `interpolation.py` implements an interpolation strategy as in
Davis & Goadrich. The file `experiments.py` compare the current scikit-learn
stragtegy and evalute the impact of interpolating for sub-integral values
of true positives.

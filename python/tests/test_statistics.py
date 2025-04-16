"""@author: Andrés Corrada-Emmanuel."""

# from ntqr import Labels
from ntqr.statistics import MClassifiersVariables

labels = ("ax", "by")
classifiers = ("ci", "cj")


def test_n1_r2_variables():
    vars = MClassifiersVariables(labels, classifiers[:1])

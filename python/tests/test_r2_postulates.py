"""@author: Andrés Corrada-Emmanuel."""
import pytest
import sympy

from ntqr.r2.datasketches import (
    TrioVoteCounts,
    TrioLabelVoteCounts,
    classifier_label_votes,
)

from ntqr.r2.examples import uciadult_label_counts


@pytest.mark.parametrize(
    "label_accuracies, voting_frequencies",
    (
        (1, sympy.Rational(18432653, 1357332964)),
        (1, sympy.Rational(18272925, 1357332964)),
        (1, sympy.Rational(16803485, 1357332964)),
    ),
)
def test_one_classifier_postulate(label_accuracies, voting_frequencies):
    pass

"""
Evaluate noisy decision makers using logic and algebra.

Classes:

Functions:

Misc variables:

    __version__
    uci_adult_test_example
"""

__version__ = "0.6.1"
from typing_extensions import Iterable, Sequence


from ntqr.r2.examples import uciadult_label_counts

from ntqr.r2.datasketches import TrioLabelVoteCounts, TrioVoteCounts

from ntqr.r2.evaluators import (
    SupervisedEvaluation,
    ErrorIndependentEvaluation,
    MajorityVotingEvaluation,
)

# import ntqr.statistics
# import ntqr.r2
# import ntqr.r3


class Label(str):
    """
    Label object to guarantee a label is stringifiable.

    """

    def __init__(self, label):
        """


        Parameters
        ----------
        label : str
            Label to appear in variable subscripts.

        Returns
        -------
        None.

        """
        self._label = label

    def __str__(self):
        return self._label


class Labels(tuple):
    """
    Labels used in test question responses. The NTQR package assumes that
    all test questions have the same, fixed, label set. These are a
    tuple-like object so the user can specify the canonical order of
    the labels.

    The number of labels defines the integer R in the acronym NTQR.
    Binary classification: R=2
    Three label classification: R=3, etc.
    """

    def __init__(self, labels: Iterable[str]):
        """


        Parameters
        ----------
        labels : Iterable[str]
           Labels for question responses.

        Returns
        -------
        None.

        """
        self._labels = tuple([Label(label) for label in labels])


class AlignedDecisions(tuple):
    """
    The atomic fact we have about the decisions of test takers is
    how they agree and disagree on question responses. There are
    many ways we could do this comparison. If we imagine the columns
    to be each test taker and the rows to be each question, the grid
    comparisons are combinatorial in size.

    To get individual performance statistics we align test takers
    responses/decisions by question. The resulting tuple of decisions
    is then of size N = number of classifiers being aligned with each
    other.
    """

    def __init__(self, decisions: Sequence[Label]):
        """


        Parameters
        ----------
        decisions : Sequence[Label]
            The question-aligned responses of one or more test takers.

        Returns
        -------
        None.

        """
        self._decisions = tuple(decisions)

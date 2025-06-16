"""
Module for classes and datastructures related to test sketches.

The logic of unsupervised evaluation can be viewed as similar to
data streaming algorithms. We use summary statistics of a test
(observable agreements and disagreements between classifiers)
to deduce other statistics of the test (the set of possible group
evaluations for them).
"""

from typing import Self
from typing_extensions import Mapping, Sequence
from itertools import combinations, product

from ntqr import Label, Labels


class QuestionAlignedDecisions:
    """
    The question aligned test sketch.

    It can be shown algebraically that the count of question aligned
    responses by the test takers are "generated" by statistics of
    their correctness on the test. These sample statistics of a given
    test include their individual average performance as well as
    sample statistics tracking their error correlations with each
    other. This coupling of observed test sketch and polynomial system
    of unknown sample statistics of test takers is the core idea of the
    NTQR package.

    This class represents the observable side of that
    coupling - the counts of the R^N ways that N test takers
    can agree and disagree when they are taking a test with
    R responses.
    """

    def __init__(
        self, observed_responses: Mapping[Sequence[Label], int], labels: Labels
    ):
        """
        Validate test responses and transform into question aligned counts.

        Parameters
        ----------
        observed_responses : Mapping[Sequence[Label], int]
            Observed counts of agreements and disagreements between
            N test takers responding to questions with R possible
            responses. This can contain at most R^N keys.
        labels : Labels
            The R labels.

        Returns
        -------
        None.

        """
        # It must have some keys
        if len([key for key in observed_responses.keys()]) == 0:
            raise ValueError("There must be at least one decision key.")

        # All the keys must be the same length
        key_lengths = list([len(key) for key in observed_responses.keys()])

        if not (len(set(key_lengths)) == 1):
            raise ValueError("Not all decision tuples have the same length.")

        # Fill up a new dictionary with AlignedDecisions keys
        self.N = len(list(observed_responses.keys())[0])

        if not all(
            [
                (decision in product(labels, repeat=self.N))
                for decision in observed_responses.keys()
            ]
        ):
            raise ValueError(
                """One or more decisions in 'observed_responses'
                contain label(s) not in 'labels' arg."""
            )

        self.counts = {
            decisions: observed_responses.get(decisions, 0)
            for decisions in product(labels, repeat=self.N)
        }

        # And remember the labels used in the decisions keys
        self.labels = labels

    def marginalize(self, indices: Sequence[int]) -> Self:
        """
        Marginalize counts to 'indices' only counts.

        Parameters
        ----------
        indices : Sequence[int]
            Subset of classifiers.

        Returns
        -------
        Self
            Question aligned counts for subset of classifiers specified
            by 'indices'.

        """
        new_counts = {}
        for key, count in self.counts.items():
            new_key = tuple([key[i] for i in indices])
            new_counts[new_key] = new_counts.setdefault(new_key, 0) + count

        return QuestionAlignedDecisions(new_counts, self.labels)

    def m_subset_indices_to_val(self, m: int) -> Mapping[tuple, int]:
        """
        Response counts for all possible classifier subsets of size 'm'.

        Parameters
        ----------
        m : int
            Size of the subsets of the aligned decisions to use.

        Returns
        -------
        Mapping from m-sized subsets indices to observed response counts.

        """
        if m > self.N:
            raise ValueError(
                "Size of m-subsets, m, cannot be larger than the number "
                + "of test-takers, N"
            )

        indices_to_val = {
            m_subset: self.marginalize(m_subset)
            for m_subset in combinations(range(self.N), m)
        }

        return indices_to_val

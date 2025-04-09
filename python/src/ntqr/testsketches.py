# -*- coding: utf-8 -*-
from typing_extensions import Collection, Mapping, Sequence
from itertools import product

from ntqr import Label, Labels, AlignedDecisions


class QuestionAlignedDecisions:
    """
    In unsupervised settings there is no answer key by which to
    evaluate test takers. All that we have available is their
    agreements and disagreements on correct responses to questions
    in the test.

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
        The observed_responses keys must be sequences of labels all of
        the same length since they come from aligning N test takers.

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
        # All the keys must be the same length
        assert len(set([len(key) for key in observed_responses.keys()])) == 1

        # Fill up a new dictionary with AlignedDecisions keys
        N = len(list(observed_responses.keys())[0])

        assert all(
            [
                self.decision_in_possible_set(
                    product(labels, repeat=N), decision
                )
                for decision in observed_responses.keys()
            ]
        )

        self.counts = {
            decisions: observed_responses.get(decisions, 0)
            for decisions in product(labels, repeat=N)
        }

    def decision_in_possible_set(
        self,
        possible_set: Collection[Sequence[Label]],
        decision: Sequence[Label],
    ) -> bool:
        """


        Parameters
        ----------
        possible_set : Collection[Sequence[Label]]
            All possible decision tuples.
        decision : Sequence[Label]
            Decision tuple

        Returns
        -------
        bool
            The decision is in the possible set.

        """
        return decision in possible_set

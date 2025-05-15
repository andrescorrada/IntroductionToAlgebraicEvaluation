"""@author: Andrés Corrada-Emmanuel."""

from itertools import combinations, product
from types import MappingProxyType
from typing_extensions import Iterable, Literal, Mapping, Sequence, Union

import sympy

from ntqr import Labels
from ntqr.statistics import MClassifiersVariables


class MAxiomsIdeal:
    """
    Class for generating the axioms related an (M=m)-sized subsets
    of the classifiers.

    Each subset of the classifiers has a set of axioms, of size R,
    involving the marginalized decisions of that subset. For each
    value of the size of a subset, M, we can establish universal
    relations between counts of observed M-sized decision tuples
    and the counts of those tuples and smaller ones given true labels.

    The bottom of the M axioms ladder is M=1, the individual axioms.
    These carve out the evaluations for a given test taker given their
    marginalized response counts. This is a subset of the R-simplex,
    for an individual test taker, for any test of size Q. Thus, each
    test-taker in a group of N gets their own M=1 set of axioms.

    The M=2 axioms correspond to all possible pairs in a group of N
    test takers. These involve all the variables in the  M=1 axioms
    but now define a new set of axioms involving the pair response
    variables.

    In R-space, the space of integer response counts, all these
    ideals are linear equations. Thus, it may seem that calling them
    'ideals', while strictly true, is overkill. However, in P-space,
    these ideals are polynomials of degree 2 or higher.
    """

    def __init__(
        self,
        labels: Labels,
        classifiers: Sequence[str],
        m: int = 1,
    ):
        """
        The M=m axioms given answer labels and classifier labels.

        Parameters
        ----------
        labels : Labels
            The R possible label answers to the Q questions.
        classifiers : Sequence[str]
            Labels for indexing classifiers.
        m : int, optional
            M=m axioms index. The default is 1.

        Returns
        -------
        None.

        """
        self.labels = labels

        # There is a an axiomatic ideal for every subset of the
        # classifiers of size m.
        self._m_complex = {}
        for m_subset in combinations(classifiers, m):
            # An 'ideal' in this software needs to keep track
            # of equations and ways to refer to var symbols in it.
            # Narrowly speaking, this ideal is not the logical ideal.
            # The logical ideal includes all the equations in this one
            # but also all the equations in all of its subsets.
            # This is an algebraic ideal that tells us the new equations
            # that have to be obeyed by m-decision vars - no more.
            subset_m_ideal = self._m_complex.setdefault(m_subset, {})

            # Managing the vars is the hardest part.
            vars = subset_m_ideal.setdefault("vars", {})
            # We need the new variables associated with m-sized decision
            # tuples.
            vars.update({m_subset: MClassifiersVariables(labels, m_subset)})
            #
            for m_smaller in range(1, m):
                vars.update(
                    {
                        m_small_subset: MClassifiersVariables(
                            labels, m_small_subset
                        )
                        for m_small_subset in combinations(m_subset, m_smaller)
                    }
                )

            match m:
                case 1:
                    axiomatic_ideal = self._m_one_ideal(labels, m_subset)
                case 2:
                    axiomatic_ideal = self._m_two_ideal(m_subset)
                case _:
                    raise ValueError(
                        "Only up to M=2 axiom ideals are currently supported."
                    )

            subset_m_ideal["axioms"] = axiomatic_ideal

        self.m_complex = MappingProxyType(self._m_complex)

    def _m_one_ideal(self, labels, m_subset):
        """


        Parameters
        ----------
        labels : TYPE
            DESCRIPTION.
        vars : TYPE
            DESCRIPTION.

        Returns
        -------
        axioms_by_label : TYPE
            DESCRIPTION.

        """
        vars = self._m_complex[m_subset]["vars"][m_subset]
        qs = vars.qs
        responses = vars.responses
        responses_by_label = vars.label_responses

        axioms_by_label = {
            l_true: sympy.UnevaluatedExpr(
                sum(qs[label] for label in self.labels if label != l_true)
                + sum(
                    var
                    for var in responses_by_label[l_true]["errors"].values()
                )
                - sum(
                    responses[(label,)]
                    for label in self.labels
                    if label != l_true
                )
                - sum(
                    label_responses["errors"][(l_true,)]
                    for label, label_responses in responses_by_label.items()
                    if label != l_true
                )
            )
            for l_true in labels
        }

        return axioms_by_label

    def _m_two_ideal(self, pair):
        """
        The M=2 axiom constructor.

        Parameters
        ----------
        labels : TYPE
            DESCRIPTION.
        vars : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        labels = self.labels
        pair_vars = self._m_complex[pair]["vars"][pair]
        qs = pair_vars.qs
        m2_responses = pair_vars.responses
        m2_label_responses = pair_vars.label_responses

        # Now we have to build the variables for m1 decision
        # tuples.
        m1_responses = {
            m1: self._m_complex[pair]["vars"][m1].responses
            for m1 in combinations(pair, 1)
        }
        m1_label_responses = {
            m1: self._m_complex[pair]["vars"][m1].label_responses
            for m1 in combinations(pair, 1)
        }

        m2_axioms_ideal = {
            l_true: (
                sympy.simplify(
                    sum([qs[q_label] for q_label in qs if q_label != l_true])
                    + sum(
                        var
                        for var in m2_label_responses[l_true][
                            "errors"
                        ].values()
                    )
                    + sum(
                        var
                        for error_pair, var in m2_label_responses[l_true][
                            "errors"
                        ].items()
                        if error_pair[0] != error_pair[1]
                    )
                    # The single response terms
                    - sum(
                        m1_responses[m1][(l_error,)]
                        for m1 in combinations(pair, 1)
                        for l_error in labels
                        if l_error != l_true
                    )
                    + sum(
                        var
                        for m1 in combinations(pair, 1)
                        for l_error in labels
                        for le2, le2_responses in m1_label_responses[
                            m1
                        ].items()
                        for decisions, var in le2_responses["errors"].items()
                        if (l_error != l_true)
                        and (le2 != l_true)
                        and (decisions == (l_error,))
                    )
                    - sum(
                        var
                        for m1 in combinations(pair, 1)
                        for l_error in labels
                        for le2, le2_responses in m1_label_responses[
                            m1
                        ].items()
                        for decisions, var in le2_responses["errors"].items()
                        if ((l_error != l_true) and (le2 != l_true))
                    )
                    # The m2 terms
                    + sum(
                        m2_responses[(l_error, l_error)]
                        for l_error in labels
                        if l_error != l_true
                    )
                    #
                    - sum(
                        var
                        for l_error in self.labels
                        for l_error2 in self.labels
                        for decisions, var in m2_label_responses[l_error2][
                            "errors"
                        ].items()
                        if (l_error != l_true) and (l_error2 != l_true)
                    )
                    + sum(
                        var
                        for l_error in labels
                        for var in m2_label_responses[l_error][
                            "errors"
                        ].values()
                        if l_error != l_true
                    )
                )
            )
            for l_true in labels
        }

        return m2_axioms_ideal


class NAxiomsComplex:
    """
    Class that recursively builds all the M-axiom ideals for all non-empty
    subsets of N test-takers answering Q questions with R choices.
    """

    def __init__(self):
        pass

"""@author: Andrés Corrada-Emmanuel."""

from itertools import combinations, product
from typing import Any
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

    def _m_two_ideal(
        self, pair: tuple[Any, Any]
    ) -> dict[str, sympy.UnevaluatedExpr]:
        """
        The M=2 axiom constructor.

        This is the 'errors' variables version. The one used for internal
        computation of the varieties. It has the drawback that it is hard
        to check and code.

        Parameters
        ----------
        pair : Sequence[Any, Any]
            Pair of classifiers.

        Returns
        -------
        m2_axioms_ideal : Mapping[label, sympy.UnevaluatedExpr]
            A mapping from label to its corresponding r-axiom.

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
                        if (
                            (error_pair[0] != error_pair[1])
                            and (error_pair[0] != l_true)
                            and (error_pair[1] != l_true)
                        )
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
                        for decisions, var in m1_label_responses[m1][l_error][
                            "errors"
                        ].items()
                        if (l_error != l_true)
                        and (decisions != (l_true,))
                        and (decisions != (l_error,))
                    )
                    - sum(
                        var
                        for m1 in combinations(pair, 1)
                        for l_error in labels
                        for decisions, var in m1_label_responses[m1][l_error][
                            "errors"
                        ].items()
                        if ((l_error != l_true) and (decisions != (l_error,)))
                    )
                    # The m2 terms
                    + sum(
                        m2_responses[(l_error, l_error)]
                        for l_error in labels
                        if l_error != l_true
                    )
                    #
                    + sum(
                        var
                        for l_error in self.labels
                        for l_error2 in self.labels
                        for decisions, var in m2_label_responses[l_error][
                            "errors"
                        ].items()
                        if (l_error != l_true)
                        and (decisions != (l_error, l_error))
                    )
                    - sum(
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

    def m_two_ideal_agreement_representation(
        self, pair: tuple[Any, Any]
    ) -> Mapping[str, sympy.UnevaluatedExpr]:
        """
        The M=2 axioms ideal with agreement label response variables.

        The axioms in label response space are easiest to understand
        and prove when we use 'agreement' variables. Those are variables
        where all the classifiers are agreeing in their responses on the
        true label.

        Starting with M=2 this will be the only way the axioms will be
        encoded from now on. The representation with 'errors' variables
        only, the one needed for generalizable code, will be created with
        the straightforward transformation that gives the all agree on the
        true label as the number of questions of that label minus all the
        other possible responses.

        Parameters
        ----------
        pair : tuple[Any, Any]
            DESCRIPTION.

        Returns
        -------
        m2_axioms_ideal : TYPE
            DESCRIPTION.

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
            l_true: sympy.simplify(
                -m2_label_responses[l_true][(l_true, l_true)]
                + qs[l_true]
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
                    for decisions, var in m1_label_responses[m1][l_error][
                        "errors"
                    ].items()
                    if (l_error != l_true) and (decisions != (l_true,))
                )
                # The m2 terms
                + sum(
                    m2_responses[(l_error, l_error)]
                    for l_error in labels
                    if l_error != l_true
                )
                - sum(
                    m2_label_responses[l_e2][(l_e1, l_e1)]
                    for l_e1 in self.labels
                    for l_e2 in self.labels
                    if (l_e1 != l_true) and (l_e2 != l_true)
                )
                + sum(
                    var
                    for l_e1 in self.labels
                    for l_e2 in self.labels
                    for decisions, var in m2_label_responses[l_true][
                        "errors"
                    ].items()
                    if (l_e1 != l_e2) and (l_e1 != l_true) and (l_e2 != l_true)
                )
            )
            for l_true in labels
        }

        return m2_axioms_ideal

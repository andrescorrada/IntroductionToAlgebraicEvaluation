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
        self.classifiers = classifiers
        self.m = m

        # We get all the variables associated with these classifiers
        # and reorganize them to map m_subset -> "vars" -> vars
        self.mvars = MClassifiersVariables(labels, classifiers)
        self._m_complex = {}
        for m_size in range(1, m + 1):
            for m_subset in combinations(classifiers, m_size):
                self._m_complex[m_subset] = {
                    "label_rvars": self.mvars.label_responses[m_subset],
                    "rvars": self.mvars.responses[m_subset],
                }

        self.all_agree_subs_dict = self.initialize_all_agree_subs_dict()

        # Now we compute the variety for all subsets of the classifiers of
        # size m.
        for m_subset in combinations(classifiers, m):

            match m:
                case 1:
                    axiomatic_ideal = self.m_one_ideal_agreement(m_subset)
                case 2:
                    axiomatic_ideal = self.m_two_ideal_agreement(m_subset)
                case 3:
                    ideal_cr = self.m_three_ideal_agreement_representation(
                        m_subset
                    )

                    axiomatic_ideal = {
                        label: sympy.simplify(
                            axiom.subs(self.all_agree_subs_dict)
                        )
                        for label, axiom in ideal_cr.items()
                    }
                case _:
                    raise ValueError(
                        "Only up to M=2 axiom ideals are currently supported."
                    )

            self._m_complex[m_subset]["axioms"] = axiomatic_ideal

        self.m_complex = MappingProxyType(self._m_complex)

    def m_one_ideal_agreement(
        self, classifier: tuple[str]
    ) -> Mapping[str, sympy.UnevaluatedExpr]:
        """
        The M=1 axioms ideal with agreement label response variables.

        The axioms in label response space are easiest to understand
        and prove when we use 'agreement' variables. Those are variables
        where all the classifiers are agreeing in their responses on the
        true label.

        Parameters
        ----------
        classifier : tuple[str]
            DESCRIPTION.

        Returns
        -------
        m1_axioms_ideal :  Mapping[str, sympy.UnevaluatedExpr]
            A mapping from label to its corresponding m=1 axiom.

        """
        m1_responses = self.mvars.responses[classifier]
        m1_label_responses = self.mvars.label_responses[classifier]
        qs = self.mvars.qs

        m1_axioms_ideal = {
            l_true: sympy.simplify(
                -m1_label_responses[l_true][(l_true,)]
                + qs[l_true]
                # The single response terms
                - sum(
                    m1_responses[(l_error,)]
                    for l_error in self.labels
                    if l_error != l_true
                )
                + sum(
                    m1_label_responses[l_e2][(l_e1,)]
                    for l_e1 in self.labels
                    for l_e2 in self.labels
                    if (l_e1 != l_true) and (l_e2 != l_true)
                )
            )
            for l_true in self.labels
        }

        return m1_axioms_ideal

    def _m_one_ideal(self, labels, m_subset):
        """
        This may need to be deprecated.

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
        qs = self.mvars.qs
        responses = self._m_complex[m_subset]["rvars"]
        responses_by_label = self._m_complex[m_subset]["label_rvars"]

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
        Construct the m=2 ideal.

        This is the 'errors' variables version. The one used for internal
        computation of the varieties. It has the drawback that it is hard
        to check or to write code for it.

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
        qs = self.mvars.qs
        m2_responses = self._m_complex[pair]["rvars"]
        m2_label_responses = self._m_complex[pair]["label_rvars"]

        # Now we have to build the variables for m1 decision
        # tuples.
        m1_responses = {
            m1: self._m_complex[m1]["rvars"] for m1 in combinations(pair, 1)
        }
        m1_label_responses = {
            m1: self._m_complex[m1]["label_rvars"]
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
                        m1_label_responses[m1][l_e1][(l_e2,)]
                        for m1 in combinations(pair, 1)
                        for l_e1 in labels
                        for l_e2 in labels
                        if (l_e1 != l_true)
                        and (l_e2 != l_true)
                        and (l_e1 != l_e2)
                    )
                    - sum(
                        m1_label_responses[m1][l_e1][(l_e2,)]
                        for m1 in combinations(pair, 1)
                        for l_e1 in labels
                        for l_e2 in labels
                        if ((l_e1 != l_true) and (l_e2 != l_e1))
                    )
                    # The m2 terms
                    + sum(
                        m2_responses[(l_error, l_error)]
                        for l_error in labels
                        if l_error != l_true
                    )
                    #
                    - sum(
                        m2_label_responses[l_e2][(l_e1, l_e1)]
                        for l_e1 in labels
                        for l_e2 in labels
                        if (l_e1 != l_true)
                        and (l_e1 != l_e2)
                        and (l_e2 != l_true)
                    )
                    + sum(
                        var
                        for l_e1 in self.labels
                        for var in m2_label_responses[l_e1]["errors"].values()
                        if (l_e1 != l_true)
                    )
                )
            )
            for l_true in labels
        }

        return m2_axioms_ideal

    def m_two_ideal_agreement(
        self, pair: tuple[Any, Any]
    ) -> Mapping[str, sympy.UnevaluatedExpr]:
        """
        Construct the M=2 algebraic ideal.

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
            The classifier pair.

        Returns
        -------
        m2_axioms_ideal : Mapping[str, sympy.UnevaluatedExpr]
            Mapping from labels to its corresponding M=2 axiom.

        """
        labels = self.labels
        qs = self.mvars.qs
        m2_responses = self.mvars.responses[pair]
        m2_label_responses = self.mvars.label_responses[pair]

        # Now we have to build the variables for m1 decision
        # tuples.
        m1_responses = {
            m1: self.mvars.responses[m1] for m1 in combinations(pair, 1)
        }
        m1_label_responses = {
            m1: self.mvars.label_responses[m1] for m1 in combinations(pair, 1)
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
                    m1_label_responses[m1][l_e2][(l_e1,)]
                    for m1 in combinations(pair, 1)
                    for l_e1 in labels
                    for l_e2 in labels
                    if (l_e1 != l_true) and (l_e2 != l_true)
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
                    m2_label_responses[l_true]["errors"][(l_e1, l_e2)]
                    for l_e1 in self.labels
                    for l_e2 in self.labels
                    if (l_e1 != l_e2) and (l_e1 != l_true) and (l_e2 != l_true)
                )
            )
            for l_true in labels
        }

        return m2_axioms_ideal

    def m_three_ideal_agreement_representation(
        self, trio: tuple[str | int, str | int, str | int]
    ) -> Mapping[str, sympy.UnevaluatedExpr]:
        """
        Construct the M=3 algebraic ideal with agreement variables.

        Parameters
        ----------
        trio : tuple[str | int, str | int]
            Any three classifiers.

        Returns
        -------
        m2_axioms_ideal : Mapping[str, sympy.UnevaluatedExpr]
            Dictionary from label to its corresponding axiom.

        """
        labels = self.labels
        qs = self.mvars.qs
        m3_responses = self.mvars.responses[trio]
        m3_label_responses = self.mvars.label_responses[trio]

        # Now we have to build the variables for m1 decision
        # tuples.
        m1_responses = {
            m1: self.mvars.responses[m1] for m1 in combinations(trio, 1)
        }
        m1_label_responses = {
            m1: self.mvars.label_responses[m1] for m1 in combinations(trio, 1)
        }

        # Now we have to build the variables for m2 decision
        # tuples.
        m2_responses = {
            m1: self.mvars.responses[m1] for m1 in combinations(trio, 2)
        }
        m2_label_responses = {
            m1: self.mvars.label_responses[m1] for m1 in combinations(trio, 2)
        }

        m3_axioms_ideal = {
            l_true: sympy.simplify(
                -m3_label_responses[l_true][(l_true, l_true, l_true)]
                + qs[l_true]
                # The single response terms
                - sum(
                    m1_responses[m1][(l_error,)]
                    for m1 in combinations(trio, 1)
                    for l_error in labels
                    if l_error != l_true
                )
                + sum(
                    m1_label_responses[m1][l_e2][(l_e1,)]
                    for m1 in combinations(trio, 1)
                    for l_e1 in labels
                    for l_e2 in labels
                    if (l_e1 != l_true) and (l_e2 != l_true)
                )
                # The m2 terms
                + sum(
                    m2_responses[pair][(l_error, l_error)]
                    for pair in combinations(trio, 2)
                    for l_error in labels
                    if l_error != l_true
                )
                - sum(
                    m2_label_responses[pair][l_e2][(l_e1, l_e1)]
                    for pair in combinations(trio, 2)
                    for l_e1 in self.labels
                    for l_e2 in self.labels
                    if (l_e1 != l_true) and (l_e2 != l_true)
                )
                # The m3 terms
                - sum(
                    m3_responses[(l_error, l_error, l_error)]
                    for l_error in labels
                    if l_error != l_true
                )
                + sum(
                    m3_label_responses[l_e2][(l_e1, l_e1, l_e1)]
                    for l_e1 in self.labels
                    for l_e2 in self.labels
                    if (l_e1 != l_true) and (l_e2 != l_true)
                )
                # Two do not agree
                + sum(
                    m3_label_responses[l_true]["errors"][(l_e1, l_e2, l_e3)]
                    for l_e1 in self.labels
                    for l_e2 in self.labels
                    for l_e3 in self.labels
                    if len(set((l_e1, l_e2, l_e3, l_true))) == 3
                )
                # Three do not agree
                + 2
                * sum(
                    m3_label_responses[l_true]["errors"][(l_e1, l_e2, l_e3)]
                    for l_e1 in self.labels
                    for l_e2 in self.labels
                    for l_e3 in self.labels
                    if len(set((l_e1, l_e2, l_e3, l_true))) == 4
                )
            )
            for l_true in labels
        }

        return m3_axioms_ideal

    def initialize_all_agree_subs_dict(self) -> None:
        """
        Initialize the substitution dictionary for all correct responses.

        Computations of the label response simplex variables is easiest
        when we represent all agree on the correct label in terms of
        the disagreeing label responses variables. This function
        initializes the substitution dictionary that can be used to
        turn any expression in terms of label response variables into
        one that only uses disagreement ones.

        Returns
        -------
        None
            The self.all_correct_subs_dict is initialized.

        """
        subs_dict = {}
        for m_subset, m_subset_dict in self._m_complex.items():
            for l_true in self.labels:
                var = m_subset_dict["label_rvars"][l_true][
                    tuple([l_true for i in range(len(m_subset))])
                ]
                subs_dict[var] = self.mvars.qs[l_true] - sum(
                    m_subset_dict["label_rvars"][l_true]["errors"].values()
                )

        return subs_dict

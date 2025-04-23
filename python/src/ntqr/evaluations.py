"""Module for super classes of R-axioms based evaluations.

Logically consistent evaluations are defined by the size of an ensemble
and its associated axioms.

"""

from collections.abc import Sequence
from itertools import combinations, filterfalse, product
from types import MappingProxyType
from typing_extensions import Sequence, Mapping

import sympy

from ntqr import Labels
from ntqr.testsketches import QuestionAlignedDecisions
from ntqr.statistics import MClassifiersVariables
from ntqr.raxioms import MAxiomsIdeal
import ntqr.r2.raxioms
import ntqr.r3.raxioms


class AnswerKeyQSimplex:
    """
    Class to generate all possible number of labels in an
    answer key given the test size, Q, and the labels. This
    is a set of tuples of dimension R, the number of labels.

    It exists in (R-1) space since, by construction, we
    must have their sum always equal to Q. The number of
    possible qs is 1/(R-1)!*(Q+1)*(Q+2)*...*(Q+R-1).

    To summarize:
        - The number of labels in the answer key is a tuple
        of size R. Its value is unknown in unsupervised settings.

        - The number of such tuples is 1/(R-1)!(Q+1)...

        - This set exists in a (R-1) dimensional space inside
          the R space for the tuples.
    """

    def __init__(self, Q: int, labels: Labels):
        """


        Parameters
        ----------
        Q : int
            The number of question in the test.
        labels : Labels
            The labels.

        Returns
        -------
        None.

        """
        self.Q = Q
        self.N = Q + 1

        # The code is not safe or reliable beyond R=3 in
        # the current version.
        R = len(labels)
        if R > 3:
            raise NotImplementedError(
                "Only values of R=2,3 supported currently."
            )

        self.R = R

    def qs(self):
        """
        All possible number of label counts in an answer key
        of size Q.

        Returns
        -------
        iter[tuple[int]]
            DESCRIPTION.

        """
        Q = self.Q
        match self.R:
            case 2:
                qs = [(qa, Q - qa) for qa in range(self.N)]
            case 3:
                qs = [
                    (qa, qb, Q - qa - qb)
                    for qa in range(self.N)
                    for qb in range(self.N - qa)
                ]
        return qs


class MLabelResponseSimplexes:
    """
    Class to manage the response simplexes associated with each label.

    Each subset of sized-M for N classifiers has its own set of label
    response simplexes, one for each of the R labels in the test. These
    are the set of all possible values for statistics of aligned decisions
    by the members of the m-subset GIVEN true label.

    The a-priori logic of a test is that the sum of all possible responses
    by a m-subset must exactly equal to the Q_label, the count of the label
    in the answer key. This defines a simplex for a given label.

    The posterior logic is that the simplexes for any given value of M,
    M=m, have axioms that depend on all simplexes of value less than m.
    Thus the M=2 simplexes involve variables from the two M=1 simplexes
    that come from the classifier pair responses. The M=3 simplexes involve
    all M=2 simplexes, and all M=1 simplexes. And so on.

    This class is meant to internally manage that logical complexity
    by keeping track of all the variables that are needed for each
    simplex as well as the enclosing 'shells' that provide the values
    for the response variables.

    A separate class, MSubsetLabelResponseVarieties, combines the
    functionality of this class and the ntqr.raxioms.MAxiomsIdeal
    class to compute the subset of possible label test responses that
    are logically consistent with the observed test results.

    """

    def __init__(
        self,
        labels: Sequence[str],
        classifiers: Sequence[str],
        responses: Mapping[tuple, int],
        qs,
        m,
    ):
        self.labels = labels
        self.qs = qs
        self.classifiers = classifiers

        self.qad = QuestionAlignedDecisions(responses, labels)
        # We must have as many classifier labels as the length of
        # the response decision tuples.
        if len(classifiers) != self.qad.N:
            err_msg = (
                "There must be as many classifiers({}) as the length of the"
                + " response decision keys ({})."
            )
            raise ValueError(err_msg.format(len(classifiers), self.qad.N))
        # Currently only supporting up to m=1.
        if m == 0:
            raise ValueError(
                "Only M=1 axioms label simplexes supported currently."
            )

        self.qs = qs

        # We need to manage two sets of variables, the ones for the
        # observed test responses, let's call them 'responses', and
        # ones for those observed responses GIVEN true label. Call
        # those 'label_responses'
        # In addition, we want to set values for those vars. So
        # everything is organized by the decision tuples of an m-subset
        self.response_vars = {}
        # In an unsupervised setting we have access to the response values
        # after the test is taken.
        self.response_vals = {}
        # In unsupervised settings we only know these vars are needed.
        self.label_response_vars = {}
        #

        # Everything is organized by subsets of the classifier labels.
        # Then, the response vars then get pointed to by m-sized
        # decision tuples:
        #    m-subset -> m-decision tuple -> (var or val).
        # For label response vars:
        #    m-subset -> label -> m-decision tuples
        # We need all the 1-decision vars, the 2-decision vars,
        # up to m
        for m_current in range(1, m + 1):
            m_response_vals = self.m_responses(m_current)
            for m_subset in combinations(classifiers, m_current):
                m_subset_vars = MClassifiersVariables(labels, m_subset)
                self.response_vars.update({m_subset: m_subset_vars.responses})
                self.label_response_vars.update(
                    {m_subset: m_subset_vars.label_responses}
                )
                self.response_vals[m_subset] = m_response_vals[m_subset]

        # Now we construct the evaluation dict needed by SymPy to evaluate
        # the axioms
        self.response_eval_dict = {
            var: val
            for var, val in zip(
                [
                    var
                    for m_subset, m_decisions_dict in self.response_vars.items()
                    for var in m_decisions_dict.values()
                ],
                [
                    val
                    for m_subset, m_decisions_dict in self.response_vals.items()
                    for val in m_decisions_dict.values()
                ],
            )
        }
        # And add the q_s values
        self.q_vars = MappingProxyType(
            {label: sympy.Symbol(r"Q_" + label) for label in labels}
        )
        self.response_eval_dict.update(
            {self.q_vars[label]: val for label, val in zip(labels, self.qs)}
        )

        self.test_axioms = MappingProxyType(self.instantiate_axioms(m))

    def m_responses(self, m: int) -> Mapping[tuple, Mapping[tuple, int]]:
        """
        Observed responses by all m-sized subsets of the classifiers.

        Parameters
        ----------
        m : int
            Size of the subsets of the classifiers that will be used.

        Returns
        -------
        Mapping[tuple, Mapping[tuple, int]]
            This is semantically of the form,
                Mapping[m-subset-classifiers,
                    Mapping[m-decisions, observed count]
            A tuple that identifies the m-sized subset of N classifiers
            points to a mapping of question aligned decisions by that
            subset to the observed integer count in the test.

        """
        classifiers = self.classifiers
        # This code is somewhat confusing because of the convenience for
        # users to input test responses by a tuple ordering instead of
        # subscripting decisions. '(a,b)'  is easier to write than
        # set(a_i, b_j). It seems using the tuple is overall better
        # for immutability transparency in Python. Using set(...) would
        # not work and we would need immutable sets.
        # Hence, we define m_subsets by their indices into the decision
        # tuples in the user provided test responses.
        m_subsets = combinations(range(len(classifiers)), m)
        m_subsets_responses = self.qad.m_subset_indices_to_val(m)
        # And now we want to return a dictionary that maps from
        # the m-subset-indices to their decision tuple test counts.
        ret_val = {
            tuple([classifiers[i] for i in m_subset]): m_subsets_responses[
                m_subset
            ].counts
            for m_subset in m_subsets
        }

        return ret_val

    def label_m_decisions_var_range(self, classifiers, decisions, label):
        """


        Parameters
        ----------
        classifiers : TYPE
            DESCRIPTION.
        decisions : TYPE
            DESCRIPTION.
        label : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """

    def instantiate_axioms(self, m: int):
        """
        Fills in the the observed response variables in all the M-axioms
        from 1 to m.

        Parameters
        ----------
        m : int
            DESCRIPTION.

        Returns
        -------
        Mapping[m_subset:instantiated_m_axioms]

        """

        axioms = {
            m_subset: {
                label: MAxiomsIdeal(self.labels, m_subset, m_current)
                .m_complex[m_subset]["axioms"][label]
                .subs(self.response_eval_dict)
                for label in self.labels
            }
            for m_current in range(1, m + 1)
            for m_subset in combinations(self.classifiers, m_current)
        }

        return axioms


class MAxiomsVarieties:
    """
    Class to compute the test evaluation variety.

    The test evaluation variety for M=m axioms is the set of
    evaluations that are logically consistent with how we observe
    the classifiers agreeing and disagreeing on the question
    responses.
    """

    def __init__(
        self,
        labels: Sequence[str],
        classifiers: Sequence[str],
        responses: Mapping[tuple, int],
        qs,
        m,
    ):
        self.r_simplexes = MLabelResponseSimplexes(
            labels, classifiers, responses, qs, m
        )

        # Since we have the responses and the value of on the
        # Q-simplex, qs, we can construct all the axioms needed
        # to define the evaluation ideals.

        self.varieties = {}
        for curr_m in range(1, m + 1):
            pass


class MAxiomsVariety:
    """
    Class to compute an M-subset evaluation variety given the M-axioms and
    all M=m smaller.

    The logical aspect of NTQR is that evaluation varieties occur for
    nested sets of ideals. At any M<=N, there are a new set of algebraic
    axioms involving M-response variables for the M-sized subset of the
    classifiers we may want to consider.

    Varieties at a given M involve the set of points in the new M-sized
    response simplexes for each label. But they also eliminate points in
    smaller m-sized varieties. This cascading elimination is the very
    mechanism whereby we reduce the set of possible evaluations for all
    classifiers.

    Said another way, the union of smaller m-sized decision varities is larger
    than or equal to the one when we impose the M-axioms. Given values of
    label responses by smaller subsets, the supposed state of the answer key,
    qs, there may be no integer values for the M-decisions label responses
    that satisfy the M-axioms.

    The class is incomplete for this release. That trimming logic still
    under development. Currently it provides:
        1. The M=1 evaluation varieties. These are the varieties lying in
        the space of 1-decision tuples for each label.
        2. It instantiates all the axioms given test responses.

    """

    def __init__(
        self,
        classifiers: tuple,
        r_simplexes: MLabelResponseSimplexes,
        m_minus_one_varieties: dict,
        m: int,
    ):
        pass


class SingleClassifierEvaluations:
    """
    Base class for the evaluations of a single classifier
    given its test responses. This set is a subset of all
    possible evaluations.

    The axioms of unsupervised evaluation are algebraic filters
    that narrow the set of possible evaluations for test takers.
    This filtering proceeds in a ladder-like manner. The single
    classifier axioms are the bottom rung of that filtering.

    For a test with R labels, the set of possible evaluations
    is a collection of
    has dimension R*(R-1) at each setting of the number of label
    correct in the answer key. This is the space of error responses.
    There being R-1 for each of the R labels. The single classifier
    axioms cut that dimension
    by R-1. So the set of evaluations for a single classifier consists
    of points that have dimension R*(R-1), but the set itself has
    dimension (R-1)*(R-1).


    """

    def __init__(
        self,
        Q: int,
        single_axioms: (
            ntqr.r2.raxioms.SingleClassifierAxioms
            | ntqr.r3.raxioms.SingleClassifierAxioms
        ),
    ):
        self.Q = Q
        self.axioms = single_axioms

    def correct_at_qs(
        self, qs: Sequence[int], responses: Sequence[int]
    ) -> set[Sequence[int]]:
        """Calculate all possible correct responses given qs.

        Parameters
        ----------
        qs : Sequence[int]
            Number of label questions.
        responses : Sequence[int]
            Label responses by classifier.

        Returns
        -------
        set[Sequence[int]]
            Set of possible label correct evaluations given qs and responses.
        """
        errors_at_qs = self.errors_at_qs(qs, responses)
        return set(
            ((ql - sum(label_errors)) for ql, label_errors in zip(qs, errors))
            for errors in errors_at_qs
        )

    def _check_axiom_consistency_(self, eval_dict, wrong_vars, wrong_vals):
        eval_dict.update(
            {var: val for var, val in zip(wrong_vars, wrong_vals)}
        )
        return self.axioms.satisfies_axioms(eval_dict)


class EvaluationCuboid:
    """
    The basic geometric object in establishing the evaluations logically
    consistent with test takers responses is the cuboid of responses
    that satisfy all the single classifier axioms given the assumed number
    of correct responses.

    This cuboid has dimension N (the number of test takers) and contains
    at most (Q_label + 1)^N possible values. Its actual size is smaller
    once we filter out group evaluations inconsistent with the single
    classifier axioms.

    For tests with R possible question responses, there are R of these
    cuboids at given setting of (Q_l1, Q_l2, ..., Q_lR).
    """

    def __init__(
        self, observed_responses: QuestionAlignedDecisions, labels: Labels
    ):
        """


        Parameters
        ----------
        observed_responses : QuestionAlignedResponses
            The question aligned test sketch for the N test takers.
        labels : Labels
            The R possible labels for question responses.

        Returns
        -------
        None.

        """

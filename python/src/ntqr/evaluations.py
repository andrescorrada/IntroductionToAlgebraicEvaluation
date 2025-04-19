"""Module for super classes of R-axioms based evaluations.

Logically consistent evaluations are defined by the size of an ensemble
and its associated axioms.

"""

from collections.abc import Sequence
from itertools import combinations, filterfalse, product
from typing_extensions import Sequence, Mapping

import sympy

from ntqr import Labels
from ntqr.testsketches import QuestionAlignedDecisions
from ntqr.statistics import MClassifiersVariables
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


class MAxiomsLabelSimplexes:
    """
    Class for the trees of label response simplexes that are
    rooted at each of the M subsets of N test-takers and continue
    to the individual classifiers in the subset as leaves.
    """

    def __init__(
        self,
        labels: Sequence[str],
        classifiers: Sequence[str],
        responses: Mapping[tuple, int],
        qs,
        m,
    ):
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

        self.m_simplexes = {}
        self.responses = {}
        for m_current in range(1, m + 1):
            curr_m_responses = self.m_responses(m)
            # We now flatten the complicated dictionary of observed
            # responses by creating a flat dictionary that has
            # m-response variables pointing to their observed count
            # in the test.
            self.responses[m_current] = {
                m_subset: {
                    var: m_subset_counts[decisions]
                    for decisions, var in MClassifiersVariables(
                        labels, m_subset
                    ).responses.items()
                }
                for m_subset, m_subset_counts in curr_m_responses.items()
            }

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


class ClassifierQsSimplexes:
    """
    Super class for the possible evaluations of a single classifier
    taking a test with R possible responses. The classes currently
    implemented are:
        - ntqr.r2.evaluations.ClassifierQsSimplexes
        - ntqr.r3.evaluations.ClassifierQsSimplexes

    Given the number of label responses in a test by a classifier
    and assumed value for the number of labels in the answer key,
    (Q_a, Q_b, ...),  we can establish the set of values in the
    R-dimensional simplex of each label.

    There are R of these R-dimensional spaces! By test apriori
    logic, the only possible evaluations are in an (R-1)-dimensional
    simplex for each of thes R spaces.

    The single classifier axioms connect the values across these R
    spaces for each true label. Since there are R of these equations
    and they are linear in R-spaces, this defines a surface in the
    R(R-1)-dimensional space defined by the 'error' variables.

    For example, in R=2 tests, the possible number of correct responses
    for a classifier given Q_a and Q_b would be inside the square
    defined by [0,Q_a]x[0,Q_b]. The single classifier axiom narrows
    this to a line through this square.

    Using the error variables, a point in the simplex can be represented
    by a tuple of length 2 with each element a tuple of length 1:
        ((R_{b_i}_a,) (R_{a_i}_b,))

    For R=3 tests points in the simplex have the form:
        ( (R_{b_i}_a, R_{c_i}_a),
          (R_{a_i}_b, R_{c_i}_b),
          (R_{a_i}_c, R_{b_i}_c) )

    This simplex defines the largest possible set of evaluations
    for a classifier given its test responses. Observations of
    agreements and disagreements with other test takers cannot
    be used to add to this set - only filter some out. Therefore,
    this class serves as the base class for computing logically
    consistent group evaluations.
    """

    def __init__(
        self,
        qs: Sequence[int],
        aligned_decisions: QuestionAlignedDecisions,
        classifier: int = 0,
    ):
        """


        Parameters
        ----------
        qs : Sequence[int]
            DESCRIPTION.
        aligned_decisions : QuestionAlignedDecisions
            Counts of observed question aligned decision tuples. Each
            decision tuple is of the same size N, each position indexing
            the decision of a classifier.
        classifier : int, optional
           Index to the classifier for the decision keys in aligned_decisions.
           The default is 0.
        Returns
        -------
        None.

        """

        # Only


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

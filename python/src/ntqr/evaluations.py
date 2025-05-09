"""Module for super classes of R-axioms based evaluations.

Logically consistent evaluations are defined by the size of an ensemble
and its associated axioms.

"""

from collections.abc import Sequence
from itertools import combinations, product, tee
from types import MappingProxyType
from typing import Iterable
from typing_extensions import Mapping

import sympy

from ntqr import Labels
from ntqr.testsketches import QuestionAlignedDecisions
from ntqr.statistics import MClassifiersVariables
from ntqr.raxioms import MAxiomsIdeal
import ntqr.r2.raxioms
import ntqr.r3.raxioms


class AnswerKeyQSimplex:
    """
    Class to generate all answer-key simplex points.

    Given the test size, Q, and the labels (R=r) of them. This
    is a set of tuples of dimension r, the number of labels.

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
        Class initializer.

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
        self.labels = labels
        R = len(labels)
        self.R = R

    def qs(self) -> Iterable[tuple[int]]:
        """
        Generate all possible answer-key simplex points.

        Returns
        -------
        Iterable[tuple[int]]
            Generator of all possible answer-key simplex points.

        """
        ranges = (range(0, self.N) for label in self.labels)
        answer_key_simplex_points = filter(
            lambda x: sum(x) == self.Q, product(*ranges)
        )
        for point in answer_key_simplex_points:
            yield point


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

    A separate class, MAxiomsVarieties, combines the functionality
    of this class and the ntqr.raxioms.MAxiomsIdeal class to compute
    the subset of possible label test responses that are logically
    consistent with the observed test results.

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
        if m > 2:
            raise NotImplementedError("Only M=1,2 axioms supported currently.")

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
        self.labels = labels
        self.classifiers = classifiers
        self.qs = qs

        self.r_simplexes = MLabelResponseSimplexes(
            labels, classifiers, responses, qs, m
        )

        # Since we have the responses and the value of qs on the
        # Q-simplex, we can construct all the axioms needed
        # to define the evaluation ideals.

        # We instantiate the test axioms at this qs. This will
        # make them linear equations that only contain label
        # response variables.
        self.test_axioms = MappingProxyType(self.instantiate_axioms(m))

        self.varieties = {}
        for curr_m in range(1, m + 1):
            pass

    def label_mm1_vars(
        self, classifiers: Sequence[str], decisions: Sequence[str], label: str
    ) -> Iterable[sympy.Symbol]:
        """
        Generate label vars for these classifiers and their decisions.

         To properly calculate the max integer value of m-decisions given
         true label, we need to know the variables associated with
         (m-1)-sized subsets of those decisions. This function generates
         them.


         Parameters
         ----------
         classifiers : Sequence[str]
             The classifiers, there are m of them.
         decisions : Sequence[str]
             The m-decision tuple.
         label : str
             The true label.

         Returns
         -------
         Iteratable[sympy.Symbol]

        """
        m = len(classifiers)
        mm1_subsets = combinations(classifiers, m - 1)
        mm1_decisions = combinations(decisions, m - 1)

        for mm1_subset, mm1_decision in zip(mm1_subsets, mm1_decisions):
            yield self.label_response_vars[mm1_subset][label]["errors"][
                mm1_decision
            ]

    def label_mdecision_max(
        self,
        classifiers: Sequence[str],
        decisions: Sequence[str],
        label: str,
        ql: int,
        mm1_varieties_point: Iterable[dict],
    ) -> int:
        """
        Calculate the 'ratchet of crowd evaluation'.

        Every label response var is a positive integer between zero and,
        at most, the Q_label assumed value. However, this max is most
        of the time larger than the logically consistent solutions.

        The max integer value is the minimum of:
            1. Q_label
            2. R_decisions
            3. all R_decisions_label for the m (m-1)-sized subsets of
               the classifier decisions.

        Clearly the max of label response variables must be Q_label. But
        if we observe the classifiers collectively producing a lower count,
        then that is the max. For example, if we never observed a pair count
        for (l_1, l_2) then all label response vars for this tuple must be
        zero for all labels.

        The same applies to the (m-1)-subsets of the decisions tuple given
        the true label. No value of the response count for the m-sized
        decisions tuple can be higher than any response count for the
        (m-1)-sized subsets of the decisions.

        Parameters
        ----------
        classifiers : Sequence[str]
            The m classifiers.
        decisions : Sequence[str]
            Their m-decisions.
        label : str
            The true label.
        *mm1_varieties_point : Iterable[dict]
            Generator of the m-1 varities.

        Returns
        -------
        int
            Max integer value for these classifier decisions given
            true label and the m-1 varieties point.

        """

        # The m-1 vars must be turned into values
        response_val = self.r_simplexes.response_vals[classifiers][decisions]
        return min(ql, response_val)

    def label_response_simplex_points(self, ql, vars, maxs):
        """
        Generator of the allowed label response simplex points.

        Parameters
        ----------
        ql : int
            Assumed count of true label in the answer key.
        vars : Sequence[Sympy.Symbol]
            The label response variables for this simplex.
        maxs : Sequence[int]
            The maximum integer value for the label response vars.

        Returns
        -------
        Generator of dictionaries specifying each simplex point.

        """
        ranges = (range(0, max + 1) for max in maxs)
        all_points = product(*ranges)
        for point in filter(lambda x: sum(x) <= ql, all_points):
            yield {var: val for var, val in zip(vars, point)}

    def variety(self, classifiers: Sequence[str]) -> Iterable[dict]:
        """
        Generate the points in the variety for the classifiers.



        Parameters
        ----------
        classifiers : Sequence[str]
            The classifiers.
        qs: Sequence[int]
            The assumed count of labels in the answer key.

        Returns
        -------
        Iterable[dict]
            DESCRIPTION.

        """
        m = len(classifiers)
        been_here_before = False

        # Root case
        if m == 0:
            for i in range(1):
                yield tuple()
        else:
            # We need the points in the mm1_varieties for these classifiers
            mm1_varieties = (
                self.variety(mm1_subset)
                for mm1_subset in combinations(classifiers, m - 1)
            )
            for mm1_point in product(*mm1_varieties):
                mvars_only_axiom_exprs = self.mvars_only_axioms(
                    classifiers, mm1_point
                )
                # We have r of these label simplexes
                labels_m_simplexes = (
                    self.label_msimplex(classifiers, label, ql, mm1_point)
                    for label, ql in zip(self.labels, self.qs)
                )
                # A point in this space is a tuple of r label simplexes,
                # one for each true label.
                variety_points = filter(
                    lambda x: self.satisfies_axioms(
                        classifiers, x, mvars_only_axiom_exprs
                    ),
                    product(*labels_m_simplexes),
                )
                for point in variety_points:
                    yield self.make_variety_point(
                        classifiers, point, mm1_point
                    )

    def make_variety_point(
        self,
        classifiers: Sequence[str],
        mpoint: Iterable[dict],
        mm1_point_dicts: Iterable[dict],
    ) -> dict:
        """
        Make dict specifying an m-variety point for these classifiers.

        Parameters
        ----------
        classifiers : Sequence[str]
            The classifiers for this variety.
        mpoint : Iterable[dict]
            The dicts, one per label, specifying the label m-response
            vars values.
        mm1_point_dicts : Iterable[dict]
            The collection of m (m-1)-varieties for these classifiers.
            This represents a single point in the (m-1)-varieties.


        Returns
        -------
        point : dict
            Dictionary with all the label response variables required
            for the m-varity of these classifiers.

        """
        point = {}

        for label_vars_dict in mpoint:
            point.update(label_vars_dict)

        for mm1_point in mm1_point_dicts:
            point.update(mm1_point)

        return point

    def satisfies_axioms(
        self,
        classifiers: Sequence[str],
        mpoint: Iterable[dict],
        maxioms: Sequence[sympy.UnevaluatedExpr],
    ) -> bool:
        """
        Test if mpoint satisfies maxioms.

        The variety is defined by all those points in the label
        response simplexes that satisfy the m-axioms.

        Parameters
        ----------
        classifiers : Sequence[str]
            The classifiers being considered.
        mpoint : Iterable[dict]
            Iteratable of label response point dicts.
        maxioms : Sequence[sympy.UnevaluatedExpr]
            The m-axioms reduced to only contain label m-response vars.

        Returns
        -------
        bool
            DESCRIPTION.

        """
        merged_dict = {}
        for label_dict in mpoint:
            merged_dict.update(label_dict)

        return all(
            (sympy.simplify(axiom.subs(merged_dict)) == 0) for axiom in maxioms
        )

    def mvars_only_axioms(
        self, classifiers: Sequence[str], mm1_point: Iterable[dict]
    ) -> Sequence[sympy.UnevaluatedExpr]:
        """
        Reduce the test axioms to only m-decision vars.

        The logic works from the bottom up. To know what
        possible values the labels m-response vars can have,
        we must know what at which point we are in the
        (m-1)-dimensional varieties associated with all
        (m-1)-sized subsets of m classifiers.

        This function takes the dicts associated with each
        of the (m-1) varieties and reduces the test axioms
        using their var:val items. The result are linear
        equations that only contain unresolved label m-response
        variables. These are the expressions that will
        then define the variety for label m-response variables.

        The present implementation most certainly has a bug
        for m > 2 responses. The issue comes down to two varieties
        not agreeing on what values at m - 2 are allowed. This
        expression only becomes positive for m >= 3.

        The bug is a software one, not a logical one.

        Parameters
        ----------
        mm1_point : Iterable[dict]
            DESCRIPTION.

        Returns
        -------
        The test axioms with only unresolved label m-response vars.

        """
        n_axioms = len(self.test_axioms[classifiers])
        point_generators = tee(mm1_point, n_axioms)
        return [
            test_axiom.subs(self.join_point_dicts(point))
            for test_axiom, point in zip(
                self.test_axioms[classifiers].values(), point_generators
            )
        ]

    def join_point_dicts(self, point_dicts: Iterable[dict]) -> dict:
        """
        Merge the dictionaries for the point subspaces.

        Helper function to merge the dictionaries for the m-1 points.
        This is probably where the logic to create a merge should
        reside for m > 2.

        Parameters
        ----------
        point_dicts : Iterable[dict]
            The point dictionaries that will be merged.

        Returns
        -------
        Merged dictionary.

        """
        merged_dict = {}
        for mm1_point in point_dicts:
            merged_dict.update(mm1_point)
        return merged_dict

    def label_msimplex(
        self,
        classifiers: Sequence[str],
        label: str,
        ql: int,
        mm1_point: Iterable[dict],
    ) -> Iterable[dict]:
        """
        Generate all label m-response vars points.

        Each label has an m-response simplex that must
        be logically consistent with lower m response varieties.

        Parameters
        ----------
        classifiers : Sequence[str]
            The classifiers for this m-response simplex.
        label : str
            The true label.
        ql : int
            Assumed count of the true label in the answer key.
        mm1_point : Iteratable[dict]
            The m (m-1)-varieties that define the starting point
            for creating the m-variety of these classifiers.

        Returns
        -------
        Generator of all points on the label m-response simplex.

        """
        label_response_vars = self.r_simplexes.label_response_vars[classifiers]
        mdecisions, vars = zip(*label_response_vars[label]["errors"].items())
        maxs = (
            self.label_mdecision_max(
                classifiers, mdecision, label, ql, mm1_point
            )
            for mdecision in mdecisions
        )
        ranges = (range(0, max + 1) for max in maxs)
        simplex_points = filter(lambda x: sum(x) <= ql, product(*ranges))
        for point in simplex_points:
            yield {var: val for var, val in zip(vars, point)}

    def instantiate_axioms(self, m: int):
        """
        Fill in all response variables in the test axioms.

        Any m-axiom contains three sets of variables:
            1. The answer-key simplex variables. This function
               fills them in with the values given during
               class instantiation.
            2. The observed response counts. This function fills
               them in.
            3. All label m or less response variables. These need
               to be filled in as we move up the ladder of logical
               consistency.

        Parameters
        ----------
        m : int
            DESCRIPTION.

        Returns
        -------
        Mapping[m_subset:instantiated_m_axioms]

        """
        response_eval_dict = self.r_simplexes.response_eval_dict

        axioms = {
            m_subset: {
                label: MAxiomsIdeal(self.labels, m_subset, m_current)
                .m_complex[m_subset]["axioms"][label]
                .subs(response_eval_dict)
                for label in self.labels
            }
            for m_current in range(1, m + 1)
            for m_subset in combinations(self.classifiers, m_current)
        }

        return axioms


class SingleClassifierEvaluations:
    """
    Class for the evaluations of a single classifier given test responses.

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
    The cuboid of correct label response evaluations.

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
        Class initializer.

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

"""Module for super classes of R-axioms based evaluations.

Logically consistent evaluations are defined by the size of an ensemble
and its associated axioms.

"""

from collections import defaultdict
from collections.abc import Sequence
from itertools import combinations, product, tee
from types import MappingProxyType
from typing import Iterable
from typing_extensions import Mapping

import numpy as np
import sympy

from ntqr import Labels
from ntqr.algebra import extract_coefficents, extract_constant
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
        allMVars = MClassifiersVariables(labels, classifiers)
        for m_current in range(1, m + 1):
            m_response_vals = self.m_responses(m_current)
            for m_subset in combinations(classifiers, m_current):
                self.response_vars.update(
                    {m_subset: allMVars.responses[m_subset]}
                )
                self.label_response_vars.update(
                    {m_subset: allMVars.label_responses[m_subset]}
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
        self.m = m

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

        self.relevant_vars = self._relevant_vars()

        self.varieties = {}
        for curr_m in range(1, m + 1):
            pass

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
                label: sympy.simplify(
                    MAxiomsIdeal(self.labels, m_subset, m_current)
                    .m_complex[m_subset]["axioms"][label]
                    .subs(response_eval_dict)
                )
                for label in self.labels
            }
            for m_current in range(1, m + 1)
            for m_subset in combinations(self.classifiers, m_current)
        }

        return axioms

    def _relevant_vars(self):
        label_responses = self.r_simplexes.label_response_vars
        relevant_vars = {}
        for m_curr in range(1, self.m + 1):
            for m_subset in combinations(self.classifiers, m_curr):
                m_vars = relevant_vars.setdefault(m_subset, {})
                for decisions in product(self.labels, repeat=len(m_subset)):
                    decision_vars = m_vars.setdefault(decisions, {})
                    for l_true in self.labels:
                        label_vars = decision_vars.setdefault(l_true, [])
                        label_vars.append(
                            label_responses[m_subset][l_true][decisions]
                        )

        mm1_relevant_vars = {}
        for m_curr in range(2, self.m + 1):
            for m_subset in combinations(self.classifiers, m_curr):
                m_sub_dict = mm1_relevant_vars.setdefault(m_subset, {})
                for decisions in product(self.labels, repeat=m_curr):
                    d_sub_dict = m_sub_dict.setdefault(decisions, {})
                    for l_true in self.labels:
                        curr_list = d_sub_dict.setdefault(l_true, [])
                        for i_subset in combinations(
                            range(m_curr), m_curr - 1
                        ):
                            for sub_clsfs, sub_decisions in zip(
                                tuple(m_subset[i] for i in i_subset),
                                tuple(decisions[i] for i in i_subset),
                            ):

                                curr_list.extend(
                                    relevant_vars[tuple(sub_clsfs)][
                                        tuple(sub_decisions)
                                    ][l_true]
                                )

        return mm1_relevant_vars

    def simplex_points(self, total, maxs, N):
        if N == 1:

            for x in range(0, min(total, maxs[0]) + 1):
                yield [x]
        else:
            for x in range(maxs[0] + 1):
                for part in self.simplex_points(total - x, maxs[1:], N - 1):
                    yield [x] + part

    def simplex_points_equal(self, total, maxs, N):
        if sum(maxs) < total:
            return
        if N == 1:
            if maxs[0] > total:
                yield [total]
            else:
                yield [maxs[0]]
        else:
            for x in range(min(maxs[0], total) + 1):
                for part in self.simplex_points_equal(
                    total - x, maxs[1:], N - 1
                ):
                    yield [x] + part

    def find_combinations(self, target_sum, num_vars):

        if num_vars == 1:
            yield [target_sum]
        else:
            for i in range(0, target_sum + 1):

                for part in self.find_combinations(
                    target_sum - i, num_vars - 1
                ):
                    yield [i] + part

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
            A generator of the points in the variety for these classifiers.

        """
        m = len(classifiers)
        print("Doing m variety: ", m)
        labels_vars = [
            [
                self.r_simplexes.label_response_vars[classifiers][l_true][
                    "all_correct"
                ]
            ]
            + list(
                self.r_simplexes.label_response_vars[classifiers][l_true][
                    "errors"
                ].values()
            )
            for l_true in self.labels
        ]
        labels_decisions_order = [
            [tuple(l_true for classifier in classifiers)]
            + list(
                self.r_simplexes.label_response_vars[classifiers][l_true][
                    "errors"
                ].keys()
            )
            for l_true in self.labels
        ]

        # Root case
        if m > 1:
            mm1_varieties = list(
                [
                    self.variety(mm1_subset)
                    for mm1_subset in combinations(classifiers, m - 1)
                ]
            )
            # We now consider the m-simplex points. This logic is
            # currently not for general m, it assumes m <= 2
            # at m=2 the mm1 varieties are disjoint between the
            # classifiers - sets of distinct size 1 are disjoint.
            # But distinct pairs may have a classifier in common
            # and so on.
            for mm1_point_candidate in product(*mm1_varieties):
                # For m >= 3 there must be logic here to determine
                # if this is a feasible point - not contradictory
                # in assigned values. This can happen for m = 3
                # where points from two different pairs have a
                # classifier in common. That common classifier must
                # have the same values in the mm1_point that the
                # product operation above produced.
                mm1_point = self.make_consistent_if_possible(
                    mm1_point_candidate
                )
                if not mm1_point:
                    continue

                # We now use a valid mm1_point to express the
                # axioms as linear operations.
                axioms_coeffs = self.turn_axiom_exprs_to_linear_components(
                    classifiers, labels_vars, mm1_point
                )

                # We have r of these label simplexes
                labels_m_simplexes = (
                    self.label_msimplex(
                        label,
                        label_decisions_order,
                        classifiers,
                        ql,
                        mm1_point,
                    )
                    for label, label_decisions_order, ql in zip(
                        self.labels, labels_decisions_order, self.qs
                    )
                )
                # A point in this space is a tuple of r label simplexes,
                # one for each true label.
                variety_points = filter(
                    lambda x: self.satisfies_axioms(x, axioms_coeffs),
                    product(*labels_m_simplexes),
                )
                for point in variety_points:
                    yield self.make_variety_point(
                        classifiers, labels_vars, point, mm1_point
                    )
        else:
            # We must be doing one
            mm1_point = {}
            axioms_coeffs = self.turn_axiom_exprs_to_linear_components(
                classifiers, labels_vars, mm1_point
            )
            # We have r of these label simplexes
            labels_m_simplexes = (
                self.label_msimplex(
                    label, label_decisions_order, classifiers, ql, mm1_point
                )
                for label, label_decisions_order, ql in zip(
                    self.labels, labels_decisions_order, self.qs
                )
            )
            # A point in this space is a tuple of r label simplexes,
            # one for each true label.
            variety_points = filter(
                lambda x: self.satisfies_axioms(x, axioms_coeffs),
                product(*labels_m_simplexes),
            )
            for point in variety_points:
                yield self.make_variety_point(
                    classifiers, labels_vars, point, mm1_point
                )

    def turn_axiom_exprs_to_linear_components(
        self,
        classifiers: Sequence[str],
        labels_vars: Sequence[Sequence[sympy.Symbol]],
        mm1_point: Mapping[sympy.Symbol, int],
    ) -> tuple:
        """
        Turn label axioms to its linear vector components.

        Parameters
        ----------
        classifiers : Sequence[str]
            DESCRIPTION.
        mm1_point : Mapping[sympy.Symbol, int]
            DESCRIPTION.

        Returns
        -------
        tuple
            DESCRIPTION.

        """
        mvars_axioms = self.mvars_only_axioms(classifiers, mm1_point)
        label_components = []
        for axiom in mvars_axioms:
            curr_vec = np.insert(
                self.label_coefficients(classifiers, labels_vars, axiom),
                0,
                extract_constant(axiom),
            )
            label_components.append(curr_vec)

        return label_components

    def label_coefficients(
        self,
        classifiers: Sequence[str],
        labels_vars: Sequence[Sequence[sympy.Symbol]],
        axiom: sympy.UnevaluatedExpr,
    ) -> tuple[tuple[int]]:
        """


        Parameters
        ----------
        classifiers : Sequence[str]
            DESCRIPTION.
        axiom : sympy.UnevaluatedExpr
            DESCRIPTION.

        Returns
        -------
        tuple(tuple(int
            DESCRIPTION.

        """
        coeffs = []
        for vars in labels_vars:
            coeffs += extract_coefficents(axiom, vars)

        return np.array(coeffs, dtype=np.int8)

    def mvars_only_axioms(
        self, classifiers: Sequence[str], mm1_point: dict
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
        mm1_point : dict
            DESCRIPTION.

        Returns
        -------
        The test axioms with only unresolved label m-response vars.

        """
        mvars_only_axioms = [
            sympy.simplify(test_axiom.subs(mm1_point))
            for test_axiom in self.test_axioms[classifiers].values()
        ]
        return mvars_only_axioms

    def label_msimplex(
        self,
        label: str,
        label_decisions_order: Sequence[tuple[str]],
        classifiers: Sequence[str],
        ql: int,
        mm1_point: dict,
    ) -> Sequence[tuple[int]]:
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
        maxs = list(
            self.label_mdecision_max(
                classifiers, mdecision, label, ql, mm1_point
            )
            for mdecision in label_decisions_order
        )
        for point in self.simplex_points_equal(ql, maxs, len(maxs)):
            yield point

    def label_mdecision_max(
        self,
        classifiers: Sequence[str],
        decisions: Sequence[str],
        label: str,
        ql: int,
        mm1_point: dict,
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
        mm1_point : dict
            The values of the mm1 variables.

        Returns
        -------
        int
            Max integer value for these classifier decisions given
            true label and the m-1 varieties point.

        """

        # The m-1 vars must be turned into values
        m = len(classifiers)
        response_val = self.r_simplexes.response_vals[classifiers][decisions]
        label_vars = self.r_simplexes.label_response_vars
        # We also cannot exceed the mm1_point values.
        if m > 1:
            mm1_vars = self.relevant_vars[classifiers][decisions][label]
            min_mm1_val = min(var.subs(mm1_point) for var in mm1_vars)

            return min(ql, response_val, min_mm1_val)
        else:
            return min(ql, response_val)

    def make_consistent_if_possible(self, seq_dicts: Sequence[dict]) -> dict:
        """
        Make a consistent m point or return an empty dictionary.

        Parameters
        ----------
        seq_dicts : Sequence[dict]
            DESCRIPTION.

        Returns
        -------
        dict
            DESCRIPTION.

        """
        merged_dicts = self.merge_dicts(seq_dicts)
        # A consistent point has only sets of size 1
        if (
            all(len(vals) == 1 for vals in merged_dicts.values())
            and len(merged_dicts.values()) > 0
        ):
            merged_dict = {
                var: val for var, vals in merged_dicts.items() for val in vals
            }
            return merged_dict
        else:
            return {}

    def merge_dicts(self, seq_dicts):
        """


        Parameters
        ----------
        seq_dicts : Sequence[dict]
            Dictionaries to merge into set values.

        Returns
        -------
        dd : dict
            Dictionary with the union of the keys and set values.

        """
        dd = defaultdict(set)

        for curr_dict in seq_dicts:
            for key, value in curr_dict.items():
                dd[key].add(value)
        return dd

    def make_variety_point(
        self,
        classifiers: Sequence[str],
        label_vars: Sequence[Sequence[sympy.Symbol]],
        mpoint: tuple[tuple[int]],
        mm1_point: dict,
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
        point = {
            var: sympy.sympify(val)
            for vars, vals in zip(label_vars, mpoint)
            for var, val in zip(vars, vals)
        }

        point.update(mm1_point)

        return point

    def satisfies_axioms(
        self,
        mpoint: tuple[tuple[int]],
        maxioms: Mapping[str, tuple],
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
        maxioms :  Mapping[str, tuple]
            The m-axioms reduced the the components for a linear operation.

        Returns
        -------
        bool
            DESCRIPTION.

        """

        satisfies = True
        concatenated_point = [1]
        for label_mvals in mpoint:
            concatenated_point += label_mvals
        c_point = np.array(concatenated_point)

        for coeffs in maxioms:
            res = np.dot(coeffs, c_point)
            if res != 0:
                satisfies = False
                break

        return satisfies


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

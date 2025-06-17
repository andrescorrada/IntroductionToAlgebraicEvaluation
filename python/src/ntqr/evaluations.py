"""Module for super classes of R-axioms based evaluations.

Logically consistent evaluations are defined by the size of an ensemble
and its associated axioms.

"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import chain, combinations, product
from types import MappingProxyType
from typing import Iterable, Mapping, Self, Set

import numpy as np
import numpy.typing as npt
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
        self.m = m
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
        if m > 3:
            raise NotImplementedError(
                "Only M=1,2,3 axioms supported currently."
            )

        self.qs = qs

        # Initialize dicts for vars and vals needed to define
        # the label simplexes for these classifiers
        self._initialize_response_dicts()

        self._initialize_sympy_response_dict()

    def _initialize_response_dicts(self):
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
        allMVars = MClassifiersVariables(self.labels, self.classifiers)
        for m_current in range(1, self.m + 1):
            m_response_vals = self.m_responses(m_current)
            for m_subset in combinations(self.classifiers, m_current):
                self.response_vars.update(
                    {m_subset: allMVars.responses[m_subset]}
                )
                self.label_response_vars.update(
                    {m_subset: allMVars.label_responses[m_subset]}
                )
                self.response_vals[m_subset] = m_response_vals[m_subset]

    def _initialize_sympy_response_dict(self):
        # Now we construct the evaluation dict needed by SymPy to evaluate
        # the axioms
        var_val_pairs = zip(
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
        response_eval_dict = {var: val for var, val in var_val_pairs}
        # And add the q_s values
        self.q_vars = MappingProxyType(
            {label: sympy.Symbol(r"Q_" + label) for label in self.labels}
        )
        response_eval_dict.update(
            {
                self.q_vars[label]: val
                for label, val in zip(self.labels, self.qs)
            }
        )

        self.response_eval_dict = MappingProxyType(response_eval_dict)

        return

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


@dataclass(frozen=True, kw_only=True)
class MVariety:
    """
    Class for the points obeying all axiom orders up to M.

    The M=1 variety contains points that obey the M=1 axioms
    for a single classifier. The M=2 variety contains points
    that obey the M=2 axioms for a pair of classifiers, but
    also the M=1 axioms for each of them.
    """

    labels: tuple[str]
    classifiers: tuple[str]
    qs: Mapping[str, sympy.Symbol]
    m: int
    label_vars: tuple[sympy.Symbol]
    points: npt.NDArray[np.uint16] = field(repr=False)

    def __and__(self, other_variety: Self) -> Self:
        """
        Create logical intersection of self and other_variety.

        Constructing varieties of order M=m requires that we
        find the logically consistent intersection of m-1 varieties.
        Starting at M=3, two varieties of order m-1 share some of their
        variables. The 'and' operation returns the joined points that
        are logically consistent with each other: have the same value
        for the shared variables.

        Parameters
        ----------
        other_variety : MVariety
            Variety to do logical intersection with. It must be
            of the same M=m order as this one.

        Returns
        -------
        MVariety
            The points whose intersection is logically consistent.
        """
        # We first must make sure that the varieties are of the
        # same order: m_self = m_other_variety
        if not len(other_variety.classifiers) == len(self.classifiers):
            raise ValueError(
                f"""Cannot perform logical intersection on varieties
                of different order. This variety is of order
                m={len(self.classifiers)}. 'other_variety' is of order
                m={len(other_variety.classifiers)}."""
            )

    def intersection_label_vars(
        self, other_variety: Self
    ) -> Set[sympy.Symbol]:
        """
        Find variables shared by both varieties.

        Parameters
        ----------
        other_variety : Self
            DESCRIPTION.

        Returns
        -------
        Set of shared variables between the two varieties.

        """
        return frozenset(set(self.label_vars) & set(other_variety.label_vars))


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

        self.mm1_relevant_vars = self._relevant_vars()
        self.var_max_responses = self._var_max_responses()

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

    def _var_max_responses(self) -> Mapping[sympy.Symbol, int]:
        """
        Calculate the maximum possible value of all label response vars.

        Returns
        -------
        Mapping[sympy.Symbol, int]
        The observable value for the label response variable.

        """
        label_responses = self.r_simplexes.label_response_vars
        max_response_possible = {}
        for m_curr in range(1, self.m + 1):
            for m_subset in combinations(self.classifiers, m_curr):
                for decisions in product(self.labels, repeat=len(m_subset)):
                    for l_true in self.labels:
                        var = label_responses[m_subset][l_true][decisions]
                        max_response_possible[var] = (
                            self.r_simplexes.response_vals[m_subset][decisions]
                        )

        return max_response_possible

    def _relevant_vars(self):
        # Data structure needed to connect m response variables to
        # their relevant m-1 counterparts.
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

        # A mapping between an m variable and m-1 variables that
        # constrain its max
        for m_curr in range(2, self.m + 1):
            for m_subset in combinations(self.classifiers, m_curr):
                for decisions in product(self.labels, repeat=m_curr):
                    for l_true in self.labels:
                        # What label response variable are we using?
                        var = label_responses[m_subset][l_true][decisions]
                        curr_list = mm1_relevant_vars.setdefault(var, [])
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

    def mvariety(self, classifiers: Sequence[str]) -> MVariety:
        """
        Construct the MVariety dataclass for these classifiers.

        Parameters
        ----------
        classifiers : Sequence[str]
            The classifiers.

        Returns
        -------
        MVariety
            Dataclass containing the points in the variety.

        """
        m = len(classifiers)
        print("Doing m variety: ", m)

        # We concatenate all the label response vars sorted by
        # self.labels
        labels_mvars = [
            [
                self.r_simplexes.label_response_vars[classifiers][l_true][
                    decisions
                ]
                for decisions in product(self.labels, repeat=len(classifiers))
            ]
            for l_true in self.labels
        ]
        print("label_mvars: ", labels_mvars)

        labels_vars = list(chain(*labels_mvars))
        print("labels_vars: ", labels_vars)

        mvars_axioms_coeffs = self.turn_axiom_exprs_to_vectors(
            classifiers, labels_vars
        )

        # Root case
        if m == 1:
            # We are faking the empty set to keep the function call
            # the same for turn_axioms_exprs_to_linear_vectors
            mm1_point = np.empty(0, dtype=np.uint16)
            # We have r of these label simplexes
            labels_m_simplexes = (
                self.label_msimplex(label_mvars, {}, ql, mm1_point)
                for label_mvars, ql in zip(labels_mvars, self.qs)
            )
            # A point in this space is a tuple of r label simplexes,
            # one for each true label.
            variety_points = [
                mpoint
                for mpoint in product(*labels_m_simplexes)
                if self.satisfies_axioms(mpoint, mvars_axioms_coeffs)
            ]

            return MVariety(
                labels=self.labels,
                classifiers=classifiers,
                qs=self.qs,
                m=m,
                label_vars=labels_vars,
                points=np.array(
                    [
                        np.array(sum(point, []), dtype=np.uint16)
                        for point in variety_points
                    ]
                ),
            )

        elif m > 1:
            mm1_varieties = [
                self.mvariety(mm1_subset)
                for mm1_subset in combinations(classifiers, m - 1)
            ]
            # We now consider the m-simplex points. This logic is
            # currently not for general m, it assumes m <= 2
            # at m=2 the mm1 varieties are disjoint between the
            # classifiers - sets of distinct size 1 are disjoint.
            # But distinct pairs may have a classifier in common
            # and so on.

            # We need the order of the vars in the points of each variety
            # And we need to sort them for reconciliation
            mm1_label_vars = [variety.label_vars for variety in mm1_varieties]
            reconciled_mm1_vars = self.reconcile_variety_vars(mm1_label_vars)
            mm1_indices = {
                reconciled_mm1_vars[i]: i
                for i in range(len(reconciled_mm1_vars))
            }
            mm1_variety_points = [variety.points for variety in mm1_varieties]

            # We drop the constant term for the mm1 coeffs vector
            mm1_coeffs = [
                coeffs[1]
                for coeffs in self.turn_axiom_exprs_to_vectors(
                    classifiers, reconciled_mm1_vars
                )
            ]

            variety = []

            for mm1_point_candidate in product(*mm1_variety_points):
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
                if len(mm1_point) == 0:
                    continue

                # Calculate the contribution of the mm1 vars to the axioms
                new_coeffs = []
                for mcoeffs, mm1coeffs in zip(mvars_axioms_coeffs, mm1_coeffs):
                    curr_coeff = (
                        mcoeffs[0] + np.dot(mm1_point, mm1coeffs),
                        mcoeffs[1],
                    )
                    new_coeffs.append(curr_coeff)

                # We have r of these label simplexes
                labels_m_simplexes = (
                    self.label_msimplex(
                        label_mvars, mm1_indices, ql, mm1_point
                    )
                    for label_mvars, ql in zip(labels_mvars, self.qs)
                )
                # A point in this space is a tuple of r label simplexes,
                # one for each true label.
                variety_points = [
                    mpoint
                    for mpoint in product(*labels_m_simplexes)
                    if all(
                        (
                            (
                                coeffs[0]
                                + np.dot(
                                    coeffs[1], np.array(list(chain(*mpoint)))
                                )
                            )
                            == 0
                        )
                        for coeffs in new_coeffs
                    )
                ]

                for point in variety_points:
                    new_point = sum(point, [])
                    new_point += mm1_point
                    variety.append(np.array(new_point, dtype=np.uint16))

                header_vars = labels_vars + reconciled_mm1_vars

            return MVariety(
                labels=self.labels,
                classifiers=classifiers,
                qs=self.qs,
                m=m,
                label_vars=header_vars,
                points=np.stack(variety),
            )

    def reconcile_variety_vars(
        self, label_vars: Sequence[Sequence[sympy.Symbol]]
    ) -> Sequence[sympy.Symbol]:
        """
        Reconcile label response variables from the varieties.

        Parameters
        ----------
        label_vars : Sequence[Sequence[sympy.Symbol]]
            DESCRIPTION.

        Returns
        -------
        Sequence[sympy.Symbol].
            The ordering with removed common duplicates.

        """
        # This function is currently not functional for m = 2 varieties or
        # higher. We are getting away that for m = 1 there are no common
        # variables between the variety points.
        reconciled_vars = sum(label_vars, [])
        return reconciled_vars

    def turn_axiom_exprs_to_vectors(
        self,
        classifiers: Sequence[str],
        labels_vars: Sequence[sympy.Symbol],
    ) -> tuple[npt.NDArray[np.int16]]:
        """
        Turn label axioms a vector of coefficients

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
        test_axioms = self.test_axioms[classifiers]
        var_coefficients = []

        for axiom in test_axioms.values():
            curr_vec = (
                extract_constant(axiom),
                self.label_coefficients(classifiers, labels_vars, axiom),
            )
            var_coefficients.append(curr_vec)

        return var_coefficients

    def label_coefficients(
        self,
        classifiers: Sequence[str],
        labels_vars: Sequence[sympy.Symbol],
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
        coeffs = extract_coefficents(axiom, labels_vars)

        return np.array(coeffs, dtype=np.int16)

    def label_msimplex(
        self,
        label_mvars: Sequence[sympy.Symbol],
        var_indices: Mapping[sympy.Symbol, int],
        ql: int,
        mm1_point: npt.NDArray[np.uint16],
    ) -> Sequence[npt.NDArray[np.uint16]]:
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
        maxs = self.vars_max_values(label_mvars, var_indices, ql, mm1_point)

        for point in self.simplex_points_equal(ql, maxs, len(maxs)):
            yield point

    def vars_max_values(
        self,
        label_mvars: Sequence[sympy.Symbol],
        var_indices: Mapping[sympy.Symbol, int],
        ql: int,
        mm1_point: npt.NDArray[np.uint16],
    ) -> Sequence[int]:
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
        max_responses = [self.var_max_responses[var] for var in label_mvars]
        # We also cannot exceed the mm1_point values.
        if len(mm1_point) > 0:
            relevant_vars = [
                self.mm1_relevant_vars[var] for var in label_mvars
            ]
            mm1_point_indices = [
                [var_indices[var] for var in rel_vars]
                for rel_vars in relevant_vars
            ]
            mm1_mins = [
                min(mm1_point[i] for i in indices)
                for indices in mm1_point_indices
            ]

            return [
                min(ql, res_val, mm1_val)
                for res_val, mm1_val in zip(max_responses, mm1_mins)
            ]
        else:
            return [min(ql, max_val) for max_val in max_responses]

    def make_consistent_if_possible(
        self, mm1_points: Sequence[npt.NDArray[np.uint16]]
    ) -> npt.NDArray[np.uint16]:
        """
        Make a consistent m point or return an empty numpy array.

        Parameters
        ----------
        mm1_points : Sequence[npt.NDArray[np.uint16]]
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        # Again, this function will not work correctly for m = 2 varieties
        # or higher.
        # We concatenate the point arrays for now
        consistent_point = np.concatenate(mm1_points).tolist()
        return consistent_point

    def satisfies_axioms(
        self,
        mpoint: tuple[tuple[int]],
        maxioms: Sequence[tuple[int, npt.NDArray[np.uint16]]],
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
        # satisfies = True
        # c_point = np.array(np.concatenate(mpoint))

        # for coeffs in maxioms:
        #     res = coeffs[0] + np.dot(coeffs[1], c_point)
        #     if res != 0:
        #         satisfies = False
        #         break

        return all(
            (
                (
                    coeffs[0]
                    + np.dot(coeffs[1], np.array(np.concatenate(mpoint)))
                )
                == 0
            )
            for coeffs in maxioms
        )


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

"""Module for super classes of R-axioms based evaluations.

Logically consistent evaluations are defined by the size of an ensemble
and its associated axioms.

"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import chain, combinations, product
from types import MappingProxyType
from typing import Iterable, Mapping, Optional, Self, Set, Tuple

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

        Observed responses set the ceilings for any possible value for
        the label responses - responses given true label. For example,
        if we observe that two classifiers agreed on the same label some
        number of times, no possible evaluation of that agreement given
        true label can exceed this number.

        Observed responses create the 'ratchet' of evaluation. No label
        response variable can have a value larger than that of any subset
        of the classifiers responding similarly.

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

    This class follows the 'code smell' test, it appears because
    we need operations that can create logically consistent
    intersections of varieties of fixed order.

    These varieties, by construction, do not use any information
    about responses at higher order. So the union of m=1 varieties
    never uses, or can contain, information about pair responses
    or higher.

    Thus the intersection of m-varities is a containing variety
    for the variety that corresponds to all axioms up to m=N
    being obeyed.
    """

    labels: tuple[str]
    classifiers: tuple[str]
    qs: Mapping[str, sympy.Symbol]
    m: int
    label_vars: tuple[tuple[sympy.Symbol]]
    points: Mapping[tuple, Mapping[tuple, {}]] = field(repr=False)

    def __eq__(self, other_variety: Self) -> bool:
        """
        Test equality.

        The current implementation is a weaker check on equality.
        It just verifies that self.m and self.label_vars are equal.
        If so, it returns true.

        A strict check on equality would verify that all points are
        also equal.

        Parameters
        ----------
        other_variety : Self
            Other variety.

        Returns
        -------
        bool
            Whether the varities are equal.

        """

        if self.m != other_variety.m:
            return False

        union_classifiers = self.union_classifiers(other_variety)
        other_only_vars = self.only_other_vars(other_variety)
        if (
            set(other_variety.classifiers).issubset(union_classifiers)
            and other_only_vars == tuple()
        ):
            return (True,)
        else:
            return False

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
        if not self.m == other_variety.m:
            raise ValueError(
                f"""Cannot perform logical intersection on varieties
                of different order. This variety is of order
                m={self.m}. 'other_variety' is of order m={other_variety.m}."""
            )

        if self == other_variety:
            return self

        return self.construct_intersection_variety(other_variety)

    def compute_from_indices(
        self, other_variety: Self, var_order: Sequence[sympy.Symbol]
    ) -> Sequence[int]:
        """
        Compute the indices for finding the values of the final variety.

        Whenever we do logical AND of varities, the final variety will contain
        values from both varieties. This function returns the indices in
        their concatenated variety point where we can find the variables
        specified in 'var_order.'

        Parameters
        ----------
        var_order : Sequence[sympy.Symbol]
            Sequence that defines the order of the variables.

        Returns
        -------
        Sequence[int]
            Source index in a concatenated variety point.

        """
        from_indices = {var_order[i]: 0 for i in range(len(var_order))}

        self_only_vars = self.only_self_vars(other_variety)
        all_self_vars = list(chain(*self.label_vars))
        in_self_index = {
            var: i for var, i in zip(all_self_vars, range(len(all_self_vars)))
        }

        other_only_vars = self.only_other_vars(other_variety)
        all_other_vars = list(chain(*other_variety.label_vars))
        in_other_index = {
            var: i
            for var, i in zip(all_other_vars, range(len(all_other_vars)))
        }

        common_vars = self.common_vars(other_variety)

        # Update where to find the data we need for each new point
        for var in self_only_vars:
            from_indices[var] = in_self_index[var]
        for var in common_vars:
            from_indices[var] = in_self_index[var]
        for var in other_only_vars:
            from_indices[var] = in_other_index[var] + len(all_self_vars)

        return from_indices

    def intersection_label_vars(
        self, other_variety: Self
    ) -> Set[sympy.Symbol]:
        """
        Find variables shared by both varieties.

        Beginning at m=2, varieties may share lower m variables. In addition,
        we want the __and__ operation to be idempotent when a previously
        joined variety is joined again - joining self with another variety
        gives the same result if we join it again.

        Parameters
        ----------
        other_variety : Self
            The other variety.

        Returns
        -------
        Set of shared variables between the two varieties.

        """
        return frozenset(set(self.label_vars) & set(other_variety.label_vars))

    def union_classifiers(self, other_variety: Self) -> Set[str]:
        """
        Find union of the classifiers in self and 'other_variety'.

        Parameters
        ----------
        other_variety : Self
            Variety to compare with.

        Returns
        -------
        Set[str]
            The union of the classifiers in self and 'other_variety'.

        """
        return tuple(
            sorted(set(self.classifiers) | set(other_variety.classifiers))
        )

    def var_order(self, other_variety: Self) -> tuple[sympy.Symbol]:
        """
        Construct var order of the intersection of self with 'other_variety'.

        Parameters
        ----------
        other_variety : Self
            A variety of the same order as self.

        Returns
        -------
        The var order for the points in the intersection of the varieties.

        """
        # We have to be careful, we will be missing variables
        vars_present = set(chain(*self.label_vars)).union(
            set(chain(*other_variety.label_vars))
        )
        union_classifiers = tuple(
            sorted(self.union_classifiers(other_variety))
        )
        mvars = MClassifiersVariables(self.labels, union_classifiers)
        vars = []
        for m_curr in range(1, self.m + 1):
            for m_subset in combinations(
                self.union_classifiers(other_variety), m_curr
            ):
                for label in self.labels:
                    for decisions in product(self.labels, repeat=m_curr):
                        var = mvars.label_responses[m_subset][label][decisions]
                        if var in vars_present:
                            vars.append(var)

        return tuple(vars)

    def common_m1_vars(
        self, other_variety: Self, var_order: tuple[sympy.Symbol]
    ) -> tuple[str]:
        """
        Find common m=1 vars following var order.

        Parameters
        ----------
        other_variety : Self
            Variety to check.
        var_order : tuple[sympy.Symbol]
            Canonical var order.

        Returns
        -------
        tuple[str]
            m=1 label responses variables shared by self with other_variety.

        """
        other_m1_vars = set(other_variety.label_vars[0])
        self_m1_vars = set(self.label_vars[0])
        intersection_set = self_m1_vars.intersection(other_m1_vars)
        common_m1_vars = [var for var in var_order if var in intersection_set]
        return common_m1_vars

    def m1_var_indices(
        self, m1_vars: Sequence[sympy.Symbol], variety: Self
    ) -> tuple[int]:
        """
        Find the position of m=1 vars in variety.label_vars.

        Parameters
        ----------
        m1_vars : Sequence[sympy.Symbol]
            m=1 label response variables for which we want the index.
        variety : Self
            The variety whose m1 variables will be indexed.

        Returns
        -------
        tuple[int]
            Indices of the m=1 label response variables in a variety point.

        """
        variety_m1_vars = variety.label_vars[0]
        indices = [
            i
            for i in range(len(variety_m1_vars))
            if variety_m1_vars[i] in m1_vars
        ]
        return indices

    def common_m2p_vars(
        self, other_variety: Self, var_order: tuple[sympy.Symbol]
    ) -> tuple[str]:
        """
        Find common m=2 or higher vars following var order.

        Parameters
        ----------
        other_variety : Self
            Other variety.
        var_order : tuple[sympy.Symbol]
            Canonical var order.

        Returns
        -------
        tuple[str]
            m>=2 label response variables shared by self and 'other_variety'.

        """
        other_m2p_vars = set(chain(*other_variety.label_vars[1:]))
        self_m2p_vars = set(chain(*self.label_vars[1:]))
        intersection_set = self_m2p_vars.intersection(other_m2p_vars)
        common_m2p_vars = [var for var in var_order if var in intersection_set]
        return common_m2p_vars

    def m2p_var_indices(
        self, m2p_vars: Sequence[sympy.Symbol], variety: Self
    ) -> tuple[int]:
        """
        Find the indices for the m>=2 label responses variables.

        Parameters
        ----------
        m2p_vars : Sequence[sympy.Symbol]
            m>=2 label responses variables.
        other_variety : Self
            The other variety.

        Returns
        -------
        tuple[int]
            Indices of the m>=2 label response variables in a point from
            'variety'.

        """
        these_m2p_vars = list(chain(*variety.label_vars[1:]))
        indices = [
            i
            for i in range(len(these_m2p_vars))
            if these_m2p_vars[i] in m2p_vars
        ]
        return indices

    def only_self_vars(self, other_variety: Self) -> tuple[sympy.Symbol]:
        """
        Find label response variables only self has.

        Parameters
        ----------
        other_variety : Self
            The other variety.

        Returns
        -------
        tuple[sympy.Symbol]
        Label response variables only found in self and not in 'other_variety'.

        """
        only_in_self = tuple(
            var
            for var in list(chain(*self.label_vars))
            if var
            in set(chain(*self.label_vars)).difference(
                set(chain(*other_variety.label_vars))
            )
        )
        return only_in_self

    def only_other_vars(self, other_variety: Self) -> tuple[sympy.Symbol]:
        """
        Find label response variables only 'other_variety' has.

        Parameters
        ----------
        other_variety : Self
            The variety we are comparing with.

        Returns
        -------
        tuple[sympy.Symbol]
        Label response variables only found in 'other_variety'.

        """
        only_in_other = tuple(
            var
            for var in list(chain(*other_variety.label_vars))
            if var
            in set(chain(*other_variety.label_vars)).difference(
                set(chain(*self.label_vars))
            )
        )
        return only_in_other

    def common_vars(self, other_variety: Self) -> tuple[sympy.Symbol]:
        """
        Find label response variables in self and other_variety.

        Parameters
        ----------
        other_variety : Self
            The other variety.

        Returns
        -------
        tuple[sympy.Symbol]
        Label response variables found in self and other_variety.

        """
        return tuple(
            var
            for var in list(chain(*self.label_vars))
            if var
            in set(chain(*self.label_vars)).intersection(
                set(chain(*other_variety.label_vars))
            )
        )

    def var_indices(
        self, vars: Sequence[sympy.Symbol], variety: Self
    ) -> Iterable:
        """
        Get indices for vars in self.label_vars.

        Parameters
        ----------
        vars : Sequence[sympy.Symbol]
            Variables for which we want indices in points from 'variety'.
        variety : Self
            The variety where the indices are to be found.

        Returns
        -------
        Iterable
            The indices for 'vars' in a point from 'variety'.

        """
        var_indices = tuple(
            i
            for i in range(len(variety.label_vars))
            if variety.label_vars[i] in vars
        )

        return var_indices

    def join_label_vars(
        self, var_order: Sequence[sympy.Symbol], other_variety: Self
    ) -> tuple[tuple[sympy.Symbol]]:
        """
        Join label_vars by m-order.

        Parameters
        ----------
        other_variety : Self
            The variety whose variables will be joined with self.

        Returns
        -------
        The joined vars by m-order.

        """
        joined_vars = tuple(
            tuple(
                var
                for var in var_order
                if var in set(self_vars).union(set(other_vars))
            )
            for self_vars, other_vars in zip(
                self.label_vars, other_variety.label_vars
            )
        )

        return joined_vars


class MVarietyTupleDict(MVariety):
    """Concrete class for MVariety storing points as dict of tuples.

    Warning: this class is memory intensive.
    """

    def generate_points(self) -> Iterable:
        """
        Generate the points in this variety.

        Yields
        ------
        Iterable
            Points in the variety.

        """

        for key in self.points.keys():
            list_val = self.points[key]
            for val in list_val:
                yield key + val

    def generate_consistent_point_pairs(
        self, var_order: Sequence[sympy.Symbol], other_variety: Self
    ) -> Iterable[tuple[tuple[int], tuple[int]]]:
        """
        Generate pairs of points from each variety consisten with each other.

        Whenever we are joining varieties of order $m \geq 2$, care must
        be taken to only combine points that agree on their common variables.
        For example, if we have the variety for classifiers 'i' and 'j' and
        want to join it with the variety for classifiers 'j' and 'k', we
        must make sure that we return point pairs, one from each variety,
        that agree in their values of 'j' responses.

        Parameters
        ----------
        var_order : Sequence[sympy.Symbol]
            Sequence of vars that defines order of variables in the
            joined variety.
        other_variety : Self
            The variety to check consistency on common variables.

        Yields
        ------
        (Iterable[tuple[tuple[int], tuple[int]]])
            Pairs of points, one from each variety, that are logically
            consistent with each other (agree on common variables).

        """
        common_m2p_vars = self.common_m2p_vars(other_variety, var_order)
        scm2pi = self.m2p_var_indices(common_m2p_vars, self)
        ocm2pi = self.m2p_var_indices(common_m2p_vars, other_variety)

        common_m1_vars = self.common_m1_vars(other_variety, var_order)

        for self_key, other_key in self.generate_consistent_key_pairs(
            common_m1_vars, other_variety
        ):

            for self_tail, other_tail in self.generate_consistent_tail_pairs(
                other_variety, self_key, other_key, scm2pi, ocm2pi
            ):

                yield (self_key + self_tail, other_key + other_tail)

    def generate_consistent_key_pairs(
        self,
        common_m1_vars: Sequence[sympy.Symbol],
        other_variety: Self,
    ) -> Iterable[Tuple[Tuple[int], Tuple[int]]]:
        """
        Generate key pairs for self and other_variety that are consistent.

        Parameters
        ----------
        other_variety : Self
            The variety we are joining self with.

        Returns
        -------
        Tuples of m1 keys for the points dict of each variety.

        """
        # Self common m1 var indices
        scm1i = self.m1_var_indices(common_m1_vars, self)
        ocm1i = self.m1_var_indices(common_m1_vars, other_variety)
        for self_key, other_key in product(
            self.points.keys(), other_variety.points.keys()
        ):
            test_value = [self_key[i] for i in scm1i] == [
                other_key[i] for i in ocm1i
            ]

            if test_value:
                yield (self_key, other_key)

    def generate_consistent_tail_pairs(
        self,
        other_variety: Self,
        self_key: Tuple[int],
        other_key: Tuple[int],
        scm2pi: Tuple[int],
        ocm2pi: Tuple[int],
    ) -> Iterable[Tuple[Tuple[int], Tuple[int]]]:
        """
        Generate key pairs for self and other_variety that are consistent.

        Parameters
        ----------
        other_variety : Self
            The variety we are joining self with.

        Returns
        -------
        Tuples of m1 keys for the points dict of each variety.

        """
        if self.m == 1:
            yield (tuple(), tuple())
        for self_tail, other_tail in product(
            self.points[self_key],
            other_variety.points[other_key],
        ):
            test_value = [self_tail[i] for i in scm2pi] == [
                other_tail[i] for i in ocm2pi
            ]
            if test_value:
                yield (self_tail, other_tail)

    def construct_intersection_variety(self, other_variety: MVariety) -> Self:
        """
        Construct the intersection.

        Parameters
        ----------
        other_variety : MVariety
            The variety we are intersecting self with.

        Returns
        -------
        Self
            The intersection of the two varieties..

        """
        # We first construct the var order in the final variety
        var_order = self.var_order(other_variety)

        # Figure out where to find the data needed for the final variety
        from_indices = self.compute_from_indices(other_variety, var_order)

        # We want to create the key so we need the length of the m=1
        # vars
        joined_label_vars = self.join_label_vars(var_order, other_variety)
        len_key = len(joined_label_vars[0])
        points_dict = {}

        var_indices = [from_indices[var] for var in var_order]

        for self_point, other_point in self.generate_consistent_point_pairs(
            var_order, other_variety
        ):
            cat_point = self_point + other_point
            new_key = tuple(cat_point[i] for i in var_indices[:len_key])
            new_tail = tuple(cat_point[i] for i in var_indices[len_key:])
            key_list = points_dict.setdefault(new_key, [])
            key_list.append(new_tail)

        return MVarietyTupleDict(
            labels=self.labels,
            classifiers=self.union_classifiers(other_variety),
            qs=self.qs,
            m=self.m,
            label_vars=joined_label_vars,
            points=points_dict,
        )


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
            The order of the axioms, an integer of value 1 or greather.

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

    def simplex_points_equal(self, total: int, maxs: Sequence[int], N: int):
        """
        Generate all simplex points with values less than or equal to maxs.

        This is a recursive generator to handle arbitrary number of variables
        in a simplex.

        Parameters
        ----------
        total : int
            Total value required for the sum of the vars on the simplex.
        maxs : Sequence[int]
            The max integer value that a var can have on that simplex.
        N : int
            The number of vars defining the simplex.

        Yields
        ------
        tuple[int]
            Simplex points, of dimension 'N', that sum to 'total' and whose
            var values do not exceed maxs values.

        """
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
        # print("Doing m variety: ", m)

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

        labels_vars = list(chain(*labels_mvars))

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

            return MVarietyTupleDict(
                labels=self.labels,
                classifiers=classifiers,
                qs=self.qs,
                m=m,
                label_vars=(tuple(labels_vars),),
                points={tuple(sum(point, [])): [] for point in variety_points},
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
            mm1_joined_variety = mm1_varieties[0]
            for mm1_variety in mm1_varieties:
                mm1_joined_variety &= mm1_variety
            mm1_vars = tuple(chain(*mm1_joined_variety.label_vars))
            mm1_indices = {
                var: i for var, i in zip(mm1_vars, range(len(mm1_vars)))
            }

            # We drop the constant term for the mm1 coeffs vector
            mm1_coeffs = [
                coeffs[1]
                for coeffs in self.turn_axiom_exprs_to_vectors(
                    classifiers, mm1_vars
                )
            ]

            points_dict = {}
            for mm1_point in mm1_joined_variety.generate_points():

                # Calculate the contribution of the mm1 vars to the axioms
                new_coeffs = []
                for label_mcoeffs, label_mm1coeffs in zip(
                    mvars_axioms_coeffs, mm1_coeffs
                ):
                    curr_coeff = (
                        label_mcoeffs[0]
                        + np.dot(np.array(mm1_point), label_mm1coeffs),
                        label_mcoeffs[1],
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

                # The indices of the key are all the values
                # at the front that point to a point in single
                # classifier responses
                len_key = len(classifiers) * (len(self.labels)) ** 2

                for point in variety_points:
                    new_point = mm1_point + tuple(chain(*point))
                    point_list = points_dict.setdefault(
                        new_point[:len_key], []
                    )
                    point_list.append(new_point[len_key:])

            header_vars = list(mm1_joined_variety.label_vars)
            header_vars.append(tuple(labels_vars))

            return MVarietyTupleDict(
                labels=self.labels,
                classifiers=classifiers,
                qs=self.qs,
                m=m,
                label_vars=header_vars,
                points=points_dict,
            )

    def turn_axiom_exprs_to_vectors(
        self, classifiers: Sequence[str], labels_vars: Sequence[sympy.Symbol]
    ) -> tuple[npt.NDArray[np.int16]]:
        """
        Turn label axioms into an array of coefficient vectors.

        Parameters
        ----------
        classifiers : Sequence[str]
            Sequence of the classifiers to consider.
        labels_vars : Sequence[sympy.Symbol]
            The order of the label vars in the returned tuple of ints.


        Returns
        -------
        var_coefficients : tuple[npt.NDArray[np.int16]]
            Sequence of integer coefficients for the axioms, one for each
            label.
        """
        test_axioms = self.test_axioms[classifiers]
        var_coefficients = []

        for axiom in test_axioms.values():
            curr_vec = (
                extract_constant(axiom),
                self.label_coefficients(labels_vars, axiom),
            )
            var_coefficients.append(curr_vec)

        return var_coefficients

    def label_coefficients(
        self, labels_vars: Sequence[sympy.Symbol], axiom: sympy.UnevaluatedExpr
    ) -> tuple[int]:
        """
        Compute the coefficients for labels_vars than appear in axiom.

        Parameters
        ----------
        labels_vars : Sequence[sympy.Symbol]
            Variables for which we want coefficients in axiom.
        axiom : sympy.UnevaluatedExpr
            The algebraic expression of the axiom.

        Returns
        -------
        tuple[int]
            The integer coefficients for label_vars in axiom.

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
            Order m minus 1 points to be joined logically.

        Returns
        -------
        npt.NDArray[np.uint16]
            An array of the joined points if possible, empty otherwise.

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
            Does this order m point obey the m-order axioms?

        """
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

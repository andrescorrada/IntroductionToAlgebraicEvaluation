"""
Evaluations for binary tests (R=2).

For any finite test there is a finite set of evaluations possible. The
classes in this module compute them.

Classes:
    SingleClassifierEvaluations: Class related to the evaluations for a
    single classifier consistent with its observed test responses.

Functions:

Misc variables:

"""

import itertools

import sympy

import ntqr.evaluations
from ntqr.r2.datasketches import VoteCounts


class SingleClassifierEvaluations(
    ntqr.evaluations.SingleClassifierEvaluations
):
    """
    Single classifier evaluations in (Q_a, Q_b, R_{b_i, a}, R_{a_i,b}) space.

    Deprecated class.
    """

    def __init__(self, Q, single_axioms):
        super().__init__(Q, single_axioms)

    def number_apriori_evaluations(self):
        """
        Calculate all the possible evaluations for a binary response test.

        Returns
        -------
        int

        """
        q = self.Q
        return 1 / 6 * (q + 1) * (q + 2) * (q + 3)

    def errors_at_qs(self, qs, responses):
        """
        Return all evaluations logically consistent with responses.

        In binary classification we have Q_a + Q_b = Q. Thus, we
        really need to specify only two of the three (Q, Q_a, Q_b).
        Making a choice is arbitrary and breaks the symmetry in the
        algebra between the two labels. Instead, we specify Q_a, and
        Q_b and since we have Q from the instance initialization, we
        do a quality check (logic check) of the equality between the
        three quantities.
        """
        eval_dict = {
            var: val
            for var, val in zip(self.axioms.questions_number.values(), qs)
        }
        eval_dict.update(
            {
                var: val
                for var, val in zip(self.axioms.responses.values(), responses)
            }
        )

        wrong_vars = [
            [
                wrong_var
                for wrong_var in self.axioms.responses_by_label[true_label][
                    "errors"
                ].values()
            ]
            for true_label in self.axioms.labels
        ]

        evals = set(
            (first_lbl_wrong, second_lbl_wrong)
            for first_lbl_wrong in self._label_wrongs_(qs[0])
            for second_lbl_wrong in self._label_wrongs_(qs[1])
            if self._check_axiom_consistency_(
                eval_dict,
                itertools.chain(*wrong_vars),
                itertools.chain(first_lbl_wrong, second_lbl_wrong),
            )
        )

        return evals

    def max_correct_at_qs(self, qs, responses):
        """Gives highest performing correct for each label.

        Meant to save memory for alarm applications.
        """

        eval_dict = {
            var: val
            for var, val in zip(self.axioms.questions_number.values(), qs)
        }
        eval_dict.update(
            {
                var: val
                for var, val in zip(self.axioms.responses.values(), responses)
            }
        )

        wrong_vars = [
            [
                wrong_var
                for wrong_var in self.axioms.responses_by_label[true_label][
                    "errors"
                ].values()
            ]
            for true_label in self.axioms.labels
        ]

        max_correct = (0, (0, 0))

        for first_label_wrongs in self._label_wrongs_(qs[0]):
            vars = wrong_vars[0]
            vals = first_label_wrongs
            for second_label_wrongs in self._label_wrongs_(qs[1]):
                vars += wrong_vars[1]
                vals += second_label_wrongs

                if self._check_axiom_consistency_(eval_dict, vars, vals):
                    corrects = [
                        ql - sum(wrongs)
                        for ql, wrongs in zip(
                            qs,
                            [first_label_wrongs, second_label_wrongs],
                        )
                    ]
                    corrects_sum = sum(corrects)
                    if corrects_sum > max_correct[0]:
                        max_correct = (corrects_sum, corrects)

        return max_correct[1]

    def all_qs(self):
        """Return all possible question numbers."""
        Q = self.Q
        return [(qa, Q - qa) for qa in range(0, self.Q + 1)]

    def _label_wrongs_(self, ql):
        """Return all possible incorrect given Q_l."""
        return [(num_wrong,) for num_wrong in range(0, ql + 1)]


# This class needs to be deleted and code that uses refactored
class PosteriorSingleEvaluations:
    """Evaluations logically consistent with observed responses."""

    def __init__(self, tvc: VoteCounts):
        self.Q = tvc.test_size
        self.tvc = tvc

    def all_possible_evaluations(self, classifier: int):
        # The b label frequency for the classifier
        Rb = int(self.tvc.classifier_label_frequency(classifier, "b") * self.Q)
        evaluations = {}
        for Qa in range(0, self.Q + 1):
            evaluations[Qa] = []
            for Raa in range(0, Qa + 1):
                for Rbb in range(0, self.Q - Qa + 1):
                    if Raa - Rbb - Qa + Rb == 0:
                        evaluations[Qa].append((Raa, Rbb))

        return evaluations

    def find_k_nearest_at_prevalence(
        self, classifier: int, qa: int, point, k: int
    ):
        # The b label frequency for the classifier
        Rb = int(self.tvc.classifier_label_frequency(classifier, "b") * self.Q)

        Qa = qa
        qa_candidates = []
        for Raa in range(0, Qa + 1):
            for Rbb in range(0, self.Q - Qa + 1):
                if Raa - Rbb - Qa + Rb == 0:
                    distances = self.distances_to_target(Qa, Raa, Rbb, point)
                    qa_candidates.append((distances, (Raa, Rbb)))
        # Since all list items have the same distance to Qa,
        # this sorting returns min total distance
        qa_candidates.sort(key=lambda x: x[0][0] + x[0][1])
        evaluations = qa_candidates[:k]

        return evaluations

    def find_k_nearest_at_prevalence_all_classifiers(
        self, qa: int, points, k: int
    ):
        single_evals = [
            self.find_k_nearest_at_prevalence(
                classifier, qa, points[classifier], k
            )
            for classifier in range(3)
        ]
        joint_evals = []
        for distances0, r0 in single_evals[0]:
            qa_distance = distances0[0]
            r0distance = distances0[1]
            for distances1, r1 in single_evals[1]:
                r1distance = distances1[1]
                for distances2, r2 in single_evals[2]:
                    r2distance = distances2[1]
                    joint_evals.append(
                        (
                            qa_distance + r0distance + r1distance + r2distance,
                            (r0, r1, r2),
                        )
                    )
        joint_evals.sort()
        return joint_evals[:k]

    def distances_to_target(self, Qa, Raa, Rbb, point):
        # point_pspace = self.to_pspace(point)
        pe_pspace = self.to_pspace((Qa, Raa, Rbb))

        return (
            sympy.simplify((point[0] - pe_pspace[0]) ** 2),
            sympy.simplify(
                (point[1] - pe_pspace[1]) ** 2 + (point[2] - pe_pspace[2]) ** 2
            ),
        )

    def to_pspace(self, point):
        if point[0] != 0:
            pia = sympy.Rational(point[1], point[0])
        else:
            pia = sympy.Rational(0, 1)

        if self.Q - point[0] != 0:
            pib = sympy.Rational(point[2], (self.Q - point[0]))
        else:
            pib = sympy.Rational(0, 1)

        return (sympy.Rational(point[0], self.Q), pia, pib)

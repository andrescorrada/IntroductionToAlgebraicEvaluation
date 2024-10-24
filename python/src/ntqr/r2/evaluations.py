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

import sympy

from ntqr.r2.datasketches import VoteCounts


class SingleClassifierEvaluations:
    """
    Single classifier evaluations in (Q_a, Q_b, R_{b_i, a}, R_{a_i,b})
    space.
    """

    def __init__(self, Q, single_axioms):

        self.Q = Q
        self.axioms = single_axioms

    def number_aprior_evaluations(self):
        """
        Calculates all the possible evaluations for a binary response test
        with Q questions.

        Returns
        -------
        int

        """
        q = self.Q
        return 1 / 6 * (q + 1) * (q + 2) * (q + 3)

    def evaluations_at_qa_qb(self, eval_dict):
        """
        Returns all evaluations logically consistent with the
        single classifier axiom given the correct number of each
        label and a classifier's responses.

        In binary classification we have Q_a + Q_b = Q. Thus, we
        really need to specify only two of the three (Q, Q_a, Q_b).
        Making a choice is arbitrary and breaks the symmetry in the
        algebra between the two labels. Instead, we specify Q_a, and
        Q_b and since we have Q from the instance initialization, we
        do a quality check (logic check) of the equality between the
        three quantities.
        """
        questions_number = self.axioms.questions_number
        vars_to_check = [
            question_number for question_number in questions_number.values()
        ]
        vars_to_check += [
            response_variable
            for response_variable in self.axioms.responses.values()
        ]
        assert all([(var in eval_dict) for var in vars_to_check])

        # Copy the input eval dict
        work_dict = eval_dict.copy()

        wrong_vars = [
            wrong_var
            for true_label in self.axioms.labels
            for wrong_var in self.axioms.responses_by_label[true_label][
                "errors"
            ].values()
        ]
        print(wrong_vars)

        q_label_vals = [
            eval_dict[questions_number[label]] for label in self.axioms.labels
        ]
        evals = [
            (rl2l1, rl1l2)
            for rl2l1 in range(0, q_label_vals[0] + 1)
            for rl1l2 in range(0, q_label_vals[1] + 1)
            if self._check_axiom_consistency_(
                work_dict, {wrong_vars[0]: rl2l1, wrong_vars[1]: rl1l2}
            )
        ]

        return evals

    def _check_axiom_consistency_(self, eval_dict, errors_dict):
        eval_dict.update(errors_dict)
        return self.axioms.satisfies_axioms(eval_dict)


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

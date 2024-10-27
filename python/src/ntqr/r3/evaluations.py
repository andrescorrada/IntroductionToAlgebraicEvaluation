"""
Evaluations for binary tests (R=3).

For any finite test there is a finite set of evaluations possible. The
classes in this module compute them for the case of 3 classes or test
responses.

Classes:
    SingleClassifierEvaluations: Class related to the evaluations for a
    single classifier consistent with its observed test responses.

Functions:

Misc variables:

"""

import itertools

import sympy
import ntqr.evaluations


class SingleClassifierEvaluations(
    ntqr.evaluations.SingleClassifierEvaluations
):
    """
    Single classifier evaluations in (Q_a, Q_b, R_{b_i, a}, R_{a_i,b})
    space.
    """

    def __init__(self, Q, single_axioms):

        super().__init__(Q, single_axioms)

    def number_aprior_evaluations(self):
        """
        Calculates all the possible evaluations for a binary response test
        with Q questions.

        Returns
        -------
        int

        """
        Q = self.Q
        prod = 1
        for i in range(1, 10):
            prod *= Q + i
        return prod / 362880  # 9! division

    def errors_at_qs(self, qs, responses):
        """
        Returns all evaluations logically consistent with the
        single classifier axiom given the correct number of each
        label and a classifier's responses.
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
            (first_lbl_wrong, second_lbl_wrong, third_lbl_wrong)
            for first_lbl_wrong in self._label_wrongs_(qs[0])
            for second_lbl_wrong in self._label_wrongs_(qs[1])
            for third_lbl_wrong in self._label_wrongs_(qs[2])
            if self._check_axiom_consistency_(
                eval_dict,
                itertools.chain(*wrong_vars),
                itertools.chain(
                    first_lbl_wrong, second_lbl_wrong, third_lbl_wrong
                ),
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

        max_correct = (0, (0, 0, 0))

        for first_label_wrongs in self._label_wrongs_(qs[0]):
            vars = wrong_vars[0]
            vals = first_label_wrongs
            for second_label_wrongs in self._label_wrongs_(qs[1]):
                vars += wrong_vars[1]
                vals += second_label_wrongs
                for third_label_wrong in self._label_wrongs_(qs[2]):
                    vars += wrong_vars[2]
                    vals += third_label_wrong

                    if self._check_axiom_consistency_(eval_dict, vars, vals):
                        corrects = [
                            ql - sum(wrongs)
                            for ql, wrongs in zip(
                                qs,
                                [
                                    first_label_wrongs,
                                    second_label_wrongs,
                                    third_label_wrong,
                                ],
                            )
                        ]
                        corrects_sum = sum(corrects)
                        if corrects_sum > max_correct[0]:
                            max_correct = (corrects_sum, corrects)

        return max_correct[1]

    def all_qs(self):
        "Returns all possible question numbers."
        Q = self.Q
        return [
            (qa, qb, Q - qa - qb)
            for qa in range(0, self.Q + 1)
            for qb in range(0, self.Q - qa + 1)
        ]

    def _label_wrongs_(self, q):
        return [
            (w1, w2) for w1 in range(0, q + 1) for w2 in range(0, q - w1 + 1)
        ]

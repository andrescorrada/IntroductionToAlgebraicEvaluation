"""Module for R-axioms related super classess."""


class SingleClassifierEvaluations:
    """
    Single classifier evaluations
    """

    def __init__(self, Q, single_axioms):

        self.Q = Q
        self.axioms = single_axioms

    def correct_at_qs(self, qs, responses):
        errors_at_qs = self.errors_at_qs(qs, responses)
        return set(
            ((ql - sum(label_errors)) for ql, label_errors in zip(qs, errors))
            for errors in errors_at_qs
        )

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

    def _check_axiom_consistency_(self, eval_dict, wrong_vars, wrong_vals):
        eval_dict.update(
            {var: val for var, val in zip(wrong_vars, wrong_vals)}
        )
        return self.axioms.satisfies_axioms(eval_dict)

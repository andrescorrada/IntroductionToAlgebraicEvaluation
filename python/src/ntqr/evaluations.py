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

    def _check_axiom_consistency_(self, eval_dict, wrong_vars, wrong_vals):
        eval_dict.update(
            {var: val for var, val in zip(wrong_vars, wrong_vals)}
        )
        return self.axioms.satisfies_axioms(eval_dict)

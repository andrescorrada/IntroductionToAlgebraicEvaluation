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

import sympy


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
        Q = self.Q
        prod = 1
        for i in range(1, 10):
            prod *= Q + i
        return prod / 362880  # 9! division

    def evaluations_at_qa_qb(self, eval_dict):
        """
        Returns all evaluations logically consistent with the
        single classifier axiom given the correct number of each
        label and a classifier's responses.
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
            [
                wrong_var
                for wrong_var in self.axioms.responses_by_label[true_label][
                    "errors"
                ].values()
            ]
            for true_label in self.axioms.labels
        ]

        q_label_vals = [
            eval_dict[questions_number[label]] for label in self.axioms.labels
        ]
        evals = set(
            [
                ((rl2l1, rl3l1), (rl1l2, rl3l2), (rl1l3, rl2l3))
                for rl2l1 in range(0, q_label_vals[0] + 1)
                for rl3l1 in range(0, q_label_vals[0] - rl2l1 + 1)
                for rl1l2 in range(0, q_label_vals[1] + 1)
                for rl3l2 in range(0, q_label_vals[1] - rl1l2 + 1)
                for rl1l3 in range(0, q_label_vals[2] + 1)
                for rl2l3 in range(0, q_label_vals[2] - rl1l3 + 1)
                if self._check_axiom_consistency_(
                    work_dict,
                    wrong_vars,
                    ((rl2l1, rl3l1), (rl1l2, rl3l2), (rl1l3, rl2l3)),
                )
            ]
        )

        return evals

    def _check_axiom_consistency_(self, eval_dict, wrong_vars, wrong_vals):
        for vars, vals in zip(wrong_vars, wrong_vals):
            eval_dict.update({var: val for var, val in zip(vars, vals)})
        return self.axioms.satisfies_axioms(eval_dict)

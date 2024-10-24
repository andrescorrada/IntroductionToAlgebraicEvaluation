"""Tests related to the ntqr.r3.raxioms module."""

import ntqr

r3axioms = ntqr.r3.raxioms.SingleClassifierAxioms(["a", "b", "c"], "k")

mul = 2
qa = 2 * mul
qb = 3 * mul
qc = 4 * mul

# Setting the question numbers
eval_dict = {
    qlabel: val
    for qlabel, val in zip(r3axioms.questions_number.values(), [qa, qb, qc])
}

# Setting the observed label responses
eval_dict.update(
    {
        response_var: count
        for response_var, count in zip(
            r3axioms.responses.values(), [1 * mul, 2 * mul, 6 * mul]
        )
    }
)


def test_r3_single_classifier_axioms():
    rbl_vars = r3axioms.responses_by_label
    eval_dict.update(
        {
            rbl_vars["a"]["b"]: 1,
            rbl_vars["a"]["c"]: 2,
            rbl_vars["b"]["a"]: 0,
            rbl_vars["b"]["c"]: 5,
            rbl_vars["c"]["a"]: 1,
            rbl_vars["c"]["b"]: 2,
        }
    )
    evaluated_axioms = r3axioms.evaluate_axioms(eval_dict)
    assert evaluated_axioms == {"a": 0, "b": 0, "c": 0}

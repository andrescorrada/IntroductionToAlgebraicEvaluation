"""Tests to check the ntqr.r2.raxioms module."""

import pytest

import ntqr.r2.raxioms

labels = ["a", "b"]
classifier = "k"


r2axioms = ntqr.r2.raxioms.SingleClassifierAxioms(labels, classifier)

eval_dict = {
    var: val for var, val in zip(r2axioms.questions_number.values(), [3, 7])
}
eval_dict.update(
    {var: val for var, val in zip(r2axioms.responses.values(), [5, 5])}
)

wrong_vars = [
    var
    for true_label in r2axioms.labels
    for var in r2axioms.responses_by_label[true_label]["errors"].values()
]


@pytest.mark.parametrize(
    "update_dict, satisfies_axioms",
    (
        ({var: val for var, val in zip(wrong_vars, (0, 2))}, True),
        ({var: val for var, val in zip(wrong_vars, (1, 2))}, False),
    ),
)
def test_evaluations_at_qa_qb(update_dict, satisfies_axioms):
    eval_dict.update(update_dict)
    test_val = r2axioms.satisfies_axioms(eval_dict)
    assert test_val == satisfies_axioms

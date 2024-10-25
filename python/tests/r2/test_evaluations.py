"""Tests for the ntqr.r2.evaluations module."""

import ntqr.r2.raxioms
import ntqr.r2.evaluations

labels = ["a", "b"]
classifier = "k"


r2axioms = ntqr.r2.raxioms.SingleClassifierAxioms(labels, classifier)
scEvaluations = ntqr.r2.evaluations.SingleClassifierEvaluations(10, r2axioms)

eval_dict = {
    var: val for var, val in zip(r2axioms.questions_number.values(), [3, 7])
}
eval_dict.update(
    {var: val for var, val in zip(r2axioms.responses.values(), [5, 5])}
)


def test_evaluations_at_qa_qb():
    evals = scEvaluations.evaluations_at_qa_qb(eval_dict)
    assert evals == [(0, 2), (1, 3), (2, 4), (3, 5)]

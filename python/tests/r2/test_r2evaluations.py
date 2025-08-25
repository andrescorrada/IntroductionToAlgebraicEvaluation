"""Tests for the ntqr.r2.evaluations module."""

import ntqr.r2.raxioms
import ntqr.r2.evaluations

labels = ["a", "b"]
classifier = "k"


r2axioms = ntqr.r2.raxioms.SingleClassifierAxioms(labels, classifier)
scEvaluations = ntqr.r2.evaluations.SingleClassifierEvaluations(10, r2axioms)


def test_errors_at_qs():
    evals = scEvaluations.errors_at_qs([3, 7], [5, 5])
    assert evals == set(
        [((0,), (2,)), ((1,), (3,)), ((2,), (4,)), ((3,), (5,))]
    )


def test_max_correct_at_qa_qb():
    max_correct = scEvaluations.max_correct_at_qs([3, 7], [5, 5])
    assert max_correct == [3, 5]

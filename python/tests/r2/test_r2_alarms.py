"""Tests for the module ntqr.r2.alarms."""

import ntqr.r2.raxioms
import ntqr.r2.alarms

import pytest

labels = ("a", "b")
c1Axioms = ntqr.r2.raxioms.SingleClassifierAxioms(labels, "1")
c2Axioms = ntqr.r2.raxioms.SingleClassifierAxioms(labels, "2")
alarm = ntqr.r2.alarms.SingleClassifierAxiomAlarm(10, [c1Axioms, c2Axioms])

# Better than 50%
factors = (2, 2)
qs = (4, 6)


@pytest.mark.parametrize(
    "qs, factors, corrects, satisfies",
    (
        (qs, factors, (3, 4), True),
        (qs, factors, (2, 4), False),
        (qs, factors, (3, 3), False),
    ),
)
def test_single_classifier_alarm_generate_safety_specification(
    qs, factors, corrects, satisfies
):
    satisfies_safety_specification = alarm.generate_safety_specification(
        factors
    )
    assert satisfies_safety_specification(qs, corrects) == satisfies


@pytest.mark.parametrize(
    "qs, factors, responses, misaligned",
    (
        (qs, factors, ((2, 8), (3, 7)), False),
        (qs, factors, ((2, 8), (8, 2)), True),
    ),
)
def test_misalignment_alarm_at_qa_qb(qs, factors, responses, misaligned):
    satisfies_safety_spec = alarm.generate_safety_specification(factors)
    assert (
        alarm.misaligned_at_qs(satisfies_safety_spec, qs, responses)
        == misaligned
    )

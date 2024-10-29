"""Tests for the module ntqr.r2.alarms."""

import ntqr.r2.raxioms
import ntqr.alarms
import ntqr.r2.evaluations

import pytest

labels = ("a", "b")
c1Axioms = ntqr.r2.raxioms.SingleClassifierAxioms(labels, "1")
c2Axioms = ntqr.r2.raxioms.SingleClassifierAxioms(labels, "2")


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
    safety_specification = ntqr.alarms.LabelsSafetySpecification(factors)
    assert safety_specification.is_satisfied(qs, corrects) == satisfies


alarm = ntqr.alarms.SingleClassifierAxiomsAlarm(
    10,
    [c1Axioms, c2Axioms],
    ntqr.r2.evaluations.SingleClassifierEvaluations,
)
alarm.set_safety_specification(factors)


@pytest.mark.parametrize(
    "alarm, qs, factors, responses, misaligned",
    (
        (alarm, qs, factors, ((2, 8), (3, 7)), True),
        (alarm, qs, factors, ((4, 6), (3, 7)), False),
        (alarm, qs, factors, ((2, 8), (8, 2)), True),
    ),
)
def test_misalignment_alarm_at_qs(alarm, qs, factors, responses, misaligned):
    misalignment_test = alarm.misaligned_at_qs(qs, responses)
    assert misalignment_test == misaligned


alarm = ntqr.alarms.SingleClassifierAxiomsAlarm(
    20,
    [c1Axioms, c2Axioms],
    ntqr.r2.evaluations.SingleClassifierEvaluations,
)
alarm.set_safety_specification(factors)


@pytest.mark.parametrize(
    "alarm, factors, responses, misaligned",
    (
        (alarm, factors, ((16, 4), (17, 3)), False),
        (alarm, factors, ((16, 4), (1, 19)), True),
    ),
)
def test_misalignment_alarm(alarm, factors, responses, misaligned):
    assert alarm.are_misaligned(responses) == misaligned

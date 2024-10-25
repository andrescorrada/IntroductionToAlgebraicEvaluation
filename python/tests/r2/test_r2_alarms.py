"""Tests for the module ntqr.r2.alarms."""

import ntqr.r2.raxioms
import ntqr.r2.alarms

import pytest

labels = ("a", "b")
c1Axioms = ntqr.r2.raxioms.SingleClassifierAxioms(labels, "1")
c2Axioms = ntqr.r2.raxioms.SingleClassifierAxioms(labels, "2")
alarm = ntqr.r2.alarms.SingleClassifierAxiomAlarm([c1Axioms, c2Axioms])

# Better than 50%
factors = (2, 2, 2)
qs = (4, 6, 8)


@pytest.mark.parametrize(
    "qs, factors, corrects, satisfies",
    (
        (qs, factors, (3, 4, 5), True),
        (qs, factors, (2, 4, 5), False),
        (qs, factors, (3, 3, 5), False),
        (qs, factors, (3, 4, 4), False),
    ),
)
def test_single_classifier_alarm_generate_safety_specification(
    qs, factors, corrects, satisfies
):
    satisfies_safety_specification = alarm.generate_safety_specification(
        factors
    )
    assert satisfies_safety_specification(qs, corrects) == satisfies

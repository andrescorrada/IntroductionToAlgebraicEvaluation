import ntqr.alarms
import ntqr.r3.raxioms
import ntqr.r3.evaluations

import pytest

labels = ("a", "b", "c")
c1Axioms = ntqr.r3.raxioms.SingleClassifierAxioms(labels, "1")
c2Axioms = ntqr.r3.raxioms.SingleClassifierAxioms(labels, "2")
alarm = ntqr.alarms.SingleClassifierAxiomsAlarm(
    18, [c1Axioms, c2Axioms], ntqr.r3.evaluations.SingleClassifierEvaluations
)

# Better than 50%
factors = (2, 2, 2)
alarm.set_safety_specification(factors)
qs = (4, 6, 8)


@pytest.mark.parametrize(
    "qs, factors, corrects, satisfies",
    (
        (qs, factors, (3, 4, 5), True),
        (qs, factors, (2, 4, 5), False),
        (qs, factors, (3, 3, 5), False),
    ),
)
def test_safety_specification(qs, factors, corrects, satisfies):
    safety_specification = ntqr.alarms.LabelsSafetySpecification(factors)
    assert safety_specification.is_satisfied(qs, corrects) == satisfies


@pytest.mark.parametrize(
    "qs, responses, misaligned",
    (
        (qs, ((4, 7, 7), (3, 7, 8)), False),
        (qs, ((10, 8, 0), (0, 8, 10)), True),
    ),
)
def test_misalignment_alarm_at_qs(qs, responses, misaligned):

    assert alarm.misaligned_at_qs(qs, responses) == misaligned

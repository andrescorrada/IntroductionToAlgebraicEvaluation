import ntqr.alarms
import ntqr.r3.evaluations

import pytest

labels = ("a", "b", "c")
c1Axioms = ntqr.r3.raxioms.SingleClassifierAxioms(labels, "1")
c2Axioms = ntqr.r3.raxioms.SingleClassifierAxioms(labels, "2")
alarm = ntqr.alarms.SingleClassifierAxiomAlarm(
    18, [c1Axioms, c2Axioms], ntqr.r3.evaluations.SingleClassifierEvaluations
)

# Better than 50%
factors = (2, 2, 2)
qs = (4, 6, 8)


@pytest.mark.parametrize(
    "qs, factors, corrects, satisfies",
    (
        (qs, factors, (3, 4, 5), True),
        (qs, factors, (2, 4, 5), False),
        (qs, factors, (3, 3, 5), False),
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
        (qs, factors, ((4, 7, 7), (3, 7, 8)), False),
        (qs, factors, ((10, 8, 0), (0, 8, 10)), True),
    ),
)
def test_misalignment_alarm_at_qs(qs, factors, responses, misaligned):
    satisfies_safety_spec = alarm.generate_safety_specification(factors)
    assert (
        alarm.misaligned_at_qs(satisfies_safety_spec, qs, responses)
        == misaligned
    )

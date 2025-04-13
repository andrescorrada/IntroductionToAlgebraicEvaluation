"""@author: Andrés Corrada-Emmanuel."""
import itertools

import pytest

from ntqr import Label, Labels

from ntqr.evaluations import QuestionAlignedDecisions
from ntqr.r2.examples import uciadult_label_counts

#
# Testing R=2 QuestionAlignedDecisions functionality
#


@pytest.mark.parametrize(
    "observed_responses, error_msg",
    (
        ({}, "There must be at least one decision key."),
        ({("a", "b"): 1, ("a",): 3}, ""),
    ),
)
def test_questionaligneddecisions_valueerror(observed_responses, error_msg):
    with pytest.raises(ValueError, match=error_msg):
        QuestionAlignedDecisions(observed_responses, ("a", "b"))


#
# Testing marginalization of QuestionAlignedDecisions
#
labels = ("a", "b")
observed_responses = {}
for label, decisions_counts in uciadult_label_counts.items():
    for decisions, count in decisions_counts.items():
        observed_responses[decisions] = (
            observed_responses.get(decisions, 0) + count
        )


@pytest.mark.parametrize(
    "indices, counts",
    (
        ((0,), {("a",): 15389, ("b",): 21453}),
        ((1,), {("a",): 2671, ("b",): 34171}),
        ((2,), {("a",): 15061, ("b",): 21781}),
    ),
)
def test_questionaligneddecisions_marginalize(indices, counts):
    qads = QuestionAlignedDecisions(observed_responses, labels)
    assert qads.marginalize(indices).counts == counts

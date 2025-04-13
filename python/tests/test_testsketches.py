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

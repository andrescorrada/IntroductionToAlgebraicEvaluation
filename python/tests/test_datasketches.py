"""@author: Andrés Corrada-Emmanuel."""
from fractions import Fraction

import pytest

from ntqr.r2.datasketches import TrioVoteCounts, TrioLabelVoteCounts
from ntqr.r2.examples import uciadult_label_counts

#
# Testing TrioLabelVoteCounts functionality
#


def test_triolabelvotecounts_init():
    tlvc = TrioLabelVoteCounts(uciadult_label_counts)

    assert isinstance(tlvc, TrioLabelVoteCounts)

#
# Testing TrioVoteCounts functionality
#


def test_triovotecounts_init():

    tlvc = TrioLabelVoteCounts(uciadult_label_counts)
    tvc = tlvc.to_TrioVoteCounts()

    assert isinstance(tvc, TrioVoteCounts)


tvc = TrioLabelVoteCounts(uciadult_label_counts).to_TrioVoteCounts()
uci_freqs = tvc.to_frequencies_exact()


@pytest.mark.parametrize("freqs, votes,freq", (
    (
         uci_freqs,
         ('a', 'a', 'a'),
         Fraction(493, 18421)
     ),))
def test_frequency(freqs, votes, freq):
    assert freqs[votes] == freq

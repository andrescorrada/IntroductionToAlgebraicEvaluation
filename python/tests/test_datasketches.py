"""@author: Andrés Corrada-Emmanuel."""
from ntqr.r2.datasketches import TrioVoteCounts, TrioLabelVoteCounts
from ntqr.r2.examples import uciadult_label_counts


# Testing TrioVoteCounts functionality
def test_triolabelvotecounts_init():
    tvc = TrioLabelVoteCounts(uciadult_label_counts)

    assert isinstance(tvc, TrioLabelVoteCounts)

"""@author: Andrés Corrada-Emmanuel."""
import itertools
from fractions import Fraction

import pytest

from ntqr.r2.datasketches import (
    TrioVoteCounts,
    TrioLabelVoteCounts,
    classifier_label_votes,
)
from ntqr.r2.examples import uciadult_label_counts

trio_vote_patterns = list(itertools.product(*["ab" for i in range(3)]))


def test_classifier_a_label_votes():
    vote_patterns = classifier_label_votes(0, "a", trio_vote_patterns)
    assert vote_patterns == (
        ("a", "a", "a"),
        ("a", "a", "b"),
        ("a", "b", "a"),
        ("a", "b", "b"),
    )


def test_classifier_b_label_votes():
    vote_patterns = classifier_label_votes(0, "b", trio_vote_patterns)
    assert vote_patterns == (
        ("b", "a", "a"),
        ("b", "a", "b"),
        ("b", "b", "a"),
        ("b", "b", "b"),
    )


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


@pytest.mark.parametrize(
    "freqs, votes,freq",
    (
        (uci_freqs, ("a", "a", "a"), Fraction(493, 18421)),
        (uci_freqs, ("a", "b", "a"), Fraction(5801, 36842)),
    ),
)
def test_frequency(freqs, votes, freq):
    assert freqs[votes] == freq


pair_moments = tvc.label_pairs_frequency_moments("b")


@pytest.mark.parametrize(
    "pair_moments, pair, freq",
    (
        (pair_moments, (0, 1), Fraction(18432653, 1357332964)),
        (pair_moments, (0, 2), Fraction(18272925, 1357332964)),
        (pair_moments, (1, 2), Fraction(16803485, 1357332964)),
    ),
)
def test_pair_moment(pair_moments, pair, freq):
    assert pair_moments[pair] == freq

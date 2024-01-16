"""@author: Andrés Corrada-Emmanuel."""
import math

# The mathematics of evaluating finite samples is, by construction, one
# of estimating integer fractions. We import this module so we can create
# two versions of every computation - the exact one using integer ratios,
# and the one using the default floating point numbers
from fractions import Fraction
from typing_extensions import Union, Literal, Mapping, Iterable
from dataclasses import dataclass, field

# Types
Label = Union[Literal["a"], Literal["b"]]


# A vote is an ordered tuple of the decisions
Votes = tuple[Label, ...]

# Vote counts are what we see when we have unlabeled data.
VoteCounts = Mapping[Votes, int]

# Label vote counts, vote counts by true label, are only available
# when we are carrying out experiments on labeled data.
LabelVoteCounts = Mapping[Label, VoteCounts]

VoteFrequencies = Mapping[Votes, Fraction]


# Some definitions and utilities to help various classes

# Three binary classifiers have eight possible voting patterns
trio_vote_patterns: tuple[tuple[Label, ...], ...] = (
    ("a", "a", "a"),
    ("a", "a", "b"),
    ("a", "b", "a"),
    ("a", "b", "b"),
    ("b", "a", "a"),
    ("b", "a", "b"),
    ("b", "b", "a"),
    ("b", "b", "b"),
)

trio_pairs = ((0, 1), (0, 2), (1, 2))


# To compute algebraic functions of classifier statistics,
# we need helper functions that pick out the trio vote patterns
# where a classifer voted with a given label.
def classifier_label_votes(classifier: int, label: Label) -> tuple[Votes, ...]:
    """
    All the trio vote patterns where classifier voted with label.

    Parameters
    ----------
    classifier : int
        Index of the classifier, one of (0, 1, 2).
    label : Label
        The classifier label vote.

    Returns
    -------
    tuple[Votes, ...]
        All the trio vote patterns where classifier voted with label.

    """
    return tuple(
        [votes for votes in trio_vote_patterns if votes[classifier] == label]
    )


def votes_match(
    votes: tuple[Label, ...],
    classifiers: Iterable[int],
    labels: Iterable[Label],
) -> bool:
    """Test if labels by classifiers matches votes.

    Parameters
    ----------
    votes : Tuple[Label, ...]
        Vote pattern for the trio.
    classifiers : Iterable[int]
        Classifiers to check.
    labels : Iterable[Label]
        Label voted by each classifier - the voting
        sub-pattern we want a trio pattern to match.

    Returns
    -------
    bool.

    """
    return all(
        [
            votes[classifier] == label
            for classifier, label in zip(classifiers, labels)
        ]
    )


def classifiers_labels_votes(
    classifiers: Iterable[int], labels: Iterable[Label]
) -> tuple[Votes, ...]:
    """Trio voting patterns that match labels by classifiers.

    Parameters
    ----------
    classifiers : Iterable[int]
        The indices of the classifiers in the trio
        to match.
    labels : Iterable[Label]
        Voting sub-pattern by classifiers to match.

    Returns
    -------
    tuple[Votes, ...]
        Tuple of trio voting patterns that matched labels
        for the classifiers.

    """
    return tuple(
        [
            votes
            for votes in trio_vote_patterns
            if votes_match(votes, classifiers, labels)
        ]
    )


@dataclass(frozen=True)
class TrioLabelVoteCounts:
    """
    Data class for the by-label aligned votes of three binary classifiers.

    Initialized with a Mapping[Label, Mapping[Votes, int]] of the form
    {'a':
     {('a', 'a', 'a'): int, ..., ('b', 'b', 'b'):int},
     'b':
     {('a', 'a', 'a'): int, ..., ('b', 'b', 'b'):int}}
    """

    label_vote_counts: Mapping[Label, Mapping[Votes, int]]

    def __post_init__(self):
        """All labels and vote patterns are set."""
        object.__setattr__(
            self,
            "label_vote_counts",
            {label:
             {votes: self.label_vote_counts.get(label, {}).get(votes, 0)
              for votes in trio_vote_patterns}
             for label in ('a', 'b')})

    def to_vote_counts(self) -> VoteCounts:
        """
        Turn by-label counts into by-vote-pattern counts.

        Using {'a':{..., ('a', 'b', 'a'): x, ...},
               'b':{..., ('a', 'b', 'a'): y, ...}}

        Returns
        -------
              {..., ('a', 'b', 'a'): x+y, ...}
        """
        return {
            votes: sum([
                self.label_vote_counts[label][votes]
                for label in ('a', 'b')])
            for votes in trio_vote_patterns}

    def to_TrioVoteCounts(self):
        """Return TrioVoteCounts object by summing votes across labels."""
        return TrioVoteCounts(self.to_vote_counts())

    def to_voting_frequency_fractions(self) -> VoteFrequencies:
        """
        Computes observed voting pattern frequencies.
        """
        by_voting_counts = self.to_vote_counts()
        size_of_test = sum(by_voting_counts.values())
        return {
            vp: Fraction(by_voting_counts[vp], size_of_test)
            for vp in by_voting_counts.keys()
        }

    def to_voting_frequencies_float(self) -> Mapping[Votes, float]:
        """
        Same as the exact computation, but using floating point
        numbers.
        """
        by_voting_counts = self.to_vote_counts()
        size_of_test = sum(by_voting_counts.values())
        return {
            vp: by_voting_counts[vp] / size_of_test
            for vp in by_voting_counts.keys()
        }

    def __getitem__(self, label):
        return self.lbl_vote_counts.get(label)

@dataclass(frozen=True)
class TrioVoteCounts:
    """Dataclass to validate the test counts for three binary classifiers."""

    vote_counts: VoteCounts

    def __post_init__(self):
        """Check that no negative vote counts."""
        if any([count for count in self.vote_counts.values() if (count < 0)]):
            raise ValueError("No negative vote counts allowed.")
        # Make sure all patterns have a non-negative count.
        self.vote_counts = {vote_pattern: self.vote_counts.get(vote_pattern, 0)
                            for vote_pattern in trio_vote_patterns}
        # The empty test is not allowed
        if sum(self.vote_counts.values()) == 0:
            raise ValueError("The empty test is not allowed.")

    def to_frequencies_exact(self) -> VoteFrequencies:
        """
        Turn vote integer counts to exact Fraction objects.

        Returns
        -------
        VoteFrequencies:
            mapping of trio votes to percentage of occurence in the test
            as Fraction.
        """
        return {
            vp: Fraction(self.vote_counts[vp], self.test_size)
            for vp in self.vote_counts.keys()
        }

    def to_frequencies_float(self) -> Mapping[Votes, float]:
        """
        Compute observerd voting pattern frequencies as inexact floats.

        Returns
        -------
        Mapping[Votes, float]

        """
        return {
            vp: self.vote_counts[vp] / self.test_size
            for vp in self.vote_counts.keys()
        }

    def classifier_label_frequency(
        self, classifier: int, label: Label
    ) -> Fraction:
        """
        Calculate classifier label voting frequency.

        Parameters
        ----------
        classifier : int
            The index of the classifier.
        label : Label
            The label.

        Returns
        -------
        Fraction(label_vote_counts, test_size)

        """
        vote_frequencies = self.to_frequencies_exact()
        return sum(
            [
                vote_frequencies[votes]
                for votes in classifier_label_votes(classifier, label)
            ]
        )

    def pair_label_frequency(
        self, pair: Iterable[int], label: Label
    ) -> Fraction:
        """
        Compute frequency of times a pair voted with the same label.

        Parameters
        ----------
        pair : Iterable[int, int]
            Classifier indicies.
        label : Label
            The label.

        Returns
        -------
        Fraction

        """
        vote_frequencies = self.to_frequencies_exact()
        return sum(
            [
                vote_frequencies[votes]
                for votes in classifiers_labels_votes(pair, (label, label))
            ]
        )

    def pair_frequency_moment(
        self, pair: Iterable[int], label: Label
    ) -> Fraction:
        """
        Calculate the label classifier pair frequency moment.

        Parameters
        ----------
        pair : Iterable(int, int)
            The pair of classifiers.
        label : Label
            One of the binary labels.

        Returns
        -------
        The pair frequency as a Fraction object
        """
        label_frequencies = [
            self.classifier_label_frequency(classifier, label)
            for classifier in pair
        ]
        return self.pair_label_frequency(pair, label) - math.prod(
            label_frequencies
        )

    def label_pairs_frequency_moments(
        self, label: Label
    ) -> Mapping[tuple[int, int], Fraction]:
        """All the label pair frequency moments."""
        return {
            pair: self.pair_frequency_moment(pair, label)
            for pair in trio_pairs
        }

    def trio_frequency_moment(self) -> Fraction:
        """Calculates the 3rd frequency moment of a trio of binary classifiers
        needed for algebraic evaluation using the error indepedent model."""
        classifier_label_frequencies = [
            self.classifier_label_frequency(classifier, "b")
            for classifier in range(3)
        ]
        prod_frequencies = math.prod(classifier_label_frequencies)
        pfmds = self.label_pairs_frequency_moments("b")
        sum_prod = (
            classifier_label_frequencies[0] * pfmds[(1, 2)]
            + classifier_label_frequencies[1] * pfmds[(0, 2)]
            + classifier_label_frequencies[2] * pfmds[(0, 1)]
        )
        return prod_frequencies + sum_prod


# TODO: the operation of turning integer counts to percentages of observations
# ocurrs in both TrioVoteCounts and TrioLabelVoteCounts. We should be careful
# to not cross semantic meanings so the classes make sense as denoting
# the observed votes in a test and the unknown label counts.
# Parallel to this is the type of these things. As data structures, it
# would make sense for TrioVoteCounts to be used by the TrioLabelVoteCounts.
# But this conflicts with the semantic meaning of these classes. For this
# reason alone, the code below contains a constructor of TrioVoteCounts from
# a TrioLabelVoteCounts object.

"""@author: Andrés Corrada-Emmanuel."""
import math, itertools

# The mathematics of evaluating finite samples is, by construction, one
# of estimating integer fractions. We import this module so we can create
# two versions of every computation - the exact one using integer ratios,
# and the one using the default floating point numbers
import sympy
from typing_extensions import Union, Literal, Mapping, Iterable
from dataclasses import dataclass

# Types
Label = Union[Literal["a"], Literal["b"]]


# A vote is an ordered tuple of the decisions
Votes = tuple[Label, ...]

# Vote counts are what we see when we have unlabeled data.
VoteCounts = Mapping[Votes, int]

# Label vote counts, vote counts by true label, are only available
# when we are carrying out experiments on labeled data.
LabelVoteCounts = Mapping[Label, VoteCounts]

VoteFrequencies = Mapping[Votes, sympy.Rational]


# Some definitions and utilities to help various classes
opposite_label = {"a": "b", "b": "a"}


# To compute algebraic functions of classifier statistics,
# we need helper functions that pick out the vote patterns
# where a classifer voted with a given label.
def classifier_label_votes(
    classifier: int, label: Label, vote_patterns: Iterable[Votes]
) -> tuple[Votes, ...]:
    """
    All the vote patterns where classifier voted with label.

    Parameters
    ----------
    classifier : int
        Index of the classifier in the vote_patterns.
    label : Label
        The classifier label vote.

    Returns
    -------
    tuple[Votes, ...]
        All the vote patterns where classifier voted with label.
    """
    return tuple(
        [votes for votes in vote_patterns if votes[classifier] == label]
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
    classifiers: Iterable[int],
    labels: Iterable[Label],
    vote_patterns: Iterable[Votes],
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
            for votes in vote_patterns
            if votes_match(votes, classifiers, labels)
        ]
    )


class TrioLabelVoteCounts:
    """
    Data class for the by-label aligned votes of three binary classifiers.

    This class is only useful in an experimental setting where one has
    observed a test with **labeled** data.
    Initialized with a Mapping[Label, Mapping[Votes, int]] of the form
    {
     'a':{('a', 'a', 'a'): int, ..., ('b', 'b', 'b'):int},
     'b':{('a', 'a', 'a'): int, ..., ('b', 'b', 'b'):int}
    }

    DEPRECATED: This clas is being replaced with the upcoming LabelVoteCounts
    for an arbitrary number of classifiers.
    """

    vote_patterns = list(itertools.product(*["ab" for i in range(3)]))
    pairs = ((0, 1), (0, 2), (1, 2))

    def __init__(self, label_vote_counts):
        """All labels and vote patterns are set."""

        self.label_vote_counts = {
            label: {
                votes: label_vote_counts.get(label, {}).get(votes, 0)
                for votes in self.vote_patterns
            }
            for label in ("a", "b")
        }

        # self.label_vote_counts = {"b":  1}

        object.__setattr__(
            self,
            "test_sizes",
            {
                label: sum(self.label_vote_counts[label].values())
                for label in ("a", "b")
            },
        )

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
            votes: sum(
                [self.label_vote_counts[label][votes] for label in ("a", "b")]
            )
            for votes in self.vote_patterns
        }

    def to_TrioVoteCounts(self):
        """Return TrioVoteCounts object by summing votes across labels."""
        return TrioVoteCounts(self.to_vote_counts())

    def to_voting_frequency_fractions(self) -> VoteFrequencies:
        """
        Compute observed voting pattern frequencies.

        Returns
        -------
        Mapping[Votes, Fraction]
        """
        by_voting_counts = self.to_vote_counts()
        size_of_test = sum(by_voting_counts.values())
        return {
            vp: sympy.Rational(by_voting_counts[vp], size_of_test)
            for vp in by_voting_counts.keys()
        }

    def to_voting_frequencies_float(self) -> Mapping[Votes, float]:
        """
        Compute observed voting frequencies inexactly, as floats.

        Returns
        -------
        Mapping[Votes, float]
        """
        by_voting_counts = self.to_vote_counts()
        size_of_test = sum(by_voting_counts.values())
        return {
            vp: by_voting_counts[vp] / size_of_test
            for vp in by_voting_counts.keys()
        }

    def __getitem__(self, label):
        """
        Return the vote pattern counts for label.

        Parameters
        ----------
        label :Label
            One of 'a' or 'b'.

        Returns
        -------
        Mapping[Votes, int]
            The aligned vote counts observed for the given label.
        """
        return self.label_vote_counts[label]

    def flip_classifiers_label_decisions(
        self, classifiers: Iterable, label: Label
    ):
        flipped_data_sketch = {}
        opp_label = opposite_label[label]
        flipped_data_sketch[opp_label] = self.label_vote_counts[opp_label]
        flipped_vote_counts = {}
        for vp, count in self.label_vote_counts[label].items():
            new_pattern = tuple(
                [
                    opposite_label[vp[i]] if i in classifiers else vp[i]
                    for i in range(3)
                ]
            )
            flipped_vote_counts[new_pattern] = count
        flipped_data_sketch[label] = flipped_vote_counts

        return TrioLabelVoteCounts(flipped_data_sketch)


@dataclass
class TrioVoteCounts:
    """Data class to validate the test counts for three binary classifiers.

    Initialized with a Mapping[Votes, int] of the form:
        {('a', 'a', 'a'): int, ..., ('b', 'b', 'b'):int}

    This is the class that is used for evaluation on unlabeled data where
    we only have access to the aligned decisions of the binary classifiers
    and have no knowledge of label of any one item that was classified.

    DEPRECATED: This class will be replaced with the ObservedVoteCounts class
    that can handle an arbitrary number of classifiers.
    """

    # Three binary classifiers have eight possible voting patterns
    vote_patterns = list(itertools.product(*["ab" for i in range(3)]))

    pairs = ((0, 1), (0, 2), (1, 2))

    vote_counts: VoteCounts

    def __post_init__(self):
        """
        Check we have counts for a valid evaluation of binary classifiers.

        1. No negative counts.
        2. Initialize all possible vote patterns by the trio.
        3. The empty test - all counts zero - is not allowed.
        """
        # Make sure all patterns have a non-negative count.
        object.__setattr__(
            self,
            "vote_counts",
            {
                vote_pattern: self.vote_counts.get(vote_pattern, 0)
                for vote_pattern in self.vote_patterns
            },
        )

        if any([count for count in self.vote_counts.values() if (count < 0)]):
            raise ValueError("No negative vote counts allowed.")

        object.__setattr__(self, "test_size", sum(self.vote_counts.values()))

        # The empty test is not allowed
        if self.test_size == 0:
            raise ValueError("The empty test is not allowed.")

    def to_frequencies_exact(self) -> VoteFrequencies:
        """
        Turn vote integer counts to exact Fraction objects.

        Returns
        -------
        VoteFrequencies:
            Maps a trio vote pattern to its Fraction occurence in the test.
        """
        return {
            vp: sympy.Rational(self.vote_counts[vp], self.test_size)
            for vp in self.vote_counts.keys()
        }

    def to_frequencies_float(self) -> Mapping[Votes, float]:
        """
        Compute observerd voting pattern frequencies, inexactly, as floats.

        Returns
        -------
        Mapping[Votes, float]:
            Maps a trio vote pattern to its percentage occurence in the test
            as an inexact float.

        """
        return {
            vp: self.vote_counts[vp] / self.test_size
            for vp in self.vote_counts.keys()
        }

    def classifier_label_frequency(
        self, classifier: int, label: Label
    ) -> sympy.Rational:
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
        sympy.Rational(label_vote_counts, test_size):
            The fraction of times the classifier voted the label when
            classifying items in the test.

        """
        vote_frequencies = self.to_frequencies_exact()
        return sum(
            [
                vote_frequencies[votes]
                for votes in classifier_label_votes(
                    classifier, label, self.vote_patterns
                )
            ]
        )

    def classifier_label_responses(self, classifier: int, label: Label) -> int:
        """
        Calculates number of responses with label by classifier.

        Parameters
        ----------
        classifier : int
            DESCRIPTION.
        label : Label
            DESCRIPTION.

        Returns
        -------
        int
            Number of times the classifier decided an item was label.

        """
        return sum(
            [
                self.vote_counts[votes]
                for votes in classifier_label_votes(
                    classifier, label, self.vote_patterns
                )
            ]
        )

    def pair_label_frequency(
        self, pair: Iterable[int], label: Label
    ) -> sympy.Rational:
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
        sympy.Rational:
            The fraction of times a pair of classifiers voted with the
            same label when classifying items in the test.

        """
        vote_frequencies = self.to_frequencies_exact()
        return sum(
            [
                vote_frequencies[votes]
                for votes in classifiers_labels_votes(
                    pair, (label, label), self.vote_patterns
                )
            ]
        )

    def pair_label_responses(
        self, pair: Iterable[int], label: Label
    ) -> sympy.Rational:
        """
        Computes number of times a pair voted with the same label.

        Parameters
        ----------
        pair : Iterable[int, int]
            Classifier indicies.
        label : Label
            The label.

        Returns
        -------
        int:
            Number of items pair voted with the same label.

        """
        return sum(
            [
                self.vote_counts[votes]
                for votes in classifiers_labels_votes(
                    pair, (label, label), self.vote_patterns
                )
            ]
        )

    def pair_frequency_moment(
        self, pair: Iterable[int], label: Label
    ) -> sympy.Rational:
        """
        Calculate the label classifier pair frequency moment.

        If (i, j) = pair, then this is -

            f_{label_i, label_j} - f_{label_i} * f_{label_j}

        The fraction of times the classifier pair voted with the same label
        minus the product of their individual label voting frequencies.

        Parameters
        ----------
        pair : Iterable(int, int)
            The pair of classifiers.
        label : Label
            One of the binary labels.

        Returns
        -------
        sympy.Rational:
            The pair frequency moment as a Fraction object
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
    ) -> Mapping[tuple[int, int], sympy.Rational]:
        """All the label pair frequency moments."""
        return {
            pair: self.pair_frequency_moment(pair, label)
            for pair in self.pairs
        }

    def trio_frequency_moment(self) -> sympy.Rational:
        """
        Calculate the 3rd frequency moment of a trio of binary classifiers.

        Don't ask.
        """
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


@dataclass
class ObservedVoteCounts:
    """Data class to validate the test vote counts for an arbitrary number
    of binary classifiers.

    Initialized with a Mapping[Votes, int] of the form:
        {('a', ..., a'): int, ..., ('b', ..., 'b'):int}

    Class used during evaluations with unlabeled data where we do not
    know the true label for each item labeled by the classifiers.
    """

    vote_counts: VoteCounts

    def __post_init__(self):
        """
        Check we have counts for a valid evaluation of binary classifiers.

        0. Determine the size of the ensemble and that all provided
           vote patterns conform to it.
        1. No negative counts.
        2. Initialize all possible vote patterns by the trio.
        3. The empty test - all counts zero - is not allowed.
        """

        # Start by determining the size of the ensemble is the same
        # for all provided vote patterns
        ensemble_sizes = [
            len(vote_pattern) for vote_pattern in self.vote_counts.keys()
        ]
        assert len(set(ensemble_sizes)) == 1

        # Construct the binary vote patterns for an ensemble
        # of this size
        self.ensemble_size = ensemble_sizes[0]
        self.vote_patterns = itertools.product(
            *["ab" for i in range(self.ensemble_size)]
        )

        # Make sure all patterns have a non-negative count.
        object.__setattr__(
            self,
            "vote_counts",
            {
                vote_pattern: self.vote_counts.get(vote_pattern, 0)
                for vote_pattern in self.vote_patterns
            },
        )

        if any([count for count in self.vote_counts.values() if (count < 0)]):
            raise ValueError("No negative vote counts allowed.")

        object.__setattr__(self, "test_size", sum(self.vote_counts.values()))

        # The empty test is not allowed
        if self.test_size == 0:
            raise ValueError("The empty test is not allowed.")

    def to_frequencies_exact(self) -> VoteFrequencies:
        """
        Turn vote integer counts to exact Fraction objects.

        Returns
        -------
        VoteFrequencies:
            Maps a trio vote pattern to its Fraction occurence in the test.
        """
        return {
            vp: sympy.Rational(self.vote_counts[vp], self.test_size)
            for vp in self.vote_counts.keys()
        }

    def to_frequencies_float(self) -> Mapping[Votes, float]:
        """
        Compute observerd voting pattern frequencies, inexactly, as floats.

        Returns
        -------
        Mapping[Votes, float]:
            Maps a trio vote pattern to its percentage occurence in the test
            as an inexact float.

        """
        return {
            vp: self.vote_counts[vp] / self.test_size
            for vp in self.vote_counts.keys()
        }

    def classifier_label_frequency(
        self, classifier: int, label: Label
    ) -> sympy.Rational:
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
        sympy.Rational(label_vote_counts, test_size):
            The fraction of times the classifier voted the label when
            classifying items in the test.

        """
        vote_frequencies = self.to_frequencies_exact()
        return sum(
            [
                vote_frequencies[votes]
                for votes in classifier_label_votes(classifier, label)
            ]
        )

    def classifier_label_responses(self, classifier: int, label: Label) -> int:
        """
        Calculates number of responses with label by classifier.

        Parameters
        ----------
        classifier : int
            DESCRIPTION.
        label : Label
            DESCRIPTION.

        Returns
        -------
        int
            Number of times the classifier decided an item was label.

        """
        return sum(
            [
                self.vote_counts[votes]
                for votes in classifier_label_votes(classifier, label)
            ]
        )

    def pair_label_frequency(
        self, pair: Iterable[int], label: Label
    ) -> sympy.Rational:
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
        sympy.Rational:
            The fraction of times a pair of classifiers voted with the
            same label when classifying items in the test.

        """
        vote_frequencies = self.to_frequencies_exact()
        return sum(
            [
                vote_frequencies[votes]
                for votes in classifiers_labels_votes(pair, (label, label))
            ]
        )

    def pair_label_responses(
        self, pair: Iterable[int], label: Label
    ) -> sympy.Rational:
        """
        Computes number of times a pair voted with the same label.

        Parameters
        ----------
        pair : Iterable[int, int]
            Classifier indicies.
        label : Label
            The label.

        Returns
        -------
        int:
            Number of items pair voted with the same label.

        """
        return sum(
            [
                self.vote_counts[votes]
                for votes in classifiers_labels_votes(pair, (label, label))
            ]
        )

    def pair_frequency_moment(
        self, pair: Iterable[int], label: Label
    ) -> sympy.Rational:
        """
        Calculate the label classifier pair frequency moment.

        If (i, j) = pair, then this is -

            f_{label_i, label_j} - f_{label_i} * f_{label_j}

        The fraction of times the classifier pair voted with the same label
        minus the product of their individual label voting frequencies.

        Parameters
        ----------
        pair : Iterable(int, int)
            The pair of classifiers.
        label : Label
            One of the binary labels.

        Returns
        -------
        sympy.Rational:
            The pair frequency moment as a Fraction object
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
    ) -> Mapping[tuple[int, int], sympy.Rational]:
        """All the label pair frequency moments."""
        return {
            pair: self.pair_frequency_moment(pair, label)
            for pair in trio_pairs
        }

    def trio_frequency_moment(self) -> sympy.Rational:
        """
        Calculate the 3rd frequency moment of a trio of binary classifiers.

        Don't ask.
        """
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

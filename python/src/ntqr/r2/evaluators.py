"""Purely algebraic evaluator of error independent binary classifiers.

Classes:
    ObservedVoteCounts
    ByLabelVoteCounts

Functions:

Misc variables:
    uciadult_label_counts
    """

import math

# The mathematics of evaluating finite samples is, by construction, one
# of estimating integer fractions. We import this module so we can create
# two versions of every computation - the exact one using integer ratios,
# and the one using the default floating point numbers
from fractions import Fraction
from typing_extensions import Union, Literal, Mapping, Iterable

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

uciadult_label_counts: LabelVoteCounts = {
    "a": {
        ("a", "a", "a"): 715,
        ("a", "a", "b"): 161,
        ("a", "b", "a"): 2406,
        ("a", "b", "b"): 455,
        ("b", "a", "a"): 290,
        ("b", "a", "b"): 94,
        ("b", "b", "a"): 1335,
        ("b", "b", "b"): 231,
    },
    "b": {
        ("a", "a", "a"): 271,
        ("a", "a", "b"): 469,
        ("a", "b", "a"): 3395,
        ("a", "b", "b"): 7517,
        ("b", "a", "a"): 272,
        ("b", "a", "b"): 399,
        ("b", "b", "a"): 6377,
        ("b", "b", "b"): 12455,
    },
}


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


# To compute algebraic functions of the evaluation vote counts
def classifier_label_votes(classifier: int, label: Label) -> tuple[Votes, ...]:
    return tuple(
        [votes for votes in trio_vote_patterns if votes[classifier] == label]
    )


def votes_match(
    votes: tuple[Label, ...],
    classifiers: Iterable[int],
    labels: Iterable[Label],
) -> bool:
    """
    Returns True if voting sub-pattern, labels,
    corresponding to the indices in classifiers
    matches the trio pattern in votes.

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
    """
    Provides the trio voting patterns that match the
    voting sub-pattern in labels for the classifiers.

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


class TrioVoteCounts:
    def __init__(self, vote_counts: VoteCounts):
        """
        vote_counts[VoteCounts]: mapping from three possible vote
        patterns for three binary classifiers to their observed
        integer count in the test set.

        Parameters
        ----------
        vote_counts : VoteCounts
            Mapping from three possible vote
            patterns for three binary classifiers to their observed
            integer count in the test set. There must be at least one
            or more counts.

        Returns
        -------
        None.

        """
        self.vote_counts = {
            observed_votes: vote_counts.get(observed_votes, 0)
            for observed_votes in trio_vote_patterns
        }

        self.test_size = sum(self.vote_counts.values())
        # TODO: throw exception if test_size == 0

    def to_frequencies_exact(self) -> VoteFrequencies:
        """
        Computes observerd voting pattern frequencies from
        observed voting counts on the evaluation.

        Returns
        -------
        VoteFrequencies: mapping of trio votes to percentage
        of occurence in the test as Fraction.

        """
        return {
            vp: Fraction(self.vote_counts[vp], self.test_size)
            for vp in self.vote_counts.keys()
        }

    def to_frequencies_float(self) -> Mapping[Votes, float]:
        """
        Computes observerd voting pattern frequencies from
        observed voting counts on the evaluation. This is
        not exact and returns float values.

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
         Calculates classifier label voting frequency.

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
        Computes frequency of times a pair voted with the same
        label.

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
        Calculates the pair frequency moment for a label given observed
        trio vote counts. Consult the Mathematica notebooks for details.
        f_label_i_label_j - f_label_i * f_label_j

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


class TrioLabelVoteCounts:
    def __init__(self, label_vote_counts: LabelVoteCounts):
        self.lbl_vote_counts = label_vote_counts

        self.test_sizes = {
            label: sum(self.lbl_vote_counts[label].values())
            for label in ("a", "b")
        }

    def to_vote_counts(self) -> VoteCounts:
        """
        Projects by-true-label voting pattern counts to by-voting-pattern
        counts.
        """
        return {
            voting_pattern: (
                self.lbl_vote_counts["a"].get(voting_pattern, 0)
                + self.lbl_vote_counts["b"].get(voting_pattern, 0)
            )
            for voting_pattern in trio_vote_patterns
        }

    def to_TrioVoteCounts(self):
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


class SupervisedEvaluation:
    """
    In experimental settings where labeled data is available,
    we can compute exactly the performance of the classifiers
    on the test data.
    """

    def __init__(self, label_counts: TrioLabelVoteCounts):
        self.label_counts = label_counts

        self.evaluation = {
            "prevalence": self.prevalences(),
            "accuracies": [
                {
                    label: self.classifier_label_accuracy(classifier, label)
                    for label in ("a", "b")
                }
                for classifier in range(3)
            ],
            "pair_correlations": {
                label: {
                    pair: self.pair_label_error_correlation(pair, label)
                    for pair in trio_pairs
                }
                for label in ("a", "b")
            },
        }

    def prevalences(self):
        test_sizes = self.label_counts.test_sizes
        total = sum(test_sizes.values())
        return {
            "a": Fraction(test_sizes["a"], total),
            "b": Fraction(test_sizes["b"], total),
        }

    def classifier_label_accuracy(self, classifier: int, label: Label):
        """
        Computes classifier label accuracy.
        """
        test_size = self.label_counts.test_sizes[label]
        classifier_votes = classifier_label_votes(classifier, label)
        label_counts = self.label_counts[label]
        correct_counts = [label_counts[votes] for votes in classifier_votes]
        return Fraction(sum(correct_counts), test_size)

    def other_label(self, label: Label):
        if label == "a":
            o_label = "b"
        else:
            o_label = "a"
        return o_label

    def pair_label_error_correlation(self, pair, label):
        """Calculates the label error correlation for a
        pair of binary classifiers.
        """
        test_size = self.label_counts.test_sizes[label]
        classifier_accuracies = [
            self.classifier_label_accuracy(classifier, label)
            for classifier in pair
        ]
        label_counts = self.label_counts[label]
        o_label = self.other_label(label)
        return (
            # They are both correct
            (1 - classifier_accuracies[0])
            * (1 - classifier_accuracies[1])
            * sum(
                [
                    label_counts[votes]
                    for votes in classifiers_labels_votes(pair, (label, label))
                ]
            )
            +
            # C_i is correct, C_j is incorrect
            (1 - classifier_accuracies[0])
            * (0 - classifier_accuracies[1])
            * sum(
                [
                    label_counts[votes]
                    for votes in classifiers_labels_votes(
                        pair, (label, o_label)
                    )
                ]
            )
            +
            # C_i is incorrect, C_j is correct
            (0 - classifier_accuracies[0])
            * (1 - classifier_accuracies[1])
            * sum(
                [
                    label_counts[votes]
                    for votes in classifiers_labels_votes(
                        pair, (o_label, label)
                    )
                ]
            )
            +
            # C_i and C_j are incorrect
            (0 - classifier_accuracies[0])
            * (0 - classifier_accuracies[1])
            * sum(
                [
                    label_counts[votes]
                    for votes in classifiers_labels_votes(
                        pair, (o_label, label)
                    )
                ]
            )
        ) / test_size

    # TODO: implement the three way correlations


class ErrorIndependentEvaluator:
    """
    Evaluates three binary classifiers using the data sketch of
    their aligned decisions. Assumes they were error independent
    on the test.
    """

    def __init__(self, vote_counts: TrioVoteCounts):
        self.vote_counts = vote_counts
        self.vote_frequencies = self.vote_counts.to_frequencies_exact()

    def prevalence_quadratic_terms(self):
        """Calculates the "a" coefficient associated with the evaluation
        of the sample prevalence for the alpha label."""

        # The coefficient of the square term
        pfmds = self.vote_counts.label_pairs_frequency_moments("b")
        vote_frequencies = self.vote_frequencies
        fbbb = vote_frequencies[("b", "b", "b")]
        diff1 = fbbb - self.vote_counts.trio_frequency_moment()
        prodFDs = math.prod(pfmds.values())
        term_coefficients = {
            2: diff1**2 + 4 * prodFDs,
            1: -(diff1**2 + 4 * prodFDs),
            0: -prodFDs,
        }

        return term_coefficients

    def alpha_prevalence_estimate(self):
        """Calculates the prevalence of the alpha label."""
        coeffs = self.prevalence_quadratic_terms()
        b = coeffs[1]
        c = coeffs[0]
        sqrTerm = math.sqrt(1 - 4 * c / b) / 2
        return [Fraction(1, 2) + sqrTerm, Fraction(1, 2) - sqrTerm]


if __name__ == "__main__":
    from pprint import pprint

    print(
        """The evaluation observed voting patterns by true label -
        the ground truth."""
    )
    pprint(uciadult_label_counts)

    print(
        """During unlabeled evaluation we only get to observe the voting
        patterns, without knowing the true label of each voting instance."""
    )
    data_sketch = TrioLabelVoteCounts(
        uciadult_label_counts
    ).to_TrioVoteCounts()

    print(
        """To carry out the evaluation we need the relative frequency of the
        voting patterns."""
    )
    observed_frequencies = data_sketch.to_frequencies_exact()
    pprint(observed_frequencies)

    print(
        """To estimate the prevalence of the alpha label, we need the
    coefficients of the prevalence quadratic:"""
    )
    error_ind_evaluator = ErrorIndependentEvaluator(data_sketch)
    prev_terms = error_ind_evaluator.prevalence_quadratic_terms()

    print("1. The 'a' coefficient for the quadratic:")
    pprint(prev_terms[2])
    pprint("2. The 'b' coefficient for the quadratic:")
    pprint(prev_terms[1])
    pprint("3. The 'c' coefficient for the quadratic:")
    pprint(prev_terms[0])

    pprint("Algebraic estimates of alpha label prevalence: ")
    pprint(error_ind_evaluator.alpha_prevalence_estimate())

    gt_evaluation = SupervisedEvaluation(
        TrioLabelVoteCounts(uciadult_label_counts)
    ).evaluation

    trueAPrevalence = gt_evaluation["prevalence"]["a"]
    print("The true alpha label prevalence is: ")
    pprint(trueAPrevalence)
    print(" or: ")
    pprint(float(trueAPrevalence))

    # The test run picked for this code comes from a trio of classifiers
    # that have, in fact, very small pair error correlations.
    pprint("Ground truth values: ")
    pprint(gt_evaluation, sort_dicts=False)

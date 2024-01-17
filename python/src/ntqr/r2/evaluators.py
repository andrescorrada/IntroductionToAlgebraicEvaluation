"""
Evaluators for binary classification tests (R=2).

Classes:
    SupervisedEvaluation
    ErrorIndependentEvaluation
    MajorityVotingEvaluation

Functions:

Misc variables:

"""

import math
from fractions import Fraction
from typing_extensions import Iterable

from ntqr.r2.datasketches import Label, Votes
from ntqr.r2.datasketches import trio_vote_patterns, trio_pairs
from ntqr.r2.datasketches import TrioLabelVoteCounts, TrioVoteCounts


def classifier_label_votes(classifier: int, label: Label) -> tuple[Votes, ...]:
    """
    Return trio vote patterns where classifier voted label.

    Parameters
    ----------
    classifier : int
        One of (0, 1, 2).
    label : Label
        One of ('a', 'b').

    Returns
    -------
    Iterable[Votes]
        All the trio vote patterns where classifier voted label.

    """
    return tuple(
        [votes for votes in trio_vote_patterns if votes[classifier] == label]
    )


def votes_match(
    votes: tuple[Label, ...],
    classifiers: Iterable[int],
    labels: Iterable[Label],
) -> bool:
    """
    Return True if votes matches the labels for the classifiers.

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
    Return all trio vote patterns that match labels by classifiers.

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


class SupervisedEvaluation:
    """Evaluation for experiments where the true labels are known."""

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
        """
        Calculate the prevalences of the two labels.

        Returns
        -------
        Mapping[Label, Fraction]
            Mapping from labels to percentage of appearance in the test.

        """
        test_sizes = self.label_counts.test_sizes
        total = sum(test_sizes.values())
        return {
            "a": Fraction(test_sizes["a"], total),
            "b": Fraction(test_sizes["b"], total),
        }

    def classifier_label_accuracy(self, classifier: int, label: Label):
        """Compute classifier label accuracy."""
        test_size = self.label_counts.test_sizes[label]
        classifier_votes = classifier_label_votes(classifier, label)
        label_counts = self.label_counts[label]
        correct_counts = [label_counts[votes] for votes in classifier_votes]
        return Fraction(sum(correct_counts), test_size)

    def other_label(self, label: Label):
        """Return the other binary classification label given label."""
        if label == "a":
            o_label = "b"
        else:
            o_label = "a"
        return o_label

    def pair_label_error_correlation(self, pair, label):
        """Calculate the label error correlation a classifier pair."""
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


class ErrorIndependentEvaluation:
    """
    Evaluates three binary classifiers assuming they are error independent.

    WARNING: 'error independence' is defined here as an empirical quantity
    that is only defined for any given test. Classifiers error independent
    in one test, may not be so in another.
    """

    def __init__(self, vote_counts: TrioVoteCounts):
        self.vote_counts = vote_counts
        self.vote_frequencies = self.vote_counts.to_frequencies_exact()

    def alpha_prevalence_quadratic_terms(self):
        """
        Calculate the coefficients of the 'a' label prevalence quadratic.

        If the quadratic is represented as:
            a * (P_a)**2 + b * P_a + c
        then,
            a = terms[2], b = terms[1], c = terms[0].
        The quadratic is written in the 'standard' way seen in algebra
        textbooks. Be careful to not mistake the 'a' or 'b' coefficients
        described here with the two labels being used for classification -
        currently implemented as ('a', 'b').
        """
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
        """Calculate the prevalence of the alpha label."""
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

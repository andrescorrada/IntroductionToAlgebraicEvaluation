"""
Evaluators for binary classification tests (R=2).

Classes:
    SupervisedEvaluation
    ErrorIndependentEvaluation
    MajorityVotingEvaluation

Functions:

Misc variables:

"""

import math, itertools
import sympy
from fractions import Fraction
from typing_extensions import Iterable

from ntqr.r2.datasketches import Label, Votes
from ntqr.r2.datasketches import (
    classifier_label_votes,
    classifiers_labels_votes,
)
from ntqr.r2.datasketches import TrioLabelVoteCounts, TrioVoteCounts
from ntqr.r2.examples import uciadult_label_counts


class SupervisedEvaluation:
    """Evaluation for experiments where the true labels are known."""

    # While the classes are transitioned to ensembles of arbitrary size,
    # this class is hard-wired for three classifiers
    vote_patterns = list(itertools.product(*["ab" for i in range(3)]))
    pairs = ((0, 1), (0, 2), (1, 2))

    def __init__(self, label_counts: TrioLabelVoteCounts):
        self.label_counts = label_counts

        self.evaluation_exact = {
            "prevalence": self.prevalences(),
            "accuracy": [
                {
                    label: self.classifier_label_accuracy(classifier, label)
                    for label in ("a", "b")
                }
                for classifier in range(3)
            ],
            "pair_correlation": {
                pair: {
                    label: self.pair_label_error_correlation(pair, label)
                    for label in ("a", "b")
                }
                for pair in self.pairs
            },
            "3_way_correlation": {
                trio: {
                    label: self.three_way_label_error_correlation(trio, label)
                    for label in ("a", "b")
                }
                for trio in ((0, 1, 2),)
            },
        }

        self.evaluation_float = {
            "prevalence": {
                label: float(val)
                for label, val in self.evaluation_exact["prevalence"].items()
            },
            "accuracy": [
                {label: float(val) for label, val in cdict.items()}
                for cdict in self.evaluation_exact["accuracy"]
            ],
            "pair_correlation": {
                pair: {label: float(val) for label, val in corrs.items()}
                for pair, corrs in self.evaluation_exact[
                    "pair_correlation"
                ].items()
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
            "a": sympy.Rational(test_sizes["a"], total),
            "b": sympy.Rational(test_sizes["b"], total),
        }

    def classifier_label_accuracy(self, classifier: int, label: Label):
        """Compute classifier label accuracy."""
        test_size = self.label_counts.test_sizes[label]
        classifier_votes = classifier_label_votes(
            classifier, label, self.vote_patterns
        )
        label_counts = self.label_counts[label]
        correct_counts = [label_counts[votes] for votes in classifier_votes]
        return sympy.Rational(sum(correct_counts), test_size)

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
                    for votes in classifiers_labels_votes(
                        pair, (label, label), self.vote_patterns
                    )
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
                        pair, (label, o_label), self.vote_patterns
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
                        pair, (o_label, label), self.vote_patterns
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
                        pair, (o_label, o_label), self.vote_patterns
                    )
                ]
            )
        ) / test_size

    def three_way_label_error_correlation(self, triplet, label):
        """Calculate the label error correlation a classifier pair."""
        test_size = self.label_counts.test_sizes[label]
        classifier_accuracies = [
            self.classifier_label_accuracy(classifier, label)
            for classifier in triplet
        ]
        label_counts = self.label_counts[label]
        o_label = self.other_label(label)
        return (
            # They are all correct
            (1 - classifier_accuracies[0])
            * (1 - classifier_accuracies[1])
            * (1 - classifier_accuracies[2])
            * sum(
                [
                    label_counts[votes]
                    for votes in classifiers_labels_votes(
                        triplet, (label, label, label), self.vote_patterns
                    )
                ]
            )
            +
            # C_i is correct, C_j is correct, C_k is incorrect
            (1 - classifier_accuracies[0])
            * (1 - classifier_accuracies[1])
            * (0 - classifier_accuracies[2])
            * sum(
                [
                    label_counts[votes]
                    for votes in classifiers_labels_votes(
                        triplet, (label, label, o_label), self.vote_patterns
                    )
                ]
            )
            +
            # C_i is correct, C_j is incorrect, C_k is correct
            (1 - classifier_accuracies[0])
            * (0 - classifier_accuracies[1])
            * (1 - classifier_accuracies[2])
            * sum(
                [
                    label_counts[votes]
                    for votes in classifiers_labels_votes(
                        triplet, (label, o_label, label), self.vote_patterns
                    )
                ]
            )
            +
            # C_i is incorrect, C_j is correct, C_k is correct
            (0 - classifier_accuracies[0])
            * (1 - classifier_accuracies[1])
            * (1 - classifier_accuracies[2])
            * sum(
                [
                    label_counts[votes]
                    for votes in classifiers_labels_votes(
                        triplet, (o_label, label, label), self.vote_patterns
                    )
                ]
            )
            +
            # C_i is correct, C_j is incorrect, C_k is incorrect
            (1 - classifier_accuracies[0])
            * (0 - classifier_accuracies[1])
            * (0 - classifier_accuracies[2])
            * sum(
                [
                    label_counts[votes]
                    for votes in classifiers_labels_votes(
                        triplet, (label, o_label, o_label), self.vote_patterns
                    )
                ]
            )
            +
            # C_i is incorrect, C_j is correct, C_k is incorrect
            (0 - classifier_accuracies[0])
            * (1 - classifier_accuracies[1])
            * (0 - classifier_accuracies[2])
            * sum(
                [
                    label_counts[votes]
                    for votes in classifiers_labels_votes(
                        triplet, (o_label, label, o_label), self.vote_patterns
                    )
                ]
            )
            +
            # C_i is incorrect, C_j is incorrect, C_k is correct
            (0 - classifier_accuracies[0])
            * (0 - classifier_accuracies[1])
            * (1 - classifier_accuracies[2])
            * sum(
                [
                    label_counts[votes]
                    for votes in classifiers_labels_votes(
                        triplet, (o_label, o_label, label), self.vote_patterns
                    )
                ]
            )
            +
            # C_i is incorrect, C_j is incorrect, C_k is incorrect
            (0 - classifier_accuracies[0])
            * (0 - classifier_accuracies[1])
            * (0 - classifier_accuracies[2])
            * sum(
                [
                    label_counts[votes]
                    for votes in classifiers_labels_votes(
                        triplet,
                        (o_label, o_label, o_label),
                        self.vote_patterns,
                    )
                ]
            )
        ) / test_size


class ErrorIndependentEvaluation:
    """
    Evaluate three binary classifiers assuming they are error independent.

    Returns
    -------
    Absent labeled data, there are two logically consistent solutions
    given only their decision voting frequencies. For binary classification,
    this means that there are 2 possible points in evaluation space that
    can possibly explain the test results. The ground truth evaluation is
    one of these points -- if the assumption of error independence is true.

    The exact algebraic results have a unique virtue that few alarm systems
    have - it can warn about the failures of its own assumption of error
    independence. If the two possible solutions for the 'a' label prevalence
    return an unresolved integer square root - the classifiers are error
    correlated in the evaluation.

    In version 0.2 the math needed to take handle the almost certain detection
    of error correlation will be added. It is already being built as
    can be seen in ntqr.r2.postulates file where the postulates related to
    computing the error correlation have been expressed using the SymPy
    package.

    Warnings:
    ---------
    A. The ntqr package uses a notion of 'error independence' that is
    different than the one most familiar in the ML/AI community. There are
    many notions of independence in mathematics. In the context of ML/AI
    papers/discussions, the term 'error independence' is taken to be:
    A.1. Functional independence of distributions: P(x, y) = P(x)P(y)
    The one used in the ntqr package is sample defined since there is no
    probability theory used in its logic. For that reason, you must define a
    set of error correlation parameters. 'Error independence' in the ntqr
    package means:
    A.2. pair_label_correlations = 0, trio_label_correlations = 0, ...
    It is best to think of 'error independence' in the ntqr package as a
    property that belongs to the classifiers AND the test they took.

    B. This class currently assumes that the observed classifier
    vote counts supplied by the user are not fake. The set of all valid
    observations from a classification test is much smaller than the set
    of all sets of eight positive integers. Future versions of the ntqr
    package will implement the algebraic geometry needed to detect when
    TrioVoteCounts objects are not explainable as observations from a
    classification test.

    The error independent solution can fail if, in fact, the classifiers
    are highly correlated on the test being evaluated. Tests can fail.
    Future versions will have implemented the exceptions.
        1. PrevalenceImaginaryException
        2. NoSolutionException

    The PrevalenceImaginaryException is a iron-clad detection of highly
    correlated classifiers. Its main utility will be in "warning light"
    applications in AI safety.

    The NoSolutionException means that NO independent system can possibly
    explain the observations. There are two different reasons for this -
    higher error correlations, or the data sketch is fake. Distinguishing
    between the two comes down to the same computation of error correlation.
    """

    def __init__(self, vote_counts: TrioVoteCounts):
        self.vote_counts = vote_counts
        self.vote_frequencies = self.vote_counts.to_frequencies_exact()

        # Two solutions are possible
        self.evaluation_exact = [
            {
                "prevalence": {"a": a_prevalence, "b": 1 - a_prevalence},
                "accuracy": [
                    {
                        "a": self.classifier_a_label_accuracy(
                            classifier, a_prevalence
                        ),
                        "b": self.classifier_b_label_accuracy(
                            classifier, a_prevalence
                        ),
                    }
                    for classifier in range(3)
                ],
            }
            for a_prevalence in self.alpha_prevalence_estimates()
        ]

        self.evaluation_float = [
            {
                "prevalence": {
                    label: float(val)
                    for label, val in sol_dict["prevalence"].items()
                },
                "accuracy": [
                    {
                        label: float(val)
                        for label, val in classifier_dict.items()
                    }
                    for classifier_dict in sol_dict["accuracy"]
                ],
            }
            for sol_dict in self.evaluation_exact
        ]

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

        pfmds = self.vote_counts.label_pairs_frequency_moments("b")
        vote_frequencies = self.vote_frequencies
        fbbb = vote_frequencies[("b", "b", "b")]
        diff1 = fbbb - self.vote_counts.trio_frequency_moment()
        prodFDs = math.prod(pfmds.values())
        term_coefficients = {
            2: diff1**2 + 4 * prodFDs,
            1: -(diff1**2 + 4 * prodFDs),
            0: prodFDs,
        }

        return term_coefficients

    def alpha_prevalence_estimates(self):
        """Calculate the prevalence of the alpha label.

        Since the quadratic equation has ordered solutions by
        the plus/minus operations, we arbitrarily return the 'a' label
        less than 50% solution first."""
        coeffs = self.alpha_prevalence_quadratic_terms()
        a = coeffs[2]
        c = coeffs[0]
        sqrTerm = sympy.sqrt(1 - 4 * c / a) / 2
        prev_expr = [
            sympy.Rational(1, 2) - sqrTerm,
            sympy.Rational(1, 2) + sqrTerm,
        ]
        return [sympy.simplify(expr) for expr in prev_expr]

    def classifier_a_label_accuracy(self, classifier: int, a_prevalence):
        """
        Calculate classifier 'a' label accuracies.

        Parameters
        ----------
        classifier : int
            One of (0, 1, 2).
        a_prevalence: Sympy expression

        Returns
        -------
        The a label accuracy given the a_prevalence value
        """
        # Assuming the linear equation form a + b * P_a + c * P_{i, a}

        classifier_freqs = [
            self.vote_counts.classifier_label_frequency(classifier, "b")
            for classifier in range(3)
        ]
        fbbb = self.vote_frequencies[("b", "b", "b")]
        fijk = self.vote_counts.trio_frequency_moment()
        freqDiff = fbbb - fijk

        pair_moments = self.vote_counts.label_pairs_frequency_moments("b")
        prodFDs = math.prod(pair_moments.values())
        match classifier:
            case 0:
                other_moment = pair_moments[(1, 2)]
            case 1:
                other_moment = pair_moments[(0, 2)]
            case 2:
                other_moment = pair_moments[(0, 1)]
            case _:
                pass
        a_coeff = (
            freqDiff * freqDiff
            - other_moment * (1 - classifier_freqs[classifier]) * freqDiff
            + 2 * prodFDs
        )

        b_coeff = -4 * prodFDs - freqDiff**2
        c_coeff = other_moment * freqDiff

        return sympy.simplify((-a_coeff - b_coeff * a_prevalence) / c_coeff)

    def classifier_b_label_accuracy(self, classifier: int, a_prevalence):
        """
        Calculate classifier 'b' label accuracies.

        Parameters
        ----------
        classifier : int
            One of (0, 1, 2).

        Returns
        -------
        Two possible logically consistent estimates for P_{i,b} given the
        test error independence assumption.
        """
        # Assuming the linear equation form a + b * P_a + c * P_{i, b}

        classifier_freqs = [
            self.vote_counts.classifier_label_frequency(classifier, "b")
            for classifier in range(3)
        ]
        fbbb = self.vote_frequencies[("b", "b", "b")]
        fijk = self.vote_counts.trio_frequency_moment()
        freqDiff = fbbb - fijk

        pair_moments = self.vote_counts.label_pairs_frequency_moments("b")
        prodFDs = math.prod(pair_moments.values())
        match classifier:
            case 0:
                other_moment = pair_moments[(1, 2)]
            case 1:
                other_moment = pair_moments[(0, 2)]
            case 2:
                other_moment = pair_moments[(0, 1)]
            case _:
                pass
        a_coeff = (
            classifier_freqs[classifier] * other_moment * freqDiff
            - 2 * prodFDs
        )

        b_coeff = 4 * prodFDs + freqDiff**2
        c_coeff = -other_moment * freqDiff

        return sympy.simplify((-a_coeff - b_coeff * a_prevalence) / c_coeff)


class MajorityVotingEvaluation:
    """
    Evaluate three binary classifiers using majority voting.

    Majority voting can be used to carry out evaluation algebraically.
    Typically, majority voting is used with the assumption that the
    crowd is always right. In the context of safety, however, that the
    crowd is always wrong is an equally valid a-priori assumption. Hence,
    this class returns TWO evaluations. The first assuming the crowd is
    always right and the second assuming they are always wrong.
    Its main virtue is that it is simple and rock solid - always returns
    logically consistent evaluations.
    """

    vote_patterns = list(itertools.product(*["ab" for i in range(3)]))

    def __init__(self, vote_counts: TrioVoteCounts):
        """
        Initialize data structures.

        TODO: Implement detection that the TrioVoteCounts are not generated
        by a classification test.

        Parameters
        ----------
        vote_counts : TrioVoteCounts
            Label decision counts for the aligned trio of classifiers.

        Returns
        -------
        None.

        """
        self.vote_counts = vote_counts
        self.vote_frequencies = self.vote_counts.to_frequencies_exact()
        self.labels = ("a", "b")

        self.majority_right_vote_patterns = {
            label: [
                votes
                for votes in self.vote_patterns
                if votes.count(label) >= 2
            ]
            for label in self.labels
        }

        self.majority_wrong_vote_patterns = {
            label: [
                votes
                for votes in self.vote_patterns
                if votes.count(label) <= 1
            ]
            for label in self.labels
        }

        self.evaluation_exact = [
            self.compute_vote_pattern_evaluation(vps, flip)
            for vps, flip in [
                (self.majority_right_vote_patterns, False),
                (self.majority_wrong_vote_patterns, False),
            ]
        ]

        self.evaluation_float = [
            self.to_float(sol) for sol in self.evaluation_exact
        ]

    def compute_vote_pattern_evaluation(self, vote_patterns, flip):
        return {
            "prevalence": self.prevalences(vote_patterns),
            "accuracy": [
                {
                    label: self.classifier_label_accuracy(
                        classifier, vote_patterns, label, flip
                    )
                    for label in self.labels
                }
                for classifier in range(3)
            ],
        }

    def prevalences(self, vote_patterns):
        """Compute label prevalences in the test."""
        return {
            label: sum(
                [self.vote_frequencies[vp] for vp in vote_patterns[label]]
            )
            for label in self.labels
        }

    def classifier_label_accuracy(
        self, classifier, vote_patterns, label, flip
    ):
        """Compute the label accuracy for classifier."""
        if flip:
            if label == "a":
                correct_label = "b"
            elif label == "b":
                correct_label = "a"
        else:
            correct_label = label
        return sum(
            [
                self.vote_frequencies[vp]
                for vp in vote_patterns[correct_label]
                if vp[classifier] == correct_label
            ]
        ) / sum(
            [self.vote_frequencies[vp] for vp in vote_patterns[correct_label]]
        )

    def to_float(self, sol):
        as_floats = {}
        as_floats["prevalence"] = {
            label: float(val) for label, val in sol["prevalence"].items()
        }
        as_floats["accuracy"] = [
            ({label: float(val) for label, val in classifier_dict.items()})
            for classifier_dict in sol["accuracy"]
        ]
        return as_floats


if __name__ == "__main__":
    from pprint import pprint

    sympy.init_printing(use_unicode=True)

    print(
        """The evaluation observed voting patterns by true label -
        the ground truth."""
    )
    pprint(uciadult_label_counts)

    print()

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

    print()

    print(
        """To estimate the prevalence of the alpha label, we need the
    coefficients of the prevalence quadratic:"""
    )
    error_ind_evaluator = ErrorIndependentEvaluation(data_sketch)
    prev_terms = error_ind_evaluator.alpha_prevalence_quadratic_terms()

    print("1. The 'a' coefficient for the quadratic:")
    pprint(prev_terms[2])
    pprint("2. The 'b' coefficient for the quadratic:")
    pprint(prev_terms[1])
    pprint("3. The 'c' coefficient for the quadratic:")
    pprint(prev_terms[0])

    print("Algebraic evaluation in its exact form: ")
    pprint(error_ind_evaluator.evaluation_exact, sort_dicts=False)
    print()
    print("Algebraic evaluation with inexact floats: ")
    pprint(error_ind_evaluator.evaluation_float, sort_dicts=False)

    gt = SupervisedEvaluation(TrioLabelVoteCounts(uciadult_label_counts))
    print()

    trueAPrevalence = gt.evaluation_exact["prevalence"]["a"]
    print("The true alpha label prevalence is: ")
    pprint(trueAPrevalence)
    print(" or: ")
    pprint(float(trueAPrevalence))

    print()

    # The test run picked for this code comes from a trio of classifiers
    # that have, in fact, very small pair error correlations.
    print("Ground truth values:")
    pprint(gt.evaluation_exact, sort_dicts=False)
    print("Ground truth as inexact floats:")
    pprint(gt.evaluation_float, sort_dicts=False)

    print("The majority voting evaluator:")
    mv_eval = MajorityVotingEvaluation(data_sketch)
    pprint(mv_eval.evaluation_exact, sort_dicts=False)
    print("As floats:")
    pprint(mv_eval.evaluation_float, sort_dicts=False)

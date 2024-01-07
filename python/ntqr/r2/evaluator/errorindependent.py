#! /usr/bin/env Python
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
from typing_extensions import Union, Literal, Mapping

# Types
Label = Union[Literal["a"], Literal["b"]]


# Vote counts are what we see when we have unlabeled data.
VoteCounts = Mapping[tuple[Label, ...], int]

# Label vote counts, vote counts by true label, are only available
# when we are carrying out experiments on labeled data.
LabelVoteCounts = Mapping[Label, VoteCounts]

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


def to_vote_counts(by_label_counts: LabelVoteCounts) -> VoteCounts:
    """Projects by-true-label voting pattern counts to by-voting-pattern
    counts."""
    return {
        voting_pattern: (
            by_label_counts["a"].get(voting_pattern, 0)
            + by_label_counts["b"].get(voting_pattern, 0)
        )
        for voting_pattern in trio_vote_patterns
    }


def to_voting_frequency_fractions(by_label_counts: LabelVoteCounts) -> \
        Mapping[tuple[Label, ...], Fraction]:
    """Computes observed voting pattern frequencies."""
    by_voting_counts = to_vote_counts(by_label_counts)
    size_of_test = sum(by_voting_counts.values())
    return {
        vp: Fraction(by_voting_counts[vp], size_of_test)
        for vp in by_voting_counts.keys()
    }


def pattern_counts_to_frequencies_exact(by_voting_counts):
    """Computes observerd voting pattern frequencies from
    observed voting pattern counts."""
    size_of_test = sum(by_voting_counts.values())
    return {
        vp: Fraction(by_voting_counts[vp], size_of_test)
        for vp in by_voting_counts.keys()
    }


def project_to_voting_frequencies_fp(by_label_counts):
    """Same as the exact computation, but using floating point
    numbers."""
    by_voting_counts = to_vote_counts(by_label_counts)
    size_of_test = sum(by_voting_counts.values())
    return {
        vp: by_voting_counts[vp] / size_of_test
        for vp in by_voting_counts.keys()
    }


def project_to_voting_frequencies_fp2(by_voting_counts):
    """Same as above, but we start from the projected by-pattern counts."""
    size_of_test = sum(by_voting_counts.values())
    return {
        vp: by_voting_counts[vp] / size_of_test
        for vp in by_voting_counts.keys()
    }


# The known unknowns : All the sample statistics that are needed to
# write exact polynomials of the observed voting frequencies.
#
# Like data streaming algorithms, the desired "stream statistics" are
# coupled to the "data sketch". Depending on what observable statistics
# of the evaluation you are interested in, you would need to formulate
# its exact polynomial formulation. This is the "evaluation ideal"
# definition stage.
#
# In the case considered here where we are collecting "point statistics"
# of the classifiers decisions - the frequencies of the voting patterns
# given items in the test - all the unknown sample statistics needed to have
# an exact polynomial representation of the voting pattern frequencies are:
# 1. The prevalence of the labels (the "environmental" statistics).
# 2. The by-label accuracy of each of the three classifiers.(6)
# 3. The sample defined pair error correlation by pair and label. (6)
# 4. The trio error correlation by label. (2)
#
# Exact representations in Evaluation Land are possible.
# These sample statistics have been completely enumerated above. This
# enumeration is complete and universal. Any test, for any trio of classifiers,
# can be expressed exactly by these unknown sample statistics.
# There are no unknown unknowns in evaluation. We know exactly what we are
# missing - the values of these sample statistics.
# No such universal algorithms or representations are available in Training
# Land. No universal model of the world exists. There are two sides to this
# realization. First, it really is a fundamentally trivial statement. Sample
# statistics are easy to state and enumerate. Second, why have not conquered
# this finite space that we can completely describe?

# To make the logic of the calculations clearer, let's enumerate the voting
# patterns where each classifier votes a given label.

# The patterns for classifier 1
c1_a_votes = (
    ("a", "a", "a"),
    ("a", "a", "b"),
    ("a", "b", "a"),
    ("a", "b", "b"),
)
c1_b_votes = (
    ("b", "a", "a"),
    ("b", "a", "b"),
    ("b", "b", "a"),
    ("b", "b", "b"),
)
# The patterns for classifier 2
c2_a_votes = (
    ("a", "a", "a"),
    ("a", "a", "b"),
    ("b", "a", "a"),
    ("b", "a", "b"),
)
c2_b_votes = (
    ("a", "b", "a"),
    ("a", "b", "b"),
    ("b", "b", "a"),
    ("b", "b", "b"),
)
# The patterns for classifier 3
c3_a_votes = (
    ("a", "a", "a"),
    ("a", "b", "a"),
    ("b", "a", "a"),
    ("b", "b", "a"),
)
c3_b_votes = (
    ("a", "a", "b"),
    ("a", "b", "b"),
    ("b", "a", "b"),
    ("b", "b", "b"),
)


def classifiers_label_accuracies(by_label_counts):
    """Given the by-true label voting pattern counts, calculates the observed
    by-label accuracies of a trio of classifiers."""
    a_test_size = sum(by_label_counts["a"].values())
    b_test_size = sum(by_label_counts["b"].values())
    return {
        1: {
            "a": Fraction(
                sum({by_label_counts["a"][vp] for vp in c1_a_votes}),
                a_test_size,
            ),
            "b": Fraction(
                sum({by_label_counts["b"][vp] for vp in c1_b_votes}),
                b_test_size,
            ),
        },
        2: {
            "a": Fraction(
                sum({by_label_counts["a"][vp] for vp in c2_a_votes}),
                a_test_size,
            ),
            "b": Fraction(
                sum({by_label_counts["b"][vp] for vp in c2_b_votes}),
                b_test_size,
            ),
        },
        3: {
            "a": Fraction(
                sum({by_label_counts["a"][vp] for vp in c3_a_votes}),
                a_test_size,
            ),
            "b": Fraction(
                sum({by_label_counts["b"][vp] for vp in c3_b_votes}),
                b_test_size,
            ),
        },
    }


# We now encounter our 1st error correlation -
# the pair sample error correlation.
# Since algebraic evaluation does not use probability theory, one
# needs to define a notion of independence that is sample based. This
# turns out to be easy - just use the typical sample statistics that
# we are all familiar with. So we define an error correlation based
# on whether you were right or wrong in a particular label decision.
# In essence, we are taking moments of correct decisions indicator
# functions. So the pair error correlation is the sum of the product
# (ci_indicator_value - ci_average_label_accuracy)*
# (cj_indicator_value - cj_average_label_accuracy)
# This is very much like the sample correlation statistics we are all
# familiar with.
#
# To help us carry out the calculation, we enumerate the possible
# ways a pair can vote in terms of the trio votes.
pair_vote_patterns = {
    (1, 2): {
        ("a", "a"): (("a", "a", "a"), ("a", "a", "b")),
        ("a", "b"): (("a", "b", "a"), ("a", "b", "b")),
        ("b", "a"): (("b", "a", "a"), ("b", "a", "b")),
        ("b", "b"): (("b", "b", "a"), ("b", "b", "b")),
    },
    (1, 3): {
        ("a", "a"): (("a", "a", "a"), ("a", "b", "a")),
        ("a", "b"): (("a", "a", "b"), ("a", "b", "b")),
        ("b", "a"): (("b", "a", "a"), ("b", "b", "a")),
        ("b", "b"): (("b", "a", "b"), ("b", "b", "b")),
    },
    (2, 3): {
        ("a", "a"): (("a", "a", "a"), ("b", "a", "a")),
        ("a", "b"): (("a", "a", "b"), ("a", "a", "b")),
        ("b", "a"): (("a", "b", "a"), ("b", "b", "a")),
        ("b", "b"): (("a", "b", "b"), ("b", "b", "b")),
    },
}


def pair_error_correlations(by_label_counts, pair):
    """Calculates the by-label pair error correlation for two
    binary classifiers"""
    a_test_size = sum(by_label_counts["a"].values())
    b_test_size = sum(by_label_counts["b"].values())

    (ci, cj) = pair
    ca = classifiers_label_accuracies(by_label_counts)
    ci_accuracies = ca[ci]
    cj_accuracies = ca[cj]

    return {
        "a": (
            # They are both correct
            (1 - ci_accuracies["a"])
            * (1 - cj_accuracies["a"])
            * sum(
                {
                    by_label_counts["a"][vp]
                    for vp in pair_vote_patterns[pair][("a", "a")]
                }
            )
            +
            # C_i is correct, C_j is incorrect
            (1 - ci_accuracies["a"])
            * (0 - cj_accuracies["a"])
            * sum(
                {
                    by_label_counts["a"][vp]
                    for vp in pair_vote_patterns[pair][("a", "b")]
                }
            )
            +
            # C_i is incorrect, C_j is correct
            (0 - ci_accuracies["a"])
            * (1 - cj_accuracies["a"])
            * sum(
                {
                    by_label_counts["a"][vp]
                    for vp in pair_vote_patterns[pair][("b", "a")]
                }
            )
            +
            # C_i and C_j are incorrect
            (0 - ci_accuracies["a"])
            * (0 - cj_accuracies["a"])
            * sum(
                {
                    by_label_counts["a"][vp]
                    for vp in pair_vote_patterns[pair][("b", "b")]
                }
            )
        )
        / a_test_size,
        "b": (
            # They are both correct
            (1 - ci_accuracies["b"])
            * (1 - cj_accuracies["b"])
            * sum(
                {
                    by_label_counts["b"][vp]
                    for vp in pair_vote_patterns[pair][("b", "b")]
                }
            )
            +
            # C_i is correct, C_j is incorrect
            (1 - ci_accuracies["b"])
            * (0 - cj_accuracies["b"])
            * sum(
                {
                    by_label_counts["b"][vp]
                    for vp in pair_vote_patterns[pair][("b", "a")]
                }
            )
            +
            # C_i is incorrect, C_j is correct
            (0 - ci_accuracies["b"])
            * (1 - cj_accuracies["b"])
            * sum(
                {
                    by_label_counts["b"][vp]
                    for vp in pair_vote_patterns[pair][("a", "b")]
                }
            )
            +
            # C_i and C_j are incorrect
            (0 - ci_accuracies["b"])
            * (0 - cj_accuracies["b"])
            * sum(
                {
                    by_label_counts["b"][vp]
                    for vp in pair_vote_patterns[pair][("a", "a")]
                }
            )
        )
        / b_test_size,
    }


def ground_truth_statistics(by_label_counts):
    """Given the by-true label voting pattern counts, calculates the complete
    set of sample statistics needed to have an exact polynomial representation
    of the observed voting patterns by three binary classifiers."""
    return {
        "accuracies": classifiers_label_accuracies(by_label_counts),
        "pair-error-correlations": {
            pair: pair_error_correlations(by_label_counts, pair)
            for pair in ((1, 2), (1, 3), (2, 3))
        },
    }


# The first moments of the observable frequencies we are about to encounter are
# familiar ones, the frequencies with which the classifiers voted for each
# of the two labels.
# If a classifier was perfect, this would be a perfect measurement of the
# the prevalence of the labels. A perfect would label each item in the
# test set perfectly and we would just count the 'a' and 'b' decisions to
# compute the unknown prevalence of the true labels.
# When classifiers disagree on these frequencies, you know at least n-1
# cannot possibly be correct - a somewhat trivial universal statement that
# illustrates how evaluation is easier than training.


def label_frequencies_classifiers(by_voting_counts):
    """Calculates the label frequencies noisily counted by the three
    classifiers."""
    totalTestSize = sum(by_voting_counts.values())
    return {
        1: {
            "a": Fraction(
                sum({by_voting_counts[pt] for pt in c1_a_votes}), totalTestSize
            ),
            "b": Fraction(
                sum({by_voting_counts[pt] for pt in c1_b_votes}), totalTestSize
            ),
        },
        2: {
            "a": Fraction(
                sum({by_voting_counts[pt] for pt in c2_a_votes}), totalTestSize
            ),
            "b": Fraction(
                sum({by_voting_counts[pt] for pt in c2_b_votes}), totalTestSize
            ),
        },
        3: {
            "a": Fraction(
                sum({by_voting_counts[pt] for pt in c3_a_votes}), totalTestSize
            ),
            "b": Fraction(
                sum({by_voting_counts[pt] for pt in c3_b_votes}), totalTestSize
            ),
        },
    }


def label_frequencies_classifiers2(voting_frequencies):
    """Convenience function to compare the numerical loss associated
    with going from exact integer ratios to the inexact algebra of
    of the floating point system."""
    return {
        1: {
            "a": sum({voting_frequencies[pt] for pt in c1_a_votes}),
            "b": sum({voting_frequencies[pt] for pt in c1_b_votes}),
        }
    }


# The second group of voting pattern frequency moments should also be
# familiar to experienced readers. And yet, care must be taken to not
# infuse notions of probability distributions to this algebraic approach.
# The second moment we are going to calculate is something like:
# f_a1_a2 - (f_a1)(f_a2)
# This is very similar to the test for independence in a probabilistic
# context if you interpret the "f"s as probabilities. But they are not.
# And this becomes obvious when one is able to prove the following
# equality that tells us there is only one of these quantities to calculate
# because the label designation does not matter! In other words, it is
# universally true that f_a1_a2 - (f_a1)(f_a2) = f_b1_b2 - (f_b1)(f_b2)
# This mathematical equality for voting pattern frequencies is another
# example of how one must tread lightly in Evaluation Land when one has built
# years of intuition in Training Land.
# To whet the readers appetite, we point out that this moment has algebraic
# and engineering significance - it defines a "blindspot" in this algebraic
# evaluator. This is an important topic we gloss over now.


def frequency_moments_pairs(by_voting_counts):
    """Calculates the pair frequency moment difference for all pairs in
    the trio."""
    clfs = label_frequencies_classifiers(by_voting_counts)
    vf = pattern_counts_to_frequencies_exact(by_voting_counts)
    return {
        "a": {
            (1, 2): (
                sum(vf[vp] for vp in pair_vote_patterns[(1, 2)][("a", "a")])
                - clfs[1]["a"] * clfs[2]["a"]
            ),
            (1, 3): (
                sum(vf[vp] for vp in pair_vote_patterns[(1, 3)][("a", "a")])
                - clfs[1]["a"] * clfs[3]["a"]
            ),
            (2, 3): (
                sum(vf[vp] for vp in pair_vote_patterns[(2, 3)][("a", "a")])
                - clfs[2]["a"] * clfs[3]["a"]
            ),
        },
        "b": {
            (1, 2): (
                sum(vf[vp] for vp in pair_vote_patterns[(1, 2)][("b", "b")])
                - clfs[1]["b"] * clfs[2]["b"]
            ),
            (1, 3): (
                sum(vf[vp] for vp in pair_vote_patterns[(1, 3)][("b", "b")])
                - clfs[1]["b"] * clfs[3]["b"]
            ),
            (2, 3): (
                sum(vf[vp] for vp in pair_vote_patterns[(2, 3)][("b", "b")])
                - clfs[2]["b"] * clfs[3]["b"]
            ),
        },
    }


def pairs_agreement_frequencies(by_voting_counts):
    """Calculates the frequency a pair votes in agreement."""
    vf = pattern_counts_to_frequencies_exact(by_voting_counts)
    return {
        "a": {
            (1, 2): (
                sum(vf[vp] for vp in pair_vote_patterns[(1, 2)][("a", "a")])
            ),
            (1, 3): (
                sum(vf[vp] for vp in pair_vote_patterns[(1, 3)][("a", "a")])
            ),
            (2, 3): (
                sum(vf[vp] for vp in pair_vote_patterns[(2, 3)][("a", "a")])
            ),
        },
        "b": {
            (1, 2): (
                sum(vf[vp] for vp in pair_vote_patterns[(1, 2)][("b", "b")])
            ),
            (1, 3): (
                sum(vf[vp] for vp in pair_vote_patterns[(1, 3)][("b", "b")])
            ),
            (2, 3): (
                sum(vf[vp] for vp in pair_vote_patterns[(2, 3)][("b", "b")])
            ),
        },
    }


def pairs_frequency_moments2(by_voting_counts):
    """Function meant to illustrate, via numerical equality, that the 2nd
    moment is the same for either of the two labels."""
    clfs = label_frequencies_classifiers(by_voting_counts)
    vf = project_to_voting_frequencies_fp2(by_voting_counts)
    return {
        (1, 2): (
            (vf[("b", "b", "a")] + vf[("b", "b", "b")])
            - clfs[1]["b"] * clfs[2]["b"]
        ),
        (1, 3): (
            (vf[("b", "a", "b")] + vf[("b", "b", "b")])
            - clfs[1]["b"] * clfs[3]["b"]
        ),
        (2, 3): (
            (vf[("a", "b", "b")] + vf[("b", "b", "b")])
            - clfs[2]["b"] * clfs[3]["b"]
        ),
    }


# The last voting pattern frequency moment we need is one for which we have no
# intuition. It is a polynomial of the observed voting frequencies that
# involves all three classifiers. Its origin can only be elucidated by going
# thru the surface in sample statistics space) from the "evaluation ideal"
# (the set of polynomial equations that relate observable frequencies to
#  unknown sample statistics). See the accompanying Mathematica notebook
# if you are interested in the details of this mathematics.
# Various general remarks can be made to justify its algebraic form:
# 1. It is symmetric in all three of the classifiers.
# 2. it is a cubic in observed frequency variables but degree 6 in
#    evalutaion variables space.
def trio_frequency_moment(by_voting_counts):
    """Calculates the 3rd frequency moment of a trio of binary classifiers
    needed for algebraic evaluation using the error indepedent model."""
    clfs = label_frequencies_classifiers(by_voting_counts)
    productFreqB = clfs[1]["b"] * clfs[2]["b"] * clfs[3]["b"]

    pfmds = frequency_moments_pairs(by_voting_counts)
    sumProd = (
        clfs[1]["b"] * pfmds["b"][(2, 3)]
        + clfs[2]["b"] * pfmds["b"][(1, 3)]
        + clfs[3]["b"] * pfmds["b"][(1, 2)]
    )
    return productFreqB + sumProd


# We now begin defining the coefficients of the prevalence quadratic.
# The work in the Mathematica notebook dedicated to the independent
# evaluator shows that we can create an Elimination Ideal - a sort of
# algebraic ladder where your 1st equation solves for a single one of
# your unknown vars.
# Arbitrarily, but also because it seems symmetric to the task, the choice
# made here is for an elimination ideal that sorts with a polynomiail for
# the alpha label prevalence. That polynomial happens to be a quadratic,
# a(...)*P_alpha^2 + b(...)P_alpha + c(...)
# and we are interested in the values of P_alpha that make this polynomial
# identically zero - the evaluation variety in the P_alpha space.
#
# The polynomial of voting moments associated with the a coefficient:
#
def prevalence_b_coefficientt(eval_data_sketch):
    """Calculates the "a" coefficient associated with the evaluation
    of the sample prevalence for the alpha label."""
    return -prevalence_a_coefficient(eval_data_sketch)


def prevalence_a_coefficient(eval_data_sketch):
    """Calculates the "a" coefficient associated with the evaluation
    of the sample prevalence for the alpha label."""
    pfmds = frequency_moments_pairs(eval_data_sketch)
    vf = pattern_counts_to_frequencies_exact(eval_data_sketch)
    fbbb = vf[("b", "b", "b")]
    diff1 = fbbb - trio_frequency_moment(eval_data_sketch)
    prodFDs = pfmds["b"][(1, 2)] * pfmds["b"][(1, 3)] * pfmds["b"][(2, 3)]
    return diff1**2 + 4 * prodFDs


def prevalence_c_coefficient(eval_data_sketch):
    """Calculates the "c" coefficient associated with the evaluation
    of the sample prevalence for the alpha label."""
    pair_moments = frequency_moments_pairs(eval_data_sketch)
    return (
        -pair_moments["b"][(1, 2)]
        * pair_moments["b"][(1, 3)]
        * pair_moments["b"][(2, 3)]
    )


def alpha_prevalence_estimate(eval_data_sketch):
    """Calculates the prevalence of the alpha label."""
    b = prevalence_b_coefficientt(eval_data_sketch)
    c = prevalence_c_coefficient(eval_data_sketch)
    sqrTerm = math.sqrt(1 - 4 * c / b) / 2
    return [Fraction(1, 2) + sqrTerm, Fraction(1, 2) - sqrTerm]


def gt_alpha_prevalence(by_label_counts):
    """Given ground truth knowledge of the noisy classifiers, calculate
    the true prevalence of the alpha label."""
    nA = sum(c for c in by_label_counts["a"].values())
    nB = sum(c for c in by_label_counts["b"].values())
    return Fraction(nA, nA + nB)


if __name__ == "__main__":
    print(
        """The evaluation observed voting patterns by true label -
        the ground truth."""
    )
    print(uciadult_label_counts, "\n")

    print(
        """During unlabeled evaluation we only get to observe the voting
        patterns, without knowing the true label of each voting instance."""
    )
    data_sketch = to_vote_counts(uciadult_label_counts)
    print(data_sketch, "\n")

    print(
        """To carry out the evaluation we need the relative frequency of the
    voting patterns."""
    )
    observed_frequencies = to_voting_frequency_fractions(
        uciadult_label_counts
    )
    print(observed_frequencies, "\n")

    print(
        """To estimate the prevalence of the alpha label, we need the
    coefficients of the prevalence quadratic:"""
    )
    print("1. The 'a' coefficient for the quadratic:")
    print(prevalence_a_coefficient(data_sketch), "\n")
    print("2. The 'b' coefficient for the quadratic:")
    print(prevalence_b_coefficientt(data_sketch), "\n")
    print("3. The 'c' coefficient for the quadratic:")
    print(prevalence_c_coefficient(data_sketch), "\n")

    print("Algebraic estimates of alpha label prevalence: ")
    print(alpha_prevalence_estimate(data_sketch))
    trueAPrevalence = gt_alpha_prevalence(uciadult_label_counts)
    print(
        "The true alpha label prevalence is: ",
        trueAPrevalence,
        " or: ",
        float(trueAPrevalence),
    )

    # The test run picked for this code comes from a trio of classifiers
    # that have, in fact, very small pair error correlations.
    print("Ground truth values: ")
    print(ground_truth_statistics(uciadult_label_counts))

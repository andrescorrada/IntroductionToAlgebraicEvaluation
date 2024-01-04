#! /usr/bin/env Python
"""Basic utilities for carrying out algebraic evaluation of binary classifiers.
Contains code for evaluating an ensemble of three error-independent judges."""

import math

# The mathematics of evaluating finite samples is, by construction, one
# of estimating integer fractions. We import this module so we can create
# two versions of every computation - the exact one using integer ratios,
# and the one using the default floating point numbers
from fractions import Fraction

# The basic data structure of algebraic evaluation contains the number of
# times an ensemble of judges votes a certain way. In the case of binary
# classification for three judges, there are 8 possible ways they could vote.
# There are two versions of these counts.
# 1. The 1st version is the ``ground truth" version. It contains information
#    that is not available during unlabeled evaluation. Namely, the count of
#    of the pattern by true label. This code uses this version because we are
#    evaluating algebraic evaluation! In other words, our goal is to verify
#    how well purely algebraic evaluation works.
# 2. The 2nd version of the data structure is the data that we would observe
#    in the unlabeled setting where one would want to use this approach. The
#    counts for each voting pattern are now a sum over all possible true
#    labels. The examples that follow are voting pattern counts actually
#    observed when we trained a set of binary classifiers on two public
#    datasets: UCI adul and mushroom.

# Note on label conventions: The labels are, of course, arbitrary. Nonetheless,
# the use of '0' and '1' is not an optimal choice if we want to avoid confusion
# between labels and numbers. In particular, the mathematics of algebraic
# evaluation is based on moments of the correctness of the classifiers
# decisions. One and zero are used to carry out the calculations of these
# moments. To avoid any possible confusion from the indicator functions and the
# labels, we use 'a' and 'b' to denote the two possible labels.

# These counts were obtained in Mathematica using classifiers trained
# on the following features: {(2,4,14),(3,9,11),(6,8,13)}
# The algorithms used by each classifier were: (RandomForest, NeuralNetwork,
# LogisticRegression)
adultLabelCounts = {
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

# The ground truth of the voting pattern counts shown above is the
# ultimate goal of the algebraic evaluation in Data Engine's current
# GrounSeer technology. It is not the only possible test you may want
# to impose on your classifiers. For example, your domain may involve
# sequential data so you would be interested in sequence errors, not
# just the sequence of length 1 voting patterns shown here.
#
# The difficulty of evaluating noisy judges on unlabeled data is that we are
# not able to see the true labels or values for the data. This is a fundamental
# problem that needs to be solved so we can use it where it really matters -
# to increase the safety of deployed AI agents.
#
# Instead, we see a projected count. Any given voting pattern, say (a,b,a),
# is the sum of over the true labels. This is a glimpse into the algebra of
# of evaluation. These "by-label" voting pattern counts can be rewritten
# as polynomials of more familiar "grades" one would want to know for an
# ensemble of classifiers - by-label accuracies for each classifier, etc.
#
# So to aid in creating a simulacrum of how algebraic evaluation would occur
# when you did not have the true labels, let's write a function that projects
# the by-true-label counts into just, by-voting-pattern counts - the only thing
# one can observe in the unlabeled case.
binaryTrioVotingPatterns = (
    ("a", "a", "a"),
    ("a", "a", "b"),
    ("a", "b", "a"),
    ("a", "b", "b"),
    ("b", "a", "a"),
    ("b", "a", "b"),
    ("b", "b", "a"),
    ("b", "b", "b"),
)


def ProjectToVotingPatternCounts(byTrueLabelCounts):
    """Projects by-true-label voting pattern counts to by-voting-pattern
    counts."""
    return {
        votingPattern: (
            byTrueLabelCounts["a"][votingPattern]
            + byTrueLabelCounts["b"][votingPattern]
        )
        for votingPattern in binaryTrioVotingPatterns
    }


# We have now constructed the "easy" 1/2 half of setting up an
# algebraic evaluation. We have 8 voting pattern counts. Can we use
# them to reverse engineer the performance of the classifiers?
#
# Python is not an algebraic language. What is trivial to show, thru
# built-in functions in Mathematica, is extremely hard if not impossible
# in Python. So the following set of functions cannot be motivated
# here by direct appeal to the their origin in the algebra. They are, in
# effect moment functions of the observable counts when viewed as normalized
# frequencies - "38% of the time we saw (a,b,a), etc." So let's start by
# computing those voting pattern frequencies


# The exact computation based on using integer ratios
def ProjectToVotingPatternFrequenciesExact(byTrueLabelCounts):
    """Computes observed voting pattern frequencies."""
    byPatternCounts = ProjectToVotingPatternCounts(byTrueLabelCounts)
    sizeOfTestSet = sum(byPatternCounts.values())
    return {
        vp: Fraction(byPatternCounts[vp], sizeOfTestSet)
        for vp in byPatternCounts.keys()
    }


def ByPatternCountsToFrequenciesExact(byPatternCounts):
    """Computes observerd voting pattern frequencies from
    observed voting pattern counts."""
    sizeOfTestSet = sum(byPatternCounts.values())
    return {
        vp: Fraction(byPatternCounts[vp], sizeOfTestSet)
        for vp in byPatternCounts.keys()
    }


def ProjectToVotingPatternFrequenciesFP(byTrueLabelCounts):
    """Same as the exact computation, but using floating point
    numbers."""
    byPatternCounts = ProjectToVotingPatternCounts(byTrueLabelCounts)
    sizeOfTestSet = sum(byPatternCounts.values())
    return {
        vp: byPatternCounts[vp] / sizeOfTestSet
        for vp in byPatternCounts.keys()
    }


def ProjectToVotingPatternFrequenciesFP2(byPatternCounts):
    """Same as above, but we start from the projected by-pattern counts."""
    sizeOfTestSet = sum(byPatternCounts.values())
    return {
        vp: byPatternCounts[vp] / sizeOfTestSet
        for vp in byPatternCounts.keys()
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
c1VotesA = (("a", "a", "a"), ("a", "a", "b"), ("a", "b", "a"), ("a", "b", "b"))
c1VotesB = (("b", "a", "a"), ("b", "a", "b"), ("b", "b", "a"), ("b", "b", "b"))
# The patterns for classifier 2
c2VotesA = (("a", "a", "a"), ("a", "a", "b"), ("b", "a", "a"), ("b", "a", "b"))
c2VotesB = (("a", "b", "a"), ("a", "b", "b"), ("b", "b", "a"), ("b", "b", "b"))
# The patterns for classifier 3
c3VotesA = (("a", "a", "a"), ("a", "b", "a"), ("b", "a", "a"), ("b", "b", "a"))
c3VotesB = (("a", "a", "b"), ("a", "b", "b"), ("b", "a", "b"), ("b", "b", "b"))


def ClassifiersLabelAccuraciesExact(byTrueLabelCounts):
    """Given the by-true label voting pattern counts, calculates the observed
    by-label accuracies of a trio of classifiers."""
    aTestSize = sum(byTrueLabelCounts["a"].values())
    bTestSize = sum(byTrueLabelCounts["b"].values())
    return {
        1: {
            "a": Fraction(
                sum({byTrueLabelCounts["a"][vp] for vp in c1VotesA}), aTestSize
            ),
            "b": Fraction(
                sum({byTrueLabelCounts["b"][vp] for vp in c1VotesB}), bTestSize
            ),
        },
        2: {
            "a": Fraction(
                sum({byTrueLabelCounts["a"][vp] for vp in c2VotesA}), aTestSize
            ),
            "b": Fraction(
                sum({byTrueLabelCounts["b"][vp] for vp in c2VotesB}), bTestSize
            ),
        },
        3: {
            "a": Fraction(
                sum({byTrueLabelCounts["a"][vp] for vp in c3VotesA}), aTestSize
            ),
            "b": Fraction(
                sum({byTrueLabelCounts["b"][vp] for vp in c3VotesB}), bTestSize
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
pairVotingPatterns = {
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


def ClassifierPairByLabelErrorCorrelations(byTrueLabelCounts, pair):
    """Calculates the by-label pair error correlation for two
    binary classifiers"""
    aTestSize = sum(byTrueLabelCounts["a"].values())
    bTestSize = sum(byTrueLabelCounts["b"].values())

    (ci, cj) = pair
    ca = ClassifiersLabelAccuraciesExact(byTrueLabelCounts)
    ciAccuracies = ca[ci]
    cjAccuracies = ca[cj]

    return {
        "a": (
            # They are both correct
            (1 - ciAccuracies["a"])
            * (1 - cjAccuracies["a"])
            * sum(
                {
                    byTrueLabelCounts["a"][vp]
                    for vp in pairVotingPatterns[pair][("a", "a")]
                }
            )
            +
            # C_i is correct, C_j is incorrect
            (1 - ciAccuracies["a"])
            * (0 - cjAccuracies["a"])
            * sum(
                {
                    byTrueLabelCounts["a"][vp]
                    for vp in pairVotingPatterns[pair][("a", "b")]
                }
            )
            +
            # C_i is incorrect, C_j is correct
            (0 - ciAccuracies["a"])
            * (1 - cjAccuracies["a"])
            * sum(
                {
                    byTrueLabelCounts["a"][vp]
                    for vp in pairVotingPatterns[pair][("b", "a")]
                }
            )
            +
            # C_i and C_j are incorrect
            (0 - ciAccuracies["a"])
            * (0 - cjAccuracies["a"])
            * sum(
                {
                    byTrueLabelCounts["a"][vp]
                    for vp in pairVotingPatterns[pair][("b", "b")]
                }
            )
        )
        / aTestSize,
        "b": (
            # They are both correct
            (1 - ciAccuracies["b"])
            * (1 - cjAccuracies["b"])
            * sum(
                {
                    byTrueLabelCounts["b"][vp]
                    for vp in pairVotingPatterns[pair][("b", "b")]
                }
            )
            +
            # C_i is correct, C_j is incorrect
            (1 - ciAccuracies["b"])
            * (0 - cjAccuracies["b"])
            * sum(
                {
                    byTrueLabelCounts["b"][vp]
                    for vp in pairVotingPatterns[pair][("b", "a")]
                }
            )
            +
            # C_i is incorrect, C_j is correct
            (0 - ciAccuracies["b"])
            * (1 - cjAccuracies["b"])
            * sum(
                {
                    byTrueLabelCounts["b"][vp]
                    for vp in pairVotingPatterns[pair][("a", "b")]
                }
            )
            +
            # C_i and C_j are incorrect
            (0 - ciAccuracies["b"])
            * (0 - cjAccuracies["b"])
            * sum(
                {
                    byTrueLabelCounts["b"][vp]
                    for vp in pairVotingPatterns[pair][("a", "a")]
                }
            )
        )
        / bTestSize,
    }


def GroundTruthSampleStatistics(byTrueLabelCounts):
    """Given the by-true label voting pattern counts, calculates the complete
    set of sample statistics needed to have an exact polynomial representation
    of the observed voting patterns by three binary classifiers."""
    return {
        "accuracies": ClassifiersLabelAccuraciesExact(byTrueLabelCounts),
        "pair-error-correlations": {
            pair: ClassifierPairByLabelErrorCorrelations(
                byTrueLabelCounts, pair
            )
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


def ClassifiersObservedLabelFrequencies(byPatternCounts):
    """Calculates the label frequencies noisily counted by the three
    classifiers."""
    totalTestSize = sum(byPatternCounts.values())
    return {
        1: {
            "a": Fraction(
                sum({byPatternCounts[pt] for pt in c1VotesA}), totalTestSize
            ),
            "b": Fraction(
                sum({byPatternCounts[pt] for pt in c1VotesB}), totalTestSize
            ),
        },
        2: {
            "a": Fraction(
                sum({byPatternCounts[pt] for pt in c2VotesA}), totalTestSize
            ),
            "b": Fraction(
                sum({byPatternCounts[pt] for pt in c2VotesB}), totalTestSize
            ),
        },
        3: {
            "a": Fraction(
                sum({byPatternCounts[pt] for pt in c3VotesA}), totalTestSize
            ),
            "b": Fraction(
                sum({byPatternCounts[pt] for pt in c3VotesB}), totalTestSize
            ),
        },
    }


def ClassifiersObservedLabelFrequencies2(votingFrequencies):
    """Convenience function to compare the numerical loss associated
    with going from exact integer ratios to the inexact algebra of
    of the floating point system."""
    return {
        1: {
            "a": sum({votingFrequencies[pt] for pt in c1VotesA}),
            "b": sum({votingFrequencies[pt] for pt in c1VotesB}),
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


def PairsFrequencyMomentDifference(byPatternCounts):
    """Calculates the pair frequency moment difference for all pairs in
    the trio."""
    clfs = ClassifiersObservedLabelFrequencies(byPatternCounts)
    vf = ByPatternCountsToFrequenciesExact(byPatternCounts)
    return {
        "a": {
            (1, 2): (
                sum(vf[vp] for vp in pairVotingPatterns[(1, 2)][("a", "a")])
                - clfs[1]["a"] * clfs[2]["a"]
            ),
            (1, 3): (
                sum(vf[vp] for vp in pairVotingPatterns[(1, 3)][("a", "a")])
                - clfs[1]["a"] * clfs[3]["a"]
            ),
            (2, 3): (
                sum(vf[vp] for vp in pairVotingPatterns[(2, 3)][("a", "a")])
                - clfs[2]["a"] * clfs[3]["a"]
            ),
        },
        "b": {
            (1, 2): (
                sum(vf[vp] for vp in pairVotingPatterns[(1, 2)][("b", "b")])
                - clfs[1]["b"] * clfs[2]["b"]
            ),
            (1, 3): (
                sum(vf[vp] for vp in pairVotingPatterns[(1, 3)][("b", "b")])
                - clfs[1]["b"] * clfs[3]["b"]
            ),
            (2, 3): (
                sum(vf[vp] for vp in pairVotingPatterns[(2, 3)][("b", "b")])
                - clfs[2]["b"] * clfs[3]["b"]
            ),
        },
    }


def PairsFrequencyAgreement(byPatternCounts):
    """Calculates the frequency a pair votes in agreement."""
    vf = ByPatternCountsToFrequenciesExact(byPatternCounts)
    return {
        "a": {
            (1, 2): (
                sum(vf[vp] for vp in pairVotingPatterns[(1, 2)][("a", "a")])
            ),
            (1, 3): (
                sum(vf[vp] for vp in pairVotingPatterns[(1, 3)][("a", "a")])
            ),
            (2, 3): (
                sum(vf[vp] for vp in pairVotingPatterns[(2, 3)][("a", "a")])
            ),
        },
        "b": {
            (1, 2): (
                sum(vf[vp] for vp in pairVotingPatterns[(1, 2)][("b", "b")])
            ),
            (1, 3): (
                sum(vf[vp] for vp in pairVotingPatterns[(1, 3)][("b", "b")])
            ),
            (2, 3): (
                sum(vf[vp] for vp in pairVotingPatterns[(2, 3)][("b", "b")])
            ),
        },
    }


def PairsFrequencyMoment2(byPatternCounts):
    """Function meant to illustrate, via numerical equality, that the 2nd
    moment is the same for either of the two labels."""
    clfs = ClassifiersObservedLabelFrequencies(byPatternCounts)
    vf = ProjectToVotingPatternFrequenciesFP2(byPatternCounts)
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
def TrioFrequencyMoment(byPatternCounts):
    """Calculates the 3rd frequency moment of a trio of binary classifiers
    needed for algebraic evaluation using the error indepedent model."""
    clfs = ClassifiersObservedLabelFrequencies(byPatternCounts)
    productFreqB = clfs[1]["b"] * clfs[2]["b"] * clfs[3]["b"]

    pfmds = PairsFrequencyMomentDifference(byPatternCounts)
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
def prevalenceEvaluationQuadraticBCoefficient(evalDataSketch):
    """Calculates the "a" coefficient associated with the evaluation
    of the sample prevalence for the alpha label."""
    return -prevalenceEvaluationQuadraticACoefficient(evalDataSketch)


def prevalenceEvaluationQuadraticACoefficient(evalDataSketch):
    """Calculates the "a" coefficient associated with the evaluation
    of the sample prevalence for the alpha label."""
    # clfs = ClassifiersObservedLabelFrequencies(evalDataSketch)
    pfmds = PairsFrequencyMomentDifference(evalDataSketch)
    vf = ByPatternCountsToFrequenciesExact(evalDataSketch)
    fbbb = vf[("b", "b", "b")]
    diff1 = fbbb - TrioFrequencyMoment(evalDataSketch)
    prodFDs = pfmds["b"][(1, 2)] * pfmds["b"][(1, 3)] * pfmds["b"][(2, 3)]
    return diff1**2 + 4 * prodFDs


def prevalenceEvaluationQuadraticCCoefficient(evalDataSketch):
    """Calculates the "c" coefficient associated with the evaluation
    of the sample prevalence for the alpha label."""
    pairMoments = PairsFrequencyMomentDifference(evalDataSketch)
    return (
        -pairMoments["b"][(1, 2)]
        * pairMoments["b"][(1, 3)]
        * pairMoments["b"][(2, 3)]
    )


def alphaPrevalencAlgebraicEstimate(evalDataSketch):
    """Calculates the prevalence of the alpha label."""
    b = prevalenceEvaluationQuadraticBCoefficient(evalDataSketch)
    c = prevalenceEvaluationQuadraticCCoefficient(evalDataSketch)
    sqrTerm = math.sqrt(1 - 4 * c / b) / 2
    return [Fraction(1, 2) + sqrTerm, Fraction(1, 2) - sqrTerm]


def GTAlphaPrevalence(byTrueLabelCounts):
    """Given ground truth knowledge of the noisy classifiers, calculate
    the true prevalence of the alpha label."""
    nA = sum(c for c in byTrueLabelCounts["a"].values())
    nB = sum(c for c in byTrueLabelCounts["b"].values())
    return Fraction(nA, nA + nB)


if __name__ == "__main__":
    print(
        """The evaluation observed voting patterns by true label -
        the ground truth."""
    )
    print(adultLabelCounts, "\n")

    print(
        """During unlabeled evaluation we only get to observe the voting
        patterns, without knowing the true label of each voting instance."""
    )
    evalDataSketch = ProjectToVotingPatternCounts(adultLabelCounts)
    print(evalDataSketch, "\n")

    print(
        """To carry out the evaluation we need the relative frequency of the
    voting patterns."""
    )
    votingFrequencies = ProjectToVotingPatternFrequenciesExact(
        adultLabelCounts
    )
    print(votingFrequencies, "\n")

    print(
        """To estimate the prevalence of the alpha label, we need the
    coefficients of the prevalence quadratic:"""
    )
    print("1. The 'a' coefficient for the quadratic:")
    print(prevalenceEvaluationQuadraticACoefficient(evalDataSketch), "\n")
    print("2. The 'b' coefficient for the quadratic:")
    print(prevalenceEvaluationQuadraticBCoefficient(evalDataSketch), "\n")
    print("3. The 'c' coefficient for the quadratic:")
    print(prevalenceEvaluationQuadraticCCoefficient(evalDataSketch), "\n")

    print("Algebraic estimates of alpha label prevalence: ")
    print(alphaPrevalencAlgebraicEstimate(evalDataSketch))
    trueAPrevalence = GTAlphaPrevalence(adultLabelCounts)
    print(
        "The true alpha label prevalence is: ",
        trueAPrevalence,
        " or: ",
        float(trueAPrevalence),
    )

    # The test run picked for this code comes from a trio of classifiers
    # that have, in fact, very small pair error correlations.
    print("Ground truth values: ")
    print(GroundTruthSampleStatistics(adultLabelCounts))

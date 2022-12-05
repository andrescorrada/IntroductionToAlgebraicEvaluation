#! /usr/bin/env Python
"""Basic utilities for carrying out algebraic evaluation of binary classifiers.
Contains code for evaluating an ensemble of three error-independent judges."""

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
#    counts for each voting pattern are now a sum over all possible true labels.
#    The examples that follow are voting pattern counts actually observed when
#    we trained a set of binary classifiers on two public datasets: UCI adult
#    and mushroom

# Note on label conventions: The labels are, of course, arbitrary. Nonetheless,
# the use of '0' and '1' is not an optimal choice if we want to avoid confusion
# between labels and numbers. In particular, the mathematics of algebraic
# evaluation is based on moments of the correctness of the classifiers decisions.
# One and zero are used to carry out the calculations of these moments. To avoid
# any possible confusion from the indicator functions and the labels, we use
# 'a' and 'b' to denote the two possible labels.

# These counts were obtained in Mathematica using classifiers trained
# on the following features: {(2,4,14),(3,9,11),(6,8,13)}
# The algorithms used by each classifier were: (RandomForest, NeuralNetwork,
# LogisticRegression)
adultLabelCounts = {
'a': {('a', 'a', 'a'): 715,
      ('a', 'a', 'b'): 161,
      ('a', 'b', 'a'): 2406,
      ('a', 'b', 'b'): 455,
      ('b', 'a', 'a'): 290,
      ('b', 'a', 'b'): 94,
      ('b', 'b', 'a'): 1335,
      ('b', 'b', 'b'): 231},
'b': {('a', 'a', 'a'): 271,
      ('a', 'a', 'b'): 469,
      ('a', 'b', 'a'): 3395,
      ('a', 'b', 'b'): 7517,
      ('b', 'a', 'a'): 272,
      ('b', 'a', 'b'): 399,
      ('b', 'b', 'a'): 6377,
      ('b', 'b', 'b'): 12455}}

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
binaryTrioVotingPatterns = (('a', 'a', 'a'),
                            ('a', 'a', 'b'),
                            ('a', 'b', 'a'),
                            ('a', 'b', 'b'),
                            ('b', 'a', 'a'),
                            ('b', 'a', 'b'),
                            ('b', 'b', 'a'),
                            ('b', 'b', 'b'))

def ProjectToVotingPatternCounts(byTrueLabelCounts):
    """Projects by-true-label voting pattern counts to by-voting-pattern counts."""
    return {votingPattern:(
                    byTrueLabelCounts['a'][votingPattern] +
                    byTrueLabelCounts['b'][votingPattern]) for \
                    votingPattern in binaryTrioVotingPatterns}

# We have now constructed the "easy" 1/2 half of setting up an
# algebraic evaluation. We have 8 voting pattern counts. Can we use
# them to reverse engineer the performance of the classifiers?
#
# Python is not an algebraic language. What is trivial to show, thru
# built-in functions in Mathematica, is extremely hard if not impossible
# in Python. So the following set of functions cannot be motivated
# here by direct appeal to the their origin in the algebra. They are, in
# effect moment functions of the observable counts when viewed as normalized
# frequencies - "38% of the time we saw (a,b,a), etc."
def ProjectToVotingPatternFrequencies(byTrueLabelCounts):
    """Projects to unit-normalized voting pattern frequencies. Unfortunately,
    this will introduce real numbers! This already obscures the importance
    of remaining within an algebraic number system. But we digress. That
    insight is for later. We accept the imprecision this brings. It will
    obscure some of the information one can gain by having exact integer
    mathematics but it will not affect its numerical accuracy."""
    byPatternCounts = ProjectToVotingPatternCounts(byTrueLabelCounts)
    sizeOfTestSet = sum(byPatternCounts.values())
    return {vp:byPatternCounts[vp]/sizeOfTestSet
            for vp in byPatternCounts.keys()}

def ProjectToVotingPatternFrequencies2(byPatternCounts):
    """Same as above, but we start from the projected by-pattern counts."""
    sizeOfTestSet = sum(byPatternCounts.values())
    return {vp:byPatternCounts[vp]/sizeOfTestSet
            for vp in byPatternCounts.keys()}

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
# In addition, we are going to code to versions of the moments to see how
# floating point arithmetic is causing exact frequencies to become inexact.
# To make the logic clearer, let's enumerate the voting patterns where
# each classifier votes a given label.
(('a', 'a', 'a'),
                            ('a', 'a', 'b'),
                            ('a', 'b', 'a'),
                            ('a', 'b', 'b'),
                            ('b', 'a', 'a'),
                            ('b', 'a', 'b'),
                            ('b', 'b', 'a'),
                            ('b', 'b', 'b'))
# The patterns for classifier 1
c1VotesA = (('a', 'a', 'a'),
            ('a', 'a', 'b'),
            ('a', 'b', 'a'),
            ('a', 'b', 'b'))
c1VotesB = (('b', 'a', 'a'),
            ('b', 'a', 'b'),
            ('b', 'b', 'a'),
            ('b', 'b', 'b'))
# The patterns for classifier 2
c2VotesA = (('a', 'a', 'a'),
            ('a', 'a', 'b'),
            ('b', 'a', 'a'),
            ('b', 'a', 'b'))
c2VotesB = (('a', 'b', 'a'),
            ('a', 'b', 'b'),
            ('b', 'b', 'a'),
            ('b', 'b', 'b'))
# The patterns for classifier 3
c3VotesA = (('a', 'a', 'a'),
            ('a', 'b', 'a'),
            ('b', 'a', 'a'),
            ('b', 'b', 'a'))
c3VotesB = (('a', 'a', 'b'),
            ('a', 'b', 'b'),
            ('b', 'a', 'b'),
            ('b', 'b', 'b'))


def ClassifiersObservedLabelFrequencies(byPatternCounts):
    """Calculates the label frequencies noisily counted by the three
    classifiers."""
    totalTestSize = sum(byPatternCounts.values())
    return {1:{
                'a':sum({byPatternCounts[pt] for
                    pt in c1VotesA})/totalTestSize,
                'b':sum({byPatternCounts[pt] for
                    pt in c1VotesB})/totalTestSize},
            2:{
                'a':sum({byPatternCounts[pt] for
                    pt in c2VotesA})/totalTestSize,
                'b':sum({byPatternCounts[pt] for
                    pt in c2VotesB})/totalTestSize},
            3:{
                'a':sum({byPatternCounts[pt] for
                    pt in c3VotesA})/totalTestSize,
                'b':sum({byPatternCounts[pt] for
                    pt in c3VotesB})/totalTestSize}}

def ClassifiersObservedLabelFrequencies2(votingFrequencies):
    """Convenience function to compare the numerical loss associated
    with going from exact integer ratios to the inexact algebra of
    of the floating point system."""
    return {1:{
                'a':sum({votingFrequencies[pt] for pt in c1VotesA}),
                'b':sum({votingFrequencies[pt] for pt in c1VotesB})}}

# The second group of voting pattern frequency moments should also be
# familiar to experienced readers. And yet, care must be taken to not
# infuse notions of probability distributions to this algebraic approach.
# The second moment we are going to calculate is something like:
# f_1a_2a - (f_1a)(f_2a)
# This is very similar to the test for independence in a probabilistic
# context if you interpret the "f"s as probabilities. But they are not.
# And this becomes obvious when one is able to prove the following
# equality that tells us there is only one of these quantities to calculate
# because the label designation does not matter! In other words, it is
# universally true that f_1a_2a - (f_1a)(f_2a) = f_1b_2b - (f_1b)(f_2b)
# This mathematical equality for voting pattern frequencies is another
# example of how one must tread lightly in Evaluation Land when one has built
# years of intuition in Training Land.
# To whet the readers appetite, we point out that this moment has algebraic
# and engineering significance - it defines a "blindspot" in this algebraic
# evaluator. This is an important topic we gloss over now.
def PairsFrequencyMoment(byPatternCounts):
    """Calculates the pair moment that defines the pair error correlation
    blindspots."""
    clfs = ClassifiersObservedLabelFrequencies(byPatternCounts)
    vf = ProjectToVotingPatternFrequencies2(byPatternCounts)
    return {(1,2):((vf[('a','a','a')] + vf[('a','a','b')]) - clfs[1]['a']*clfs[2]['a']),
            (1,3):((vf[('a','a','a')] + vf[('a','b','a')]) - clfs[1]['a']*clfs[3]['a']),
            (2,3):((vf[('a','a','a')] + vf[('b','a','a')]) - clfs[2]['a']*clfs[3]['a'])}

def PairsFrequencyMoment2(byPatternCounts):
    """Function meant to illustrate, via numerical equality, that the 2nd moment is
    the same for either of the two labels."""
    clfs = ClassifiersObservedLabelFrequencies(byPatternCounts)
    vf = ProjectToVotingPatternFrequencies2(byPatternCounts)
    return {(1,2):((vf[('b','b','a')] + vf[('b','b','b')]) - clfs[1]['b']*clfs[2]['b']),
            (1,3):((vf[('b','a','b')] + vf[('b','b','b')]) - clfs[1]['b']*clfs[3]['b']),
            (2,3):((vf[('a','b','b')] + vf[('b','b','b')]) - clfs[2]['b']*clfs[3]['b'])}


if __name__ == '__main__':
    print(adultLabelCounts)

    byPatternCounts = ProjectToVotingPatternCounts(adultLabelCounts)
    print(byPatternCounts)

    votingFrequencies = ProjectToVotingPatternFrequencies(adultLabelCounts)
    print(votingFrequencies)

    print(ClassifiersObservedLabelFrequencies(byPatternCounts))
    print(ClassifiersObservedLabelFrequencies2(votingFrequencies))
    print(PairsFrequencyMoment(byPatternCounts))
    print(PairsFrequencyMoment2(byPatternCounts))

# -*- coding: utf-8 -*-
"""Algebrai postulates for the logic of evaluation on unlabeled R=2 tests.

    There are two equivalent ways to demonstrate the postulates. The first
    is the familiar case from machine learning of classification. This is
    the case where the labels 'a' and 'b' have the same semantic meaning
    across the different items used in the evaluation.
    The second case is where 'a' and 'b' are arbitrary to each item/question
    in the test and we are viewing the test as a multiple-choice exam with
    two responses possible for each question.
    Mathematically we can view the R=2 response viewpoint as the integer
    version of the binary classification viewpoint.
    In practical terms, it means that when this mathematics/logic is
    applied to classification tests, the measurement of the prevalence of
    the 'a' and 'b' labels refers to a property of the set of items classified.
    But when we view it as a binary response multiple-choice exam, there is no
    significance other than as statistics of the test. So, for example, you
    could use this to grade a Geology, Cosmology, etc. exam that had two
    possible responses for each question.
    Classification
    --------------
        The prevalences and label accuracy variables.
    Grading the evaluation of an LLM
    --------------------------------
        The percentage of correct questions: P_a * P_i_a + P_b * P_i_b.
        The auxiliary variables P_a, P_b, P_i_a, P_i_b are being used to
        compute the percentage of correct responses on the test.
        We say - "I got 83% on that exam." Not - "I got 91% on the 'a'
        questions and 79% on the 'b' questions"
        This viewpoint is the one you would use to evaluate something like
        a LLM across multiple queries that asked it to pick between two
        possible responses.
    """

import sympy

# The prevalences of the two labels
pa = sympy.Symbol(r"P_a")
pb = sympy.Symbol(r"P_b")

# The label accuracy for the classifiers "i", "j", "k"
pia = sympy.Symbol(r"P_{i, a}")
pib = sympy.Symbol(r"P_{i, b}")
pja = sympy.Symbol(r"P_{j, a}")
pjb = sympy.Symbol(r"P_{j, b}")
pka = sympy.Symbol(r"P_{k, a}")
pkb = sympy.Symbol(r"P_{k, b}")

# Postulates before observing test results
# One postulate is just about the prelances
prelances_sum_to_one = pa + pb - 1

# We omit all the linear relations that relate binary erorrs to accuracy.

# Postulates having observed responses
fai = sympy.Symbol(r"f_{a_i}")
fbi = sympy.Symbol(r"f_{b_i}")
faj = sympy.Symbol(r"f_{a_j}")
fbj = sympy.Symbol(r"f_{b_j}")

# The 'generating set' of polynomials for any binary classifier
single_binary_classifier_generating_set = [
    pa * pia + pb * (1 - pib) - fai,
    pa * (1 - pia) + pb * pib - fbi,
]

# The single classifier generating set only creates one postulate
single_binary_classifier_postulate = pa * (pia - fai) - pb * (pib - fbi)

# Data sketch observables for the pair
faiaj = sympy.Symbol(r"f_{a_i, a_j}")
faibj = sympy.Symbol(r"f_{a_i, b_j}")
fbiaj = sympy.Symbol(r"f_{b_i, a_j}")
fbibj = sympy.Symbol(r"f_{b_i, b_j}")

# The delta moment for classifier pairs
deltaij = sympy.Symbol(r"\Delta_{i,j}")

# Pair correlation variables for the two labels
gija = sympy.Symbol(r"\Gamma_{i, j, a}")
gijb = sympy.Symbol(r"\Gamma_{i, j, b}")

# The 'generating set' of polynomials for any pair of binary classifiers
pair_binary_classifiers_generating_set = [
    pa * (pia * pja + gija) + pb * ((1 - pib) * (1 - pjb) + gijb) - faiaj,
    pa * (pia * (1 - pja) - gija) + pb * ((1 - pib) * pjb - gijb) - faibj,
    pa * ((1 - pia) * pja - gija) + pb * (pib * (1 - pjb) - gijb) - fbiaj,
    pa * ((1 - pia) * (1 - pja) + gija) + pb * (pib * pjb + gijb) - fbibj,
]

# The Groebner basis for the classifier pair generating set produces a
# set of partially independent postulates
pair_binary_classifiers_postulates = [
    (pia - fai) * (pjb - fbj) - (pib - fbi) * (pja - faj),
    (pia - fai) * (pjb - fbj) * ((pia - fai) + (pib - fbi))
    + (pia - fai) * (gijb - deltaij)
    + (pib - fbi) * (gija - deltaij),
    (pia - fai) * (pjb - fbj) * ((pja - faj) + (pjb - fbj))
    + (pja - faj) * (gijb - deltaij)
    + (pjb - fbj) * (gija - deltaij),
]

if __name__ == "__main__":
    from pprint import pprint

    sympy.init_printing(use_unicode=True)

    print(pia)
    print(pib)
    print(pja)
    print(pjb)
    print(pka)
    print(pkb)

    print(prelances_sum_to_one)
    pprint(single_binary_classifier_generating_set)

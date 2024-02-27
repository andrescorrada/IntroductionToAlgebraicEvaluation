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
# One postulate is just about the prevalences
prelances_sum_to_one = pa + pb - 1

# We omit all the linear relations that relate binary erorrs to accuracy.
# By this is meant that we could work in a larger variable space that
# included the error rate for a label. This would require that we
# add equations of the type P_i_a_a + P_i_b_a = 1 to the generating set.
# So instead we just express everything in therms of binary label accuracy
# and dispense with writing out error rate variables.
# Dropping the error rate variables also allows the syntactic sugar of
# referring to P_i_a_a (the percentage of times classifier i said the label
# was "a" AND it was "a")as just P_i_a.

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
gika = sympy.Symbol(r"\Gamma_{i, k, a}")
gikb = sympy.Symbol(r"\Gamma_{i, k, b}")
gjka = sympy.Symbol(r"\Gamma_{j, k, a}")
gjkb = sympy.Symbol(r"\Gamma_{j, k, b}")

# 3-way correlations for the two labels
gijka = sympy.Symbol(r"\Gamma_{i, j, k, a}")
gijkb = sympy.Symbol(r"\Gamma_{i, j, k, b}")

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

# Data sketch observables for the trio
faiajak = sympy.Symbol(r"f_{a_i, a_j, a_k}")
faiajbk = sympy.Symbol(r"f_{a_i, a_j, b_k}")
faibjak = sympy.Symbol(r"f_{a_i, b_j, a_k}")
fbiajak = sympy.Symbol(r"f_{b_i, a_j, a_k}")

fbibjbk = sympy.Symbol(r"f_{b_i, b_j, b_k}")
fbibjak = sympy.Symbol(r"f_{b_i, b_j, a_k}")
fbiajbk = sympy.Symbol(r"f_{b_i, a_j, b_k}")
faibjbk = sympy.Symbol(r"f_{a_i, b_j, b_k}")

trio_frequencies = [
    faiajak,
    faiajbk,
    faibjak,
    faibjbk,
    fbiajak,
    fbiajbk,
    fbibjak,
    fbibjbk,
]

trio_binary_classifiers_generating_set = [
    pa * (pia * pja * pka + gija * pka + gika * pja + gjka * pia + gijka)
    + pb
    * (
        (1 - pib) * (1 - pjb) * (1 - pkb)
        + gijb * (1 - pkb)
        + gikb * (1 - pjb)
        + gjkb * (1 - pib)
        - gijkb
    )
    - faiajak,
    pa
    * (
        -gijka
        + (1 - pka) * gija
        - pja * gika
        - pia * gjka
        + pia * pja * (1 - pka)
    )
    + pb
    * (
        gijkb
        + pkb * gijb
        - (1 - pjb) * gikb
        - (1 - pib) * gjkb
        + (1 - pib) * (1 - pjb) * pkb
    )
    - faiajbk,
    pa
    * (
        pia * (1 - pja) * pka
        - pka * gija
        + (1 - pja) * gika
        - pia * gjka
        - gijka
    )
    + pb
    * (
        (1 - pib) * pjb * (1 - pkb)
        - (1 - pkb) * gijb
        + pjb * gikb
        - (1 - pib) * gjkb
        + gijkb
    )
    - faibjak,
    pa
    * (
        pia * (1 - pja) * (1 - pka)
        - (1 - pka) * gija
        - (1 - pja) * gika
        + pia * gjka
        + gijka
    )
    + pb
    * (
        (1 - pib) * pjb * pkb
        - pkb * gijb
        - pjb * gikb
        + (1 - pib) * gjkb
        - gijkb
    )
    - faibjbk,
    pa
    * (
        (1 - pia) * pja * pka
        - pka * gija
        - pja * gika
        + (1 - pia) * gjka
        - gijka
    )
    + pb
    * (
        pib * (1 - pjb) * (1 - pkb)
        - (1 - pkb) * gijb
        - (1 - pjb) * gikb
        + pib * gjkb
        + gijkb
    )
    - fbiajak,
    pa
    * (
        (1 - pia) * pja * (1 - pka)
        - (1 - pka) * gija
        + pja * gija
        - (1 - pia) * gjka
        + gijka
    )
    + pb
    * (
        pib * (1 - pjb) * pkb
        - pkb * gijb
        + (1 - pjb) * gikb
        - pib * gjkb
        - gijkb
    )
    - fbiajbk,
    pa
    * (
        (1 - pia) * (1 - pja) * pka
        + pka * gija
        - (1 - pja) * gika
        - (1 - pia) * gjka
        + gijka
    )
    + pb
    * (
        pib * pjb * (1 - pkb)
        + (1 - pkb) * gijb
        - pjb * gikb
        - pib * gjkb
        - gijkb
    )
    - fbibjak,
    pa
    * (
        (1 - pia) * (1 - pja) * (1 - pka)
        + (1 - pka) * gija
        + (1 - pja) * gika
        + (1 - pia) * gjka
        - gijka
    )
    + pb * (pib * pjb * pkb + pkb * gijb + pjb * gikb + pib * gjkb + gijkb)
    - fbibjbk,
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

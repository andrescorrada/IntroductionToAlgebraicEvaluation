# -*- coding: utf-8 -*-
"""R=2 evaluation axioms in percentage space.

This is the space of rationals - ratios of integers - where the
sample statistics of correctness in a test are expressed as sample
percentages.
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

# Axioms before observing test results
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
fak = sympy.Symbol(r"f_{a_k}")
fbk = sympy.Symbol(r"f_{b_k}")

# The 'generating set' of polynomials for any binary classifier
single_binary_classifier_generating_set = [
    pa * pia + pb * (1 - pib) - fai,
    pa * (1 - pia) + pb * pib - fbi,
]

# The single classifier generating set only creates one postulate
single_binary_classifier_axiom = pa * (pia - fai) - pb * (pib - fbi)

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
pair_binary_classifiers_axioms = [
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

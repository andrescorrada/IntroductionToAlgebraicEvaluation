"""R=2 evaluation axioms in integer space.

This is the space of integers possible since we are dealing with finite
tests. Statistics of correctness are given as integers. In the case of
R=2 the evaluation of a single classifier/responder is (Q_a, R_a_a, R_b_b).
"""
import sympy

#
# Observables
#
q = sympy.Symbol(r"Q")

rai = sympy.Symbol(r"R_{a_i}")
rbi = sympy.Symbol(r"R_{b_i}")
raj = sympy.Symbol(r"R_{a_j}")
rbj = sympy.Symbol(r"R_{b_j}")
rak = sympy.Symbol(r"R_{a_k}")
rbk = sympy.Symbol(r"R_{b_k}")

raiaj = sympy.Symbol(r"R_{a_i, a_j}")
raiak = sympy.Symbol(r"R_{a_i, a_k}")
rajak = sympy.Symbol(r"R_{a_j, a_k}")

rbibj = sympy.Symbol(r"R_{b_i, b_j}")
rbibk = sympy.Symbol(r"R_{b_i, b_k}")
rbjbk = sympy.Symbol(r"R_{b_j, b_k}")

#
# Statistics of correctness
#

# The number of 'a' and 'b' questions in the test
qa = sympy.Symbol(r"Q_a")
qb = sympy.Symbol(r"Q_b")

# Postulates before observing test results
# One postulate is just about the size of the test and how it is
# composed of 'a' and 'b' correct questions.
correct_answers_must_equal_test_size = qa + qb - q

# Insteard of percentage correct, we talk about total number
# of correct responses to a question type
raia = sympy.Symbol(r"R_{a_i,a}")
rbib = sympy.Symbol(r"R_{b_i,b}")

# The single classifier generating set only creates one postulate
single_binary_responder_axiom = (raia - rbib) - (qa - rbi)

# Symbols and axioms for pairs
raja = sympy.Symbol(r"R_{a_j,a}")
rbjb = sympy.Symbol(r"R_{b_j,b}")

raka = sympy.Symbol(r"R_{a_k,a}")
rbkb = sympy.Symbol(r"R_{b_k,b}")

raiaja = sympy.Symbol(r"R_{a_i, a_j; a}")
raiaka = sympy.Symbol(r"R_{a_i, a_k; a}")
rajaka = sympy.Symbol(r"R_{a_j, a_k; a}")

rbibjb = sympy.Symbol(r"R_{b_i, b_j; b}")
rbibkb = sympy.Symbol(r"R_{b_i, b_k; b}")
rbjbkb = sympy.Symbol(r"R_{b_j, b_k; b}")

pair_correlation_axiom_first_type = (
    q * qa * qb * (raiaja + rbibjb)
    - q * (qb * raia * raja + qa * rbib * rbjb)
    - qa * qb * (q * rbibj - rbi * rbj)
    + (q * raia - qa * rai) * (q * rbjb - qb * rbj)
)

pair_correlation_axiom_three_classifiers = (
    q
    * (q * raia - qa * rai)
    * (q * rbkb - qb * rbk)
    * (qb * raja + qa * rbjb - qa * qb)
    + q**2 * qa * (q * raka - qa * rak) * (qb * rbibjb - rbib * rbjb)
    + q**2 * qb * (q * rbkb - qb * rbk) * (qa * raiaja - raia * raja)
    - q * qa * qb * (q * rbibj - rbi * rbj) * (qb * raka + qa * rbkb - qa * qb)
)

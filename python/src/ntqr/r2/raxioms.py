"""R=2 evaluation axioms in integer space.

This is the space of integers possible since we are dealing with finite
tests. Statistics of correctness are given as integers. In the case of
R=2 the evaluation of a single classifier/responder is (Q_a, R_a_a, R_b_b).
"""
import sympy

q = sympy.Symbol(r"Q")
# The number of 'a' and 'b' questions in the test
qa = sympy.Symbol(r"Q_a")
qb = sympy.Symbol(r"Q_b")

# Postulates before observing test results
# One postulate is just about the size of the test and how it is
# composed of 'a' and 'b' correct questions.
correct_answers_must_equal_test_size = qa + qb - q

# Insteard of percentage correct, we talk about total number
# of correct responses to a question type
rai = sympy.Symbol(r"R_{a_i}")
rbi = sympy.Symbol(r"R_{b_i}")
raia = sympy.Symbol(r"R_{a_i,a}")
rbib = sympy.Symbol(r"R_{b_i,b}")

# The single classifier generating set only creates one postulate
single_binary_responder_postulate = (raia - rbib) - (qa - rbi)

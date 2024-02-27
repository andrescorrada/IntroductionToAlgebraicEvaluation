"""Postulates of evaluation for R=3 tests.

The number of decision patterns when 3 responses to a question are possible
is 3^N, where N is the number of responders. As in the R=2 postulates case,
we can start with statistics of correctness in a test that only involve a
single responder. We begin the build out of the R=3 postulates with the N=1
postulates, in other words. Then we will proceed to the N=2 postulates, the
N=3 postulates and so on.

Although we suspect that the N=3, R=3 error independent model also has a
closed algebraic solution, we have failed to get one using 128GB of RAM.
But we are publishing the N=1 case now since it is solvable.

Variables
---------

The case of R=3 is interesting because it brings into relief what is barely
visible in the R=2 case - there are many more ways to be wrong than right.
In binary response tests there is only way to be wrong. Now we have two ways
of responding incorrectly.

But the normalization condition for being right and wrong still holds.
The sum of correct answers and incorrect answers must sum to the number
of questions on the test. So we have gone from something like,
    x + y = 1
to
    x + y + z = 1
So what variables should we choose to simplify the math of the generating
polynomials? The "natural" choice is to express the label accuracies in
terms of the two wrong choices. And this is a choice that is going to work
for any R so it is the one we choose here.

So, for example, assuming the three labels are given by (a, b, c). The 'a'
label accuracy would expressed in terms of the two error rates as,
P_i_a_a = 1 - P_i_b_a - P_i_c_a
"""

import sympy

# The observable voting frequencies for the single 3-label classifier
single_classifier_voting_frequencies = []
for var_name in (r"f_a", r"f_b", r"f_c"):
    globals()[var_name] = sympy.Symbol(var_name)
    single_classifier_voting_frequencies.append(globals()[var_name])
print(single_classifier_voting_frequencies)

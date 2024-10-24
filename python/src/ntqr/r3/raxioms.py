"""Axioms of evaluation for R=3 tests.

The number of decision patterns when 3 responses to a question are possible
is 3^N, where N is the number of responders. As in the R=2 axioms case,
we can start with statistics of correctness in a test that only involve a
single responder. We begin the build out of the R=3 axioms with the N=1
axioms, in other words. Then we will proceed to the N=2 axioms, the
N=3 postulates and so on.

Although we suspect that the N=3, R=3 error independent model also has a
closed algebraic solution, we have failed to get one using 128GB of RAM.
But we are publishing the N=1 case now since it is solvable. In addition,
it is possible to build logical alarms for misaligned classifiers just
using the single classifier axioms. This is good news since the single
classifier case axioms are easily constructed for any finite number of
classes/responses.

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
label accuracy would be expressed in terms of the two error rates as,
P_i_a_a = 1 - P_i_b_a - P_i_c_a
"""

import sympy
from ntqr.statistics import SingleClassifierVariables


class SingleClassifierAxioms:
    """
    The axioms of a single R=3 classifier

    ...

    Attributes
    ----------

    question_numbers : List[simpy.Symbol..]
        The variables for the count of correct questions for each true
        label.

    responses : List[simpy.Symbol..]
        The variables for the observed interger count of the three classes
        ('a', 'b', 'c') in the test.

        Rai : number of 'a' responses
        Rbi : number of 'b' responses
        Rci : number of 'c' responses

    correctness_variables: List[simpy.Symbol..]
        The variables associated with correct and wrong responses given
        the true label.

        The variables associated with true label 'a'
        Raia : number of 'a' responses (also number of correct)
        Rbia : number of 'b' responses
        Rcia : number of 'c' responses

        The variables associated with true label 'b'
        Raib : number of 'a' responses
        Rbib : number of 'b' responses  (also number of correct)
        Rcib : number of 'c' responses

        The variables associated with true label 'c'
        Raic : number of 'a' responses
        Rbic : number of 'b' responses
        Rcic : number of 'c' responses (also number of correct)


    Methods
    -------
    evaluate_axioms(eval_dict): List[expression, expression, expression]
        Evaluates the axioms given the variable substitutions in 'eval_dict'.
    satisfies_axioms(eval_dict): Boolean
        Checks if the variable substitutions in 'eval_dict' make all three
        axioms identically zero.
    """

    def __init__(self, labels, classifier):
        "Constructs variables for 'labels' and the axioms they must satisfy."

        self.labels = labels
        self.classifier = classifier

        vars = SingleClassifierVariables(labels, classifier)
        self.questions_number = vars.questions_number
        self.responses = vars.responses
        self.responses_by_label = vars.responses_by_label

        # Construct the three, dependent axioms for a single classifier
        self.algebraic_expressions = {}
        for i_true in range(len(labels)):

            true_label = labels[i_true]
            true_label_responses = self.responses_by_label[true_label]
            q_number = self.questions_number[true_label]

            mistakes_out_of_label = q_number
            mistakes_into_label = 0
            for i_response in range(len(labels)):
                response_label = labels[i_response]
                if i_response != i_true:
                    mistakes_out_of_label -= true_label_responses[
                        response_label
                    ]
                    mistakes_into_label += self.responses_by_label[
                        response_label
                    ][true_label]

            self.algebraic_expressions[true_label] = (
                mistakes_out_of_label
                + mistakes_into_label
                - self.responses[true_label]
            )

    def evaluate_axioms(self, eval_dict):
        """
        Evaluates axioms given 'eval_dict'.

        Parameters
        ----------
        eval_dict: Map[sympySymbol -> value]

        Returns
        -------
        Dict mapping label to axiom expression.
        """

        return {
            label: axiom.subs(eval_dict)
            for label, axiom in self.algebraic_expressions.items()
        }

    def satisfies_axioms(self, eval_dict):
        """
        Tests axioms are satisfied for 'eval_dict' values.

        Parameters
        ----------
        eval_dict: Map[sympySymbol -> value]

        Returns
        -------
        Boolean: returns True if the axioms are identically zero, False
        otherwise.
        """

        evaluated_axioms = self.evaluate_axioms(eval_dict)
        return all([(axiom == 0) for axiom in evaluated_axioms.values()])

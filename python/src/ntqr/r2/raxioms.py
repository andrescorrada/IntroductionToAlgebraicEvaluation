"""R=2 evaluation axioms in integer space.

This is the space of integers possible since we are dealing with finite
tests. Statistics of correctness are given as integers. In the case of
R=2 the evaluation of a single classifier/responder is (Q_a, R_a_a, R_b_b).
"""

import sympy
from ntqr.statistics import SingleClassifierVariables


class SingleClassifierAxioms:
    """
    The axioms of a single R=2 classifier

    ...

    Attributes
    ----------

    question_numbers : List[simpy.Symbol..]
        The variables for the count of correct questions for each true
        label.

    responses : List[simpy.Symbol..]
        The variables for the observed interger count of the two classes.
        Generically of the form,

        R_{l_i} : number of 'l' responses by classifier 'i'

    correctness_variables: List[simpy.Symbol..]
        The variables associated with correct and wrong responses given
        the true label. Generically of the form,

        R_{l_i, l_true}: number of 'l' responses by classifier 'i' given
        true label 'l_true'.


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

        # Construct the dependent axioms for a single classifier.
        # These are always equal to the number of labels.
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


## The following code needs to be refactored into classes for
## pair and triplet axioms
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

# pair_binary_responders_axiom = [
#     (qb - (rai + raj) + raiaj) + ((raia + raja) - (raiaja + rbibjb)),
#     (qa - (rbi + rbj) + rbibj) + ((rbib + rbjb) - (raiaja + rbibjb)),
# ]
three_responders_axiom = []

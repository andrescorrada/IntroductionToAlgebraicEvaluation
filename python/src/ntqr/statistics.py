"""
Module with classes that encapsulate the production of statistical variables
for evaluation models.
"""

import sympy


class SingleClassifierVariables:
    """Statistical variables associated with a single classifier.

    Attributes
    ----------

    question_numbers : List[simpy.Symbol..]
        The variables for the count of correct questions for each true
        label.

    responses : List[simpy.Symbol..]
        The variables for the observed interger count of labels in
        the test. Generally, of the form

        R_{l, i} : Number of 'l' label responses by classifier 'i'

    correctness_variables: List[simpy.Symbol..]
        The variables associated with correct and wrong responses given
        the true label.

        R_{l_r_{i}, l_true} : Number of l_r responses by classifier 'i'
        given true label, l_true.
    """

    def __init__(self, labels, classifier):
        "Constructs variables for 'labels'."

        self.questions_number = {
            label: sympy.Symbol(r"Q_" + label) for label in labels
        }
        self.responses = self.response_variables(labels, classifier)
        self.responses_by_label = self.label_response_variables(
            labels, classifier
        )

    def response_variables(self, labels, classifier):
        """
        Constructs observable response variables given 'labels' and
        'classifier'.

        Parameters
        ----------
        labels : List
            Labels to use.
        classifier : int
            Index of classifier.

        Returns
        -------
        Dictionary of by-label response counts, one per label.
        """

        clsfr_str = str(classifier)
        vars = {}
        for label in labels:
            vars[label] = sympy.Symbol(
                r"R_{" + label + r"_{" + clsfr_str + r"}}"
            )

        return vars

    def label_response_variables(self, labels, classifier):
        """
        Constructs variables associated with correct and wrong
        response counts given true label.

        Parameters
        ----------
        labels : List
            Labels to use.
        classifier : int
            Index of classifier.

        Returns
        -------
        Dictionary of by-label response counts, three per label.
        In addition, each label contains a 'correct' dictionary
        that contains the true_label and its associated count.
        An 'errors' dictionary is indexed by possible wrong
        label assignments.
        """

        clsfr_str = str(classifier)
        vars = {}
        for true_label in labels:
            label_vars = vars.setdefault(true_label, {})
            label_vars_correct = label_vars.setdefault("correct", {})
            label_vars_error = label_vars.setdefault("errors", {})

            for response_label in labels:
                curr_var = sympy.Symbol(
                    r"R_{"
                    + response_label
                    + r"_{"
                    + clsfr_str
                    + r"},"
                    + true_label
                    + r"}"
                )
                label_vars[response_label] = curr_var

                # Now we add it to either 'correct' or 'errors'
                if response_label != true_label:
                    label_vars_error[response_label] = curr_var
                else:
                    label_vars_correct[true_label] = curr_var

        return vars

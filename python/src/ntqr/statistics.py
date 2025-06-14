"""
Module with classes that encapsulate the production of statistical variables
for evaluation models.
"""

from itertools import combinations, product
from types import MappingProxyType
from typing_extensions import Iterable, Literal, Mapping, Sequence, Union

import sympy

import ntqr


class MClassifiersVariables:
    """
    Statistical responses variables for M classifiers.




    Attributes
    ----------

    qs : List[simpy.Symbol..]
        The Q_l_i variables. There are R of them.

    responses : Mapping[m_subset, Mapping[Sequence[Label], simpy.Symbol]]


    responses_by_label : Mapping[m_subset, Mapping[Label, Mapping[...]]]

    """

    def __init__(self, labels: ntqr.Labels, classifiers: Sequence[str]):
        """


        Parameters
        ----------
        labels : Labels
            Labels for the question responses.
        classifiers : Sequence[str]
            Labels for the classifiers.

        Returns
        -------
        None.

        """
        self.labels = labels
        self.classifiers = classifiers

        # The variables for the answer-key Q-simplex.
        self.qs = MappingProxyType(
            {label: sympy.Symbol(r"Q_" + label) for label in labels}
        )

        self._responses = {}
        # The observable responses by the classifiers
        for m in range(1, len(classifiers) + 1):
            for m_subset in combinations(classifiers, m):
                self._responses[m_subset] = self._response_variables(
                    labels, m_subset
                )
        self.responses = MappingProxyType(self._responses)
        del self._responses

        self._label_responses = {}
        # The observable responses by the classifiers
        for m in range(1, m + 1):
            for m_subset in combinations(classifiers, m):
                self._label_responses[m_subset] = (
                    self._label_response_variables(labels, m_subset)
                )
        self.label_responses = MappingProxyType(self._label_responses)
        del self._label_responses

    def _response_variables(self, labels, classifiers):
        """
        Constructs observable response variables given 'labels' and
        'classifiers'.

        Parameters
        ----------
        labels : List
            Labels to use.
        classifiers : Sequence[str]
            Labels to use for the classifiers. The label should support
            being stringified.

        Returns
        -------
        Dictionary of by-label response counts, one per label.
        """

        clsfr_strs = [str(classifier) for classifier in classifiers]
        vars = {}

        # The possible responses by N classifiers outputting R labels
        # a set of size R^N
        vars = {
            decisions: sympy.Symbol(
                r"R_{" + self.seq_str(decisions, clsfr_strs) + r"}"
            )
            for decisions in product(labels, repeat=len(classifiers))
        }
        return vars

    def _label_response_variables(self, labels, classifiers):
        """
        Constructs variables associated with correct and wrong
        response counts given true label.

        Parameters
        ----------
        labels : Sequence[Label]
            Labels to use.
        classifier : Sequence[str]
            Index of classifier.

        Returns
        -------
        Dictionary of by-label response counts, three per label.
        In addition, each label contains a 'correct' key
        that points to the variable associated with correct responses.
        An 'errors' dictionary is indexed by possible wrong
        label assignments.
        """

        vars = {
            true_label: {
                decisions: self.label_r_var_symbol(
                    decisions, classifiers, true_label
                )
                for decisions in product(labels, repeat=len(classifiers))
            }
            for true_label in labels
        }

        # Now we define the all correct variable and the
        # not-all-correct variables
        for true_label in labels:
            all_correct = tuple([true_label for classifier in classifiers])

            # The single variable corresponding to all of the classifiers
            # being correct on the true label.
            vars[true_label]["all_correct"] = self.label_r_var_symbol(
                all_correct,
                classifiers,
                true_label,
            )

            # The variables where at least one classifier is wrong
            vars[true_label]["errors"] = {
                decisions: self.label_r_var_symbol(
                    decisions, classifiers, true_label
                )
                for decisions in product(labels, repeat=len(classifiers))
                if decisions != all_correct
            }

        return vars

    def seq_str(self, decisions, classifiers) -> str:
        """


        Parameters
        ----------
        decisions : Sequence[labels]
            A sequence of N labels.
        clsfr_strs : TYPE
            The N classifier labels.

        Returns
        -------
        Comma separated string of the forms l_c.

        """
        return ",".join(
            [
                label + r"_{" + str(classifier) + r"}"
                for label, classifier in zip(decisions, classifiers)
            ]
        )

    def label_r_var_symbol(self, decisions, classifiers, label):
        return sympy.Symbol(
            r"R_{" + self.seq_str(decisions, classifiers) + r"," + label + r"}"
        )

    def r_var_symbol(self, decisions, classifiers):
        return sympy.Symbol(
            r"R_{" + self.seq_str(decisions, classifiers) + r"}"
        )

    def pair_correlations(
        self, pair: tuple[str, str], decisions: tuple[str, str], l_true: str
    ) -> sympy.UnevaluatedExpr:
        """


        Parameters
        ----------
        pair : tuple[str,str]
            Pair of classifiers.
        decisions : tuple[str,str]
            The decisions by the pair.

        Returns
        -------
        Mapping from label to an expression for the pair correlation
        for the decisions given true label.

        """
        q_l = self.qs[l_true]
        expr = (
            1
            / q_l
            * (
                self.label_responses[pair][l_true][decisions]
                - 1
                / q_l
                * self.label_responses[(pair[0],)][l_true][(decisions[0],)]
                * self.label_responses[(pair[1],)][l_true][(decisions[1],)]
            )
        )
        return expr

    def all_agree_subs_dict(
        self,
    ) -> Mapping[sympy.Symbol, sympy.UnevaluatedExpr]:
        """ """
        subs_dict = {}
        for m_subset, m_subset_dict in self.label_responses.items():
            for l_true in self.labels:
                var = m_subset_dict[l_true][
                    tuple([l_true for i in range(len(m_subset))])
                ]
                subs_dict[var] = self.qs[l_true] - sum(
                    m_subset_dict[l_true]["errors"].values()
                )

        return subs_dict


class SingleClassifierVariables:
    """Statistical variables associated with a single classifier.

    Attributes
    ----------

    question_numbers : List[simpy.Symbol..]
        The variables for the count of correct questions for each true
        label.

    responses : Mapping[label, simpy.Symbol]
        The variables for the observed interger count of labels in
        the test. Generally, of the form

        R_{l_ i} : Number of 'l' label responses by classifier 'i'

    responses_by_label: Mapping[label, simpy.Symbol]
        The variables associated with correct and wrong responses given
        the true label.

        Variables are of the form,
        R_{l_i, l_true} : Number of l responses by classifier 'i'
        given true label, l_true.

        Given true label, responses_by_label[label] returns a dictionary
        with keys:
            'correct': the variable for correct responses given true label.
            'errors': dictionary, indexed by wrong label, to incorrect
            responses.
            'l_1', 'l_2': dictionary, indexed by response label, given
            true_label.
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
        In addition, each label contains a 'correct' key
        that points to the variable associated with correct responses.
        An 'errors' dictionary is indexed by possible wrong
        label assignments.
        """

        clsfr_str = str(classifier)
        vars = {}
        for true_label in labels:
            label_vars = vars.setdefault(true_label, {})
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
                    label_vars["correct"] = curr_var

        return vars

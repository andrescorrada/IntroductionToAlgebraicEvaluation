"""Module for error correlation conditionals."""
import sympy

from ntqr.r2.raxioms import (pair_correlation_axiom_first_type,
                             pair_correlation_axiom_three_classifiers)
from ntqr.r2.raxioms import (q, rai,rbi,raj,rbj,rak,rbk)
from ntqr.r2.raxioms import (raiaj, raiak, rajak, rbibj, rbibk, rbjbk)

class PairConditionals:
    """Conditionals that only involve pair correlations between binary
    classifiers."""

    def __init__(self,trioVoteCounts):
        """
        The pair error correlation axioms are localized by the
        observed responses of the classifiers.

        Returns
        -------
        None

        """
        self.tvc = trioVoteCounts


    def pair_axiom_type_I(self):

        axiom = pair_correlation_axiom_first_type
        evaluation_dict = self.evaluation_dict()

        return axiom.subs(evaluation_dict)

    def pair_correlation_axiom_three_classifiers(self):

        axiom = pair_correlation_axiom_three_classifiers
        evaluation_dict = self.evaluation_dict()

        return axiom.subs(evaluation_dict)

    def observed_responses(self):

        # The label responses
        label_classifier_responses = {
            label:{classifier:self.tvc.classifier_label_responses(classifier, label)
                   for classifier in range(3)}
            for label in ("a","b")}

        label_pair_responses = {
            label:{pair:self.tvc.pair_label_responses(pair,label)
                   for pair in ((0,1),(0,2),(1,2))}
            for label in ("a","b")}

        return {"classifier_responses": label_classifier_responses,
                "pair_responses": label_pair_responses}

    def evaluation_dict(self):
        "Evaluation dict from Symbols to values."
        temp_list = []
        observed_responses = self.observed_responses()
        canonical_variables = self.canonical_variables()
        for label in ("a", "b"):

            # Classifier symbols
            sym_dict = canonical_variables["classifier_responses"][label]
            val_dict = observed_responses["classifier_responses"][label]
            for classifier in range(3):
                _symbol = sym_dict[classifier]
                val = val_dict[classifier]
                temp_list.append((_symbol, val))

            # Pair symbols
            sym_dict = canonical_variables["pair_responses"][label]
            val_dict = observed_responses["pair_responses"][label]
            for pair in ((0,1),(0,2),(1,2)):
                _symbol = sym_dict[pair]
                val = val_dict[pair]
                temp_list.append((_symbol,val))

        temp_list.append( (q, self.tvc.test_size))

        return {_sym:val for (_sym,val) in temp_list}

    def canonical_variables(self):

        classifier_responses = {"a":{0:rai, 1:raj, 2:rak},
                                "b":{0:rbi, 1:rbj, 2:rbk}}

        pair_responses = {"a":{(0,1):raiaj, (0,2):raiak, (1,2):rajak},
                          "b":{(0,1):rbibj, (0,2):rbibk, (1,2):rbjbk}}

        return {"classifier_responses":classifier_responses,
                "pair_responses":pair_responses}




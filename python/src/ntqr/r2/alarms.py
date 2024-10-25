"""Algorithms for logical alarms based on the R=2 axioms.

Formal verification of unsupervised evaluations is carried out by
using the agreements and disagreements between classifiers to detect
if they are misaligned given a safety specification.
"""

import ntqr.r2.evaluations


class SingleClassifierAxiomAlarm:
    """Alarm based on the single classifier axioms for the ensemble members

    Initialize with 'classifiers_axioms', list of
    ntqr.r2.raxioms.SingleClassifierAxioms, one per classifier we want
    to compare.
    """

    def __init__(self, Q, classifiers_axioms):
        """"""

        self.Q = Q

        self.classifiers_axioms = classifiers_axioms
        self.labels = classifiers_axioms[0].labels

    def generate_safety_specification(self, factors):
        """Creates a safety specification function query given min label
        accuracies."""

        def satisfies_safety_specification(qs, correct_responses):
            """Checks that list of label accuracies satisfy the
            equation, factor*correct_response - ql > 0."""
            tests = [
                (factor * correct_response - ql) > 0
                for factor, correct_response, ql in zip(
                    factors, correct_responses, qs
                )
            ]
            print("tests for safety specification: ", tests, " ", all(tests))
            return all(tests)

        return satisfies_safety_specification

    def misaligned_at_qs(self, safety_spec, qs, responses):
        """Boolean test to see if the classifiers violate the safety
        specification at given questions correct number."""

        assert self.check_responses(qs, responses)

        sca_evals = [
            ntqr.r2.evaluations.SingleClassifierEvaluations(self.Q, axioms)
            for axioms in self.classifiers_axioms
        ]
        max_corrects = [
            sca_eval.max_correct_at_qa_qb(qs, cresponses)
            for sca_eval, cresponses in zip(sca_evals, responses)
        ]

        return all(
            [
                (not safety_spec(qs, max_correct))
                for max_correct in max_corrects
            ]
        )

    def check_responses(self, qs, responses):
        "Checks logical constraints on responses."
        q = sum(qs)
        qs_check = q == self.Q
        responses_check = all(
            [
                q == sum(classifier_responses)
                for classifier_responses in responses
            ]
        )

        return all([qs_check, responses_check])

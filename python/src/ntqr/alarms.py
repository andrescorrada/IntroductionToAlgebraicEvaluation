"""Algorithms for logical alarms based on the axioms.

Formal verification of unsupervised evaluations is carried out by
using the agreements and disagreements between classifiers to detect
if they are misaligned given a safety specification.
"""


class SingleClassifierAxiomAlarm:
    """Alarm based on the single classifier axioms for the ensemble members

    Initialize with 'classifiers_axioms', list of
    ntqr.r3.raxioms.SingleClassifierAxioms, one per classifier we want
    to compare.
    """

    def __init__(self, Q, classifiers_axioms, cls_single_evals):
        """"""

        self.Q = Q

        self.classifiers_axioms = classifiers_axioms
        self.labels = classifiers_axioms[0].labels

        self.evals = [
            cls_single_evals(self.Q, axioms)
            for axioms in self.classifiers_axioms
        ]

    def generate_safety_specification(self, factors):
        """Creates a safety specification function query given min label
        accuracies."""

        def satisfies_safety_specification(qs, correct_responses):
            """Checks that list of label accuracies satisfy the
            equation, factor*correct_response - ql > 0."""
            tests = [
                (factor * correct_response - ql) > 0 if ql > 0 else True
                for factor, correct_response, ql in zip(
                    factors, correct_responses, qs
                )
            ]
            # print("tests for safety specification: ", tests, " ", all(tests))
            return all(tests)

        return satisfies_safety_specification

    def misaligned_at_qs(self, safety_spec, qs, responses):
        """Boolean test to see if the classifiers violate the safety
        specification at given questions correct number."""

        assert self.check_responses(qs, responses)

        max_corrects = [
            sca_eval.max_correct_at_qs(qs, cresponses)
            for sca_eval, cresponses in zip(self.evals, responses)
        ]

        return any(
            [
                (not safety_spec(qs, max_correct))
                for max_correct in max_corrects
            ]
        )

    def misalignment_trace(self, safety_spec, responses):
        "Boolean trace of the classifiers test alignment."
        all_qs = self.evals[0].all_qs()
        trace = set(
            [
                (qs, self.misaligned_at_qs(safety_spec, qs, responses))
                for qs in all_qs
            ]
        )
        return trace

    def are_misaligned(self, safety_spec, responses):
        "Boolean test of misaligned"
        trace = self.misalignment_trace(safety_spec, responses)
        return all(list(zip(*trace))[1])

    def check_responses(self, qs, responses):
        "Checks logical constraints on responses."
        q = sum(qs)
        if q == self.Q:
            qs_check = True
        else:
            qs_check = False
        responses_check = all(
            [
                (q == sum(classifier_responses))
                for classifier_responses in responses
            ]
        )

        consistency_check = all([qs_check, responses_check])
        if consistency_check == False:
            print("q ", q, "Q ", self.Q),
            print("qs ", qs)
            print("responses ", responses)
            print("checks ", [qs_check, responses_check])

        return all([qs_check, responses_check])

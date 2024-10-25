"""Algorithms for logical alarms based on the R=2 axioms.

Formal verification of unsupervised evaluations is carried out by
using the agreements and disagreements between classifiers to detect
if they are misaligned given a safety specification.
"""


class SingleClassifierAxiomAlarm:
    """Alarm based on the single classifier axioms for the ensemble members

    Initialize with 'classifiers_axioms', list of
    ntqr.r2.raxioms.SingleClassifierAxioms, one per classifier we want
    to compare.
    """

    def __init__(self, classifiers_axioms):
        """"""

        self.classifiers_axioms = classifiers_axioms
        self.labels = classifiers_axioms[0].labels

    def generate_safety_specification(self, factors):
        """Creates a safety specification function query given min label
        accuracies."""

        def satisfies_safety_specification(qs, correct_responses):
            """Checks that list of label accuracies satisfy the
            equation, factor*correct_response - ql > 0."""
            tests = [
                factor * correct_response - ql > 0
                for factor, correct_response, ql in zip(
                    factors, correct_responses, qs
                )
            ]
            print(tests)
            return all(tests)

        return satisfies_safety_specification

"""
Evaluations for binary tests (R=2).

For any finite test there is a finite set of evaluations possible. The
classes in this module compute them.

Classes:
    APrioriSingleEvaluations - Before we see any test results, we can construct
    all possible evaluations given the size of the test.
    PosterioriSingleEvaluations - Once we have test results, we can restrict
    the full set to the evaluations that are consistent with them.

Functions:

Misc variables:

"""
from ntqr.r2.datasketches import VoteCounts


class APrioriSingleEvaluations:
    """All possible evaluations for a test of size Q."""

    def __init__(self, Q: int):
        self.Q = Q

    def all_possible_evaluations(self):
        """
        Calculates all the possible evaluations for a binary response test
        with Q questions.

        Returns
        -------
        Mapping[int, List[Tuple[int, int]]]
            Mapping from Q_a to all possible [R_a_a, R_b_b]

        """
        return {}


class PosteriorSingleEvaluations:
    """Evaluations logically consistent with observed responses."""

    def __init__(self, tvc: VoteCounts):
        self.Q = tvc.test_size
        self.tvc = tvc

    def all_possible_evaluations(self, classifier: int):
        # The b label frequency for the classifier
        Rb = int(self.tvc.classifier_label_frequency(classifier, "b") * self.Q)
        evaluations = {}
        for Qa in range(0, self.Q + 1):
            evaluations[Qa] = []
            for Raa in range(0, Qa + 1):
                for Rbb in range(0, self.Q - Qa + 1):
                    if Raa - Rbb - Qa + Rb == 0:
                        evaluations[Qa].append((Raa, Rbb))

        return evaluations

    def find_k_nearest_at_prevalence(self, classifier: int, point, k: int):
        # The b label frequency for the classifier
        Rb = int(self.tvc.classifier_label_frequency(classifier, "b") * self.Q)

        evaluations = {}
        for Qa in range(0, self.Q + 1):
            qa_candidates = []
            for Raa in range(0, Qa + 1):
                for Rbb in range(0, self.Q - Qa + 1):
                    if Raa - Rbb - Qa + Rb == 0:
                        distances = self.distances_to_target(
                            Qa, Raa, Rbb, point
                        )
                        qa_candidates.append((distances, (Raa, Rbb)))
            qa_candidates.sort()
            evaluations[Qa] = qa_candidates[:k]

        return evaluations

    def distances_to_target(Qa, Raa, Rbb, point):
        return (
            (Qa - point[0]) ** 2,
            (Raa - point[1]) ** 2 + (Rbb - point[2]) ** 2,
        )

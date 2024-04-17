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
import sympy

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

    def find_k_nearest_at_prevalence(
        self, classifier: int, qa: int, point, k: int
    ):
        # The b label frequency for the classifier
        Rb = int(self.tvc.classifier_label_frequency(classifier, "b") * self.Q)

        evaluations = {}
        min_qa = int(max(point[0] - 1000, 0))
        max_qa = int(min(point[0] + 1000, self.Q))
        Qa = qa
        qa_candidates = []
        for Raa in range(0, Qa + 1):
            for Rbb in range(0, self.Q - Qa + 1):
                if Raa - Rbb - Qa + Rb == 0:
                    distances = self.distances_to_target(Qa, Raa, Rbb, point)
                    qa_candidates.append((distances, (Raa, Rbb)))
        qa_candidates.sort()
        evaluations = qa_candidates[:k]

        return evaluations

    def distances_to_target(self, Qa, Raa, Rbb, point):
        # point_pspace = self.to_pspace(point)
        pe_pspace = self.to_pspace((Qa, Raa, Rbb))

        return (
            (point[0] - pe_pspace[0]) ** 2,
            (point[1] - pe_pspace[1]) ** 2 + (point[2] - pe_pspace[2]) ** 2,
        )

    def to_pspace(self, point):
        if point[0] != 0:
            pia = sympy.Rational(point[1], point[0])
        else:
            pia = sympy.Rational(0, 1)

        if self.Q - point[0] != 0:
            pib = sympy.Rational(point[2], (self.Q - point[0]))
        else:
            pib = sympy.Rational(0, 1)

        return (sympy.Rational(point[0], self.Q), pia, pib)

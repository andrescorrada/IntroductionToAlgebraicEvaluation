"""@author: Andrés Corrada-Emmanuel."""

from itertools import combinations, product
from types import MappingProxyType
from typing_extensions import Iterable, Literal, Mapping, Sequence, Union

from ntqr import Labels
from ntqr.statistics import MClassifiersVariables


class MAxiomIdeals:
    """
    Class for generating the M=m axioms given labels and classifiers.

    Each subset of the classifiers has a set of axioms, of size R,
    involving the marginalized decisions of that subset. For each
    value of the size of a subset, M, we can establish universal
    relations between counts of observed M-sized decision tuples
    and the counts of those tuples and smaller ones given true labels.

    The bottom of the M axioms ladder is M=1, the individual axioms.
    These carve out the evaluations for a given test taker given their
    marginalized response counts. This is a subset of the R-simplex,
    for an individual test taker, for any test of size Q. Thus, each
    test-taker in a group of N gets their own M=1 set of axioms.

    The M=2 axioms correspond to all possible pairs in a group of N
    test takers. These involve all the variables in the  M=1 axioms
    but now define a new set of axioms involving the pair response
    variables.

    In R-space, the space of integer response counts, all these
    ideals are linear equations. Thus, it may seem that calling them
    'ideals', while strictly true, is overkill. However, in P-space,
    these ideals are polynomials of degree 2 or higher.
    """

    def __init__(
        self,
        labels: Labels,
        classifiers: Sequence[str],
        m: int = 1,
    ):
        """
        The M=m axioms given answer labels and classifier labels.

        Parameters
        ----------
        labels : Labels
            The R possible label answers to the Q questions.
        classifiers : Sequence[str]
            Labels for indexing classifiers.
        m : int, optional
            M=m axioms index. The default is 1.

        Returns
        -------
        None.

        """

        # There is a an axiomatic ideal for every subset of the
        # classifiers of size m.
        self.m_complex = {}
        for subset in combinations(classifiers, m):
            subset_m_ideal = self.m_complex.setdefault(subset, {})
            vars = subset_m_ideal["mvars"] = MClassifiersVariables(
                labels, subset
            )

            match m:
                case 1:
                    axiomatic_ideal = self._m_one_ideal(labels, vars)
                case _:
                    raise ValueError(
                        "Only M=1 axiom ideals are currently supported."
                    )

            subset_m_ideal["axioms"] = axiomatic_ideal

    def _m_one_ideal(self, labels, vars):
        """


        Parameters
        ----------
        labels : TYPE
            DESCRIPTION.
        vars : TYPE
            DESCRIPTION.

        Returns
        -------
        axioms_by_label : TYPE
            DESCRIPTION.

        """
        qs = vars.qs
        responses = vars.responses
        responses_by_label = vars.responses_by_label

        axioms_by_label = {
            l_true: (
                responses[(l_true,)]
                - qs[l_true]
                + sum(responses_by_label[l_true]["errors"].values())
                - sum(
                    [
                        label_responses["errors"][(l_true,)]
                        for label, label_responses in responses_by_label.items()
                        if label != l_true
                    ]
                )
            )
            for l_true in labels
        }

        return axioms_by_label


class NAxiomsComplex:
    """
    Class that recursively builds all the M-axiom ideals for all non-empty
    subsets of N test-takers answering Q questions with R choices.
    """

    def __init__(self):
        pass

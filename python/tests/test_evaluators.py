# -*- coding: utf-8 -*-
from fractions import Fraction

from ntqr.r2.evaluators import (
    ErrorIndependentEvaluation,
    SupervisedEvaluation,
    MajorityVotingEvaluation,
)
from ntqr.r2.datasketches import TrioLabelVoteCounts
from ntqr.r2.examples import uciadult_label_counts


# Supervised evaluation tests


def test_supervisedevaluation():
    tlvc = TrioLabelVoteCounts(uciadult_label_counts)
    se = SupervisedEvaluation(tlvc)

    assert isinstance(se, SupervisedEvaluation)


def test_supervised_prevalences():
    supervised_eval = SupervisedEvaluation(tlvc)

    assert supervised_eval.evaluation_exact == {
        "accuracies": [
            {"a": Fraction(3737, 5687), "b": Fraction(6501, 10385)},
            {"a": Fraction(1260, 5687), "b": Fraction(29744, 31155)},
            {"a": Fraction(4746, 5687), "b": Fraction(4168, 6231)},
        ],
        "pair_correlations": {
            "a": {
                (0, 1): Fraction(273192, 32341969),
                (0, 2): Fraction(13325, 32341969),
                (1, 2): Fraction(-264525, 32341969),
            },
            "b": {
                (0, 1): Fraction(2204576, 323544675),
                (0, 2): Fraction(-79682, 12941787),
                (1, 2): Fraction(94508, 38825361),
            },
        },
        "prevalence": {
            "a": Fraction(5687, 36842),
            "b": Fraction(31155, 36842),
        },
    }


tlvc = TrioLabelVoteCounts(uciadult_label_counts)
tvc = tlvc.to_TrioVoteCounts()


# Error independent evaluation tests


def test_errorindependentevaluation():
    eieval = ErrorIndependentEvaluation(tvc)

    assert isinstance(eieval, ErrorIndependentEvaluation)


# Majority voting evaluation tests


def test_majorityevaluation():
    mv_eval = MajorityVotingEvaluation(tvc)

    assert isinstance(mv_eval, MajorityVotingEvaluation)


def test_mv_evaluation():
    mv_eval = MajorityVotingEvaluation(tvc)

    assert mv_eval.evaluation_exact == {
        "accuracies": [
            {"a": Fraction(7417, 36842), "b": Fraction(1607, 2834)},
            {"a": Fraction(1089, 18421), "b": Fraction(14185, 18421)},
            {"a": Fraction(7349, 36842), "b": Fraction(1627, 2834)},
        ],
        "prevalence": {
            "a": Fraction(7979, 36842),
            "b": Fraction(28863, 36842),
        },
    }

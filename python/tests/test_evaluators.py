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

    assert supervised_eval.evaluation == {
        "accuracies": [
            {"a": Fraction(3737, 5687), "b": Fraction(6501, 10385)},
            {"a": Fraction(1260, 5687), "b": Fraction(29744, 31155)},
            {"a": Fraction(4746, 5687), "b": Fraction(4168, 6231)},
        ],
        "pair_correlations": {
            "a": {
                (0, 1): Fraction(-4011945936, 183928777703),
                (0, 2): Fraction(23132321875, 183928777703),
                (1, 2): Fraction(356689875, 3913378249),
            },
            "b": {
                (0, 1): Fraction(678533304416, 3360011449875),
                (0, 2): Fraction(6976198814, 134400457995),
                (1, 2): Fraction(2200528244, 241920824391),
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

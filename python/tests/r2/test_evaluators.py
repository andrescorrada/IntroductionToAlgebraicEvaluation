# -*- coding: utf-8 -*-
import sympy

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
        "prevalence": {
            "a": sympy.Rational(5687, 36842),
            "b": sympy.Rational(31155, 36842),
        },
        "accuracy": [
            {
                "a": sympy.Rational(3737, 5687),
                "b": sympy.Rational(6501, 10385),
            },
            {
                "a": sympy.Rational(1260, 5687),
                "b": sympy.Rational(29744, 31155),
            },
            {"a": sympy.Rational(4746, 5687), "b": sympy.Rational(4168, 6231)},
        ],
        "pair_correlation": {
            (0, 1): {
                "a": sympy.Rational(273192, 32341969),
                "b": sympy.Rational(2204576, 323544675),
            },
            (0, 2): {
                "a": sympy.Rational(13325, 32341969),
                "b": sympy.Rational(-79682, 12941787),
            },
            (1, 2): {
                "a": sympy.Rational(-264525, 32341969),
                "b": sympy.Rational(94508, 38825361),
            },
        },
        "3_way_correlation": {
            (0, 1, 2): {
                "a": sympy.Rational(452568508, 183928777703),
                "b": sympy.Rational(-27265589, 134400457995),
            }
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

    assert mv_eval.evaluation_exact == [
        {
            "prevalence": {
                "a": sympy.Rational(7979, 36842),
                "b": sympy.Rational(28863, 36842),
            },
            "accuracy": [
                {
                    "a": sympy.Rational(7417, 7979),
                    "b": sympy.Rational(20891, 28863),
                },
                {
                    "a": sympy.Rational(2178, 7979),
                    "b": sympy.Rational(28370, 28863),
                },
                {
                    "a": sympy.Rational(7349, 7979),
                    "b": sympy.Rational(21151, 28863),
                },
            ],
        },
        {
            "prevalence": {
                "a": sympy.Rational(28863, 36842),
                "b": sympy.Rational(7979, 36842),
            },
            "accuracy": [
                {
                    "a": sympy.Rational(7972, 28863),
                    "b": sympy.Rational(562, 7979),
                },
                {
                    "a": sympy.Rational(493, 28863),
                    "b": sympy.Rational(5801, 7979),
                },
                {
                    "a": sympy.Rational(7712, 28863),
                    "b": sympy.Rational(630, 7979),
                },
            ],
        },
    ]

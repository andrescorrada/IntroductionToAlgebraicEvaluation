"""Functions for algebraic manipulation of axiom expressions.

@author: Andrés Corrada-Emmanuel."""

from collections.abc import Sequence

import sympy


def extract_coefficents(
    expr: sympy.UnevaluatedExpr, vars: Sequence[sympy.Symbol]
) -> Sequence[int]:
    """
    Extract vars coefficients from expr.

    Parameters
    ----------
    expr : sympy.UnevaluatedExpr
        DESCRIPTION.
    vars : Sequence[sympy.Symbol]
        DESCRIPTION.

    Returns
    -------
    Sequence[int]
        DESCRIPTION.

    """
    coeff_dict = expr.as_coefficients_dict()
    zero = sympy.core.numbers.Zero()
    var_coeffs = tuple(int(sympy.N(coeff_dict.get(var, zero))) for var in vars)

    return var_coeffs


def extract_constant(expr: sympy.UnevaluatedExpr) -> int:
    """


    Parameters
    ----------
    expr : sympy.UnevaluatedExpr
        DESCRIPTION.

    Returns
    -------
    int
        DESCRIPTION.

    """
    one = sympy.core.numbers.One()
    const = int(expr.as_coefficients_dict().get(one, 0))
    return const

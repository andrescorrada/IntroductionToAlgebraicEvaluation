"""Functions for algebraic manipulation of axiom expressions.

@author: Andrés Corrada-Emmanuel."""

from collections.abc import Sequence

import sympy


def extract_coefficents(
    expr: sympy.UnevaluatedExpr, vars: Sequence[sympy.Symbol]
) -> Sequence[int]:
    """
    Extract vars coefficients from expr.

    Assumes that 'expr' is a linear equation.

    Parameters
    ----------
    expr : sympy.UnevaluatedExpr
        Linear expression.
    vars : Sequence[sympy.Symbol]
        Variables for which we have their coefficient in expr.

    Returns
    -------
    Sequence[int]
        The integer coefficients, following the same order as 'vars'.

    """
    coeff_dict = expr.as_coefficients_dict()
    zero = sympy.core.numbers.Zero()
    var_coeffs = tuple(int(sympy.N(coeff_dict.get(var, zero))) for var in vars)

    return var_coeffs


def extract_constant(expr: sympy.UnevaluatedExpr) -> int:
    """
    Extracts the constant term in 'expr.'

    Returns the value of the constant term in expr.

    Parameters
    ----------
    expr : sympy.UnevaluatedExpr
        An algebraic expression.

    Returns
    -------
    int
        The integer value of the constant term.

    """
    one = sympy.core.numbers.One()
    const = int(expr.as_coefficients_dict().get(one, 0))
    return const

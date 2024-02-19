import math
import warnings
from typing import Tuple, Optional

import numpy as np
import sympy as sp
from numpy import sqrt, e as E, exp, sin, cos, log, inf, pi, tan, cosh, sinh, tanh, nan, seterr, arcsin, arctan
from scipy.optimize import minimize

from model.config import Config
from model.expr_utils.utils import time_limit, FinishException


def process_symbol_with_C(symbols: str, c: np.ndarray) -> str:
    """
    Replacing parameter placeholders with real parameters
    :param symbols: expressions
    :param c: parameter
    :return: Converted expression

    >>>process_symbol_with_C('C1*X1+C2*X2',np.array([2.1,3.3])) -> '2.1*X1+3.3*X2'

    """
    for idx, val in enumerate(c):
        symbols = symbols.replace(f"C{idx + 1}", str(val))
    return symbols


def prune_poly_c(eq: str) -> str:
    """
    Reducing multiple parameters in a parameterized expression to a single parameter
    :param eq: expression string
    :return: the modified expression
    >>> prune_poly_c('C*C+C+X1*C')->"C+C*X1"
    """
    for i in range(5):
        eq_l = eq
        c_poly = ['C**' + str(i) + ".5" for i in range(1, 4)]
        c_poly += ['C**' + str(i) + ".25" for i in range(1, 4)]
        c_poly += ['C**' + str(i) for i in range(1, 4)]
        c_poly += [' ' + str(i) + "*C" for i in range(1, 4)]
        for c in c_poly:
            if c in eq:
                eq = eq.replace(c, 'C')
        for _ in range(5):
            for _ in range(5):
                eq = eq.replace('arcsin(C)', 'C')
                eq = eq.replace('arccos(C)', 'C')
                eq = eq.replace('sin(C)', 'C')
                eq = eq.replace('cos(C)', 'C')
                eq = eq.replace('sqrt(C)', 'C')
                eq = eq.replace('log(C)', 'C')
                eq = eq.replace('exp(C)', 'C')
                eq = eq.replace('C*C', 'C')
            eq = str(sp.sympify(eq))
        if eq == eq_l:
            break
    return eq


def cal_expression_single(symbols: str, x: np.ndarray, t: np.ndarray, c: Optional[np.ndarray]) -> float:
    """
    Calculate the value of an expression with  `once` and compute the error rmse
    :param symbols: target expressions
    :param x: independent variable
    :param t: result or dependent variable
    :param c: parameter or None if there is no paramter
    :return: RMSE of function or 1e999 if error occurs
    """
    from numpy import inf, seterr
    zoo = inf
    seterr(all="ignore")
    I = complex(0, 1)
    for idx, val in enumerate(x):
        locals()[f'X{idx + 1}'] = val
    with warnings.catch_warnings(record=False) as caught_warnings:
        try:
            if c:
                target = process_symbol_with_C(symbols, c)
            else:
                target = symbols
            cal = eval(target)
            ans = float(np.linalg.norm(cal - t, 1) ** 2 / t.shape[0])  # calculate RMSE
            if math.isinf(ans) or math.isnan(ans) or caught_warnings:
                return 1e999
        except OverflowError:  # if error occurs, return 1e999 as high error.
            return 1e999
        except ValueError:
            return 1e999
        except NameError as e:
            return 1e999
        except ArithmeticError:
            return 1e999
    return ans


def replace_parameter_and_calculate(symbols: str, x: np.ndarray, t: np.ndarray, config_s: Config) -> Tuple[float, str]:
    """
    Calculate the value of the expression, the process is as follows
    1. Determine whether the parameter is included or not, if not, calculate directly
    2. Replace the parameter C in the expression with C1,C2.... CN
    3. Initialize the parameters
    4. Optimize the parameters if config_s.const_optimize, calculate the best parameters
    5. Fill the expression with the best parameters and return the expression with RMSE

    :param symbols: target expression, contains parameter C
    :param x: independent variable
    :param t: result or dependent variable
    :param config_s: config file, used to determine whether to optimize parameters or not
    :return: error, the expression containing the best parameters
    """
    symbols = str(sp.sympify(symbols))
    if symbols.count('zoo') or symbols.count('nan'):
        return 1e999, symbols
    c_len = symbols.count('C')
    if c_len == 0:
        return cal_expression_single(symbols, x, t, None), symbols
    symbols = prune_poly_c(symbols)
    c_len = symbols.count('C')
    if c_len == 0:
        return cal_expression_single(symbols, x, t, None), symbols

    symbols = symbols.replace('C', 'PPP')  # replace C with C1,C2...
    for i in range(1, c_len + 1):
        symbols = symbols.replace('PPP', f'C{i}', 1)

    if config_s.const_optimize:  # const optimize
        x0 = np.random.randn(c_len)
        if cal_expression_single(symbols, x, t, x0) > 1e900:
            return 1e999, process_symbol_with_C(symbols, x0)
        x_ans = minimize(lambda c: cal_expression_single(symbols, x, t, c),
                         x0=x0, method='Powell', tol=1e-6, options={'maxiter': 10})
        x0 = x_ans.x
    else:
        x0 = np.ones(c_len)
    val = cal_expression_single(symbols, x, t, x0)
    return val, process_symbol_with_C(symbols, x0)


def cal_expression(symbols: str, config_s: Config, t_limit: int = 1) -> float:
    """
    Calculate the value of the expression in train and test dataset
    :param symbols: target expression, contains parameter C
    :param config_s: config file, used to determine whether to optimize parameters or not and store independent variable
     and result
    :param t_limit: time limit of calculation in because of time in optimization
    :return: sum of the error of expression in train and test dataset
    """
    warnings.filterwarnings('ignore')
    config_s.symbol_tol_num += 1
    try:
        with time_limit(t_limit):
            v, s = replace_parameter_and_calculate(symbols, config_s.x, config_s.t, config_s)
            if v > config_s.best_exp[1]:
                return v
            v_, s_ = replace_parameter_and_calculate(s, config_s.x_, config_s.t_, config_s)
            if config_s.best_exp[1] > 1e-10 + v_ + v:
                config_s.best_exp = s_, v_ + v
            if v_ + v <= config_s.reward_end_threshold:
                raise FinishException
            return v
    except TimeoutError:
        pass
    except RuntimeError:
        pass
    return 1e999

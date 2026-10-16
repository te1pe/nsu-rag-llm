"""`python_calc`: arithmetic for the agent, without `eval`.

A 9B model multiplies badly, and questions from the calculus and probability
courses sometimes end in a number. The model may therefore hand an expression
to this tool - and an expression from a model is untrusted input: `eval`
would run `__import__("os").system(...)` as happily as `2 + 2`.

So the expression is parsed into a syntax tree and walked here, node by node,
and only these are allowed:

* numbers, `+ - * / // % **`, unary `+ -`, parentheses;
* the constants `pi`, `e`, `inf`;
* the functions in FUNCTIONS, called by name.

Anything else - a name, an attribute, a subscript, a string, a lambda, a
comprehension - is refused before any of it runs. Exponents and factorials
are bounded, because `9 ** 9 ** 9` is arithmetic too, and would keep a core
busy for the rest of the day.
"""

from __future__ import annotations

import ast
import math
import operator

MAX_EXPONENT = 1000
MAX_FACTORIAL = 1000
MAX_LENGTH = 500
#: The largest integer result, in bits: some 4 200 decimal digits, just under
#: the 4 300 that Python will convert to a string at all.
MAX_BITS = 14_000

BINARY = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}
CONSTANTS = {"pi": math.pi, "e": math.e, "inf": math.inf}
COMBINATORIAL = {"factorial", "comb", "perm"}
FUNCTIONS = {
    "sqrt": math.sqrt,
    "exp": math.exp,
    "log": math.log,
    "log2": math.log2,
    "log10": math.log10,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "abs": abs,
    "round": round,
    "floor": math.floor,
    "ceil": math.ceil,
    "factorial": math.factorial,
    "comb": math.comb,
    "perm": math.perm,
    "gcd": math.gcd,
    "min": min,
    "max": max,
}


class CalcError(ValueError):
    """The expression is not allowed, or has no value."""


def _number(value: object) -> int | float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise CalcError("допустимы только числа")
    return value


def _walk(node: ast.AST) -> int | float:
    match node:
        case ast.Expression(body=body):
            return _walk(body)
        case ast.Constant(value=value):
            return _number(value)
        case ast.Name(id=name) if name in CONSTANTS:
            return CONSTANTS[name]
        case ast.UnaryOp(op=op, operand=operand) if type(op) in UNARY:
            return UNARY[type(op)](_walk(operand))
        case ast.BinOp(left=left, op=op, right=right) if type(op) in BINARY:
            a, b = _walk(left), _walk(right)
            if isinstance(op, ast.Pow):
                if abs(b) > MAX_EXPONENT:
                    raise CalcError(f"показатель степени больше {MAX_EXPONENT}")
                # ((10**1000)**1000)**1000 passes the exponent check every time.
                if isinstance(a, int) and abs(a).bit_length() * abs(b) > MAX_BITS:
                    raise CalcError("результат слишком велик")
            return _bounded(BINARY[type(op)](a, b))
        case ast.Call(func=ast.Name(id=name), args=args, keywords=[]) if name in FUNCTIONS:
            values = [_walk(arg) for arg in args]
            # comb(10**6, 5 * 10**5) has hundreds of thousands of digits.
            if name in COMBINATORIAL and any(abs(v) > MAX_FACTORIAL for v in values):
                raise CalcError(f"аргумент {name} больше {MAX_FACTORIAL}")
            return _bounded(_number(FUNCTIONS[name](*values)))
    raise CalcError(f"недопустимо в выражении: {type(node).__name__}")


def _bounded(value: int | float) -> int | float:
    if isinstance(value, int) and value.bit_length() > MAX_BITS:
        raise CalcError("результат слишком велик")
    return value


def calculate(expression: str) -> int | float:
    """The value of an arithmetic expression, or CalcError saying why not."""
    if len(expression) > MAX_LENGTH:
        raise CalcError(f"выражение длиннее {MAX_LENGTH} символов")
    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except SyntaxError as exc:
        raise CalcError(f"не арифметика: {exc.msg}") from exc
    try:
        return _walk(tree)
    except (ArithmeticError, TypeError, ValueError) as exc:
        if isinstance(exc, CalcError):
            raise
        raise CalcError(f"{type(exc).__name__}: {exc}") from exc


def python_calc(expression: str) -> str:
    """The tool as the model sees it: always a string, an answer or an error."""
    try:
        value = calculate(expression)
    except CalcError as exc:
        return f"Ошибка: {exc}"
    if isinstance(value, float) and value.is_integer() and abs(value) < 1e15:
        value = int(value)
    return str(value)

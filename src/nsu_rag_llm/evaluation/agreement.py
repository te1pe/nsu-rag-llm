from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Agreement:
    n: int
    observed: float
    kappa: float
    both_yes: int
    both_no: int
    only_a: int
    """a says yes, b says no."""
    only_b: int


def agreement(a: Sequence[bool], b: Sequence[bool]) -> Agreement:
    if len(a) != len(b):
        raise ValueError(f"{len(a)} labels against {len(b)}")
    n = len(a)
    both_yes = sum(x and y for x, y in zip(a, b, strict=True))
    both_no = sum(not x and not y for x, y in zip(a, b, strict=True))
    only_a = sum(x and not y for x, y in zip(a, b, strict=True))
    only_b = sum(y and not x for x, y in zip(a, b, strict=True))
    if n == 0:
        return Agreement(0, math.nan, math.nan, 0, 0, 0, 0)
    observed = (both_yes + both_no) / n
    a_yes, b_yes = (both_yes + only_a) / n, (both_yes + only_b) / n
    chance = a_yes * b_yes + (1 - a_yes) * (1 - b_yes)
    kappa = math.nan if chance == 1 else (observed - chance) / (1 - chance)
    return Agreement(n, observed, kappa, both_yes, both_no, only_a, only_b)

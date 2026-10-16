"""How well two raters agree on yes/no labels: raw agreement and Cohen's kappa.

Raw agreement flatters a judge when one answer dominates: if nine answers in
ten are faithful, a judge that always says "faithful" agrees 90% of the time
while knowing nothing. Cohen's kappa subtracts the agreement expected by
chance from the raters' own label frequencies:

    kappa = (p_observed - p_chance) / (1 - p_chance)
    p_chance = P(a says yes) * P(b says yes) + P(a says no) * P(b says no)

1 is perfect agreement, 0 is chance, below 0 is worse than chance. It is
undefined when both raters give one and the same label to everything.
"""

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

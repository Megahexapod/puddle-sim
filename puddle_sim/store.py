"""The household's larder (D-044 step 1). Food is a kind in a state; every (kind, state) has a
spoilage rate; the weather at home speeds it up (Q10); the household eats the most perishable first.

Rates are per tick at 20 degrees. Half-lives, for the feel of it (33 ticks = a day, 12000 = a year):
    grain   0.00006  -> ~1 year unprotected (rats, damp; a granary would cut it)
    nut     0.0004   -> ~7 weeks (chestnuts mould; dried they keep)
    fruit   0.006    -> ~3.5 days (a persimmon)
    mushroom 0.010   -> ~2 days
    fern    0.010    -> ~2 days
    clam    0.020    -> ~1 day
    meat    0.008    -> ~2.6 days (fresh; smoked or salted is step 2)
"""
from __future__ import annotations

from typing import Dict, Tuple

import math

HALF_LIFE_DAYS = {                       # fresh, at 20 degrees (D-049: written in days, converted with the clock)
    "grain": 350.0, "nut": 52.0, "fruit": 3.5, "mushroom": 2.1, "fern": 2.1, "clam": 1.05, "meat": 2.6,
}
DECAY: Dict[str, float] = {}             # per tick; filled by set_clock()


def set_clock(ticks_per_day: float) -> None:
    for k, hl in HALF_LIFE_DAYS.items():
        DECAY[k] = math.log(2.0) / (hl * ticks_per_day)


set_clock(3000.0 / 365.0)
# what the wild kinds and the kills are, as food kinds
WILD_KIND = {"蕨": "fern", "香菇": "mushroom", "草菇": "mushroom", "松茸": "mushroom", "栗": "nut", "柿": "fruit", "河蚌": "clam"}
FRESH = "fresh"


class Store:
    def __init__(self, ticks_per_day: float = None):
        if ticks_per_day:
            set_clock(ticks_per_day)
        self.stock: Dict[Tuple[str, str], float] = {}
        self.eaten = 0.0
        self.spoiled = 0.0
        self.spoiled_by: Dict[str, float] = {}
        self.hungry_ticks = 0            # ticks on less than half a ration
        self.ticks = 0
        self.needed = 0.0
        self.ration_ema = 1.0            # recent ration, 0..1 (D-051: strength follows the last days' meals)

    def add(self, kind: str, amount: float, state: str = FRESH) -> None:
        if amount <= 0:
            return
        key = (kind, state)
        self.stock[key] = self.stock.get(key, 0.0) + amount

    def total(self) -> float:
        return float(sum(self.stock.values()))

    def step(self, temp_c: float, need: float, q10: float, ref_t: float, ema_ticks: float = 80.0) -> None:
        """One tick: rot, then eat `need` units, most perishable first."""
        self.ticks += 1
        self.needed += need
        tf = q10 ** ((temp_c - ref_t) / 10.0)
        for key in list(self.stock):
            kind, state = key
            rate = DECAY.get(kind, 0.005) * tf if state == FRESH else DECAY.get(kind, 0.005) * tf * 0.1
            lost = self.stock[key] * min(rate, 1.0)
            if lost > 0:
                self.stock[key] -= lost
                self.spoiled += lost
                self.spoiled_by[kind] = self.spoiled_by.get(kind, 0.0) + lost
            if self.stock[key] < 1e-6:
                del self.stock[key]
        # eat: highest decay rate first (the fish before the rice)
        left = need
        for key in sorted(self.stock, key=lambda k: -DECAY.get(k[0], 0.005) * (1.0 if k[1] == FRESH else 0.1)):
            if left <= 0:
                break
            take = min(left, self.stock[key])
            self.stock[key] -= take
            left -= take
            self.eaten += take
            if self.stock[key] < 1e-6:
                del self.stock[key]
        if left > 0.5 * need:
            self.hungry_ticks += 1
        a = 1.0 / max(1.0, ema_ticks)
        self.ration_ema = (1 - a) * self.ration_ema + a * (1.0 - left / max(1e-9, need))

    def summary(self) -> dict:
        return dict(eaten=round(self.eaten), spoiled=round(self.spoiled), hungry=self.hungry_ticks,
                    fed=round(self.eaten / max(1e-6, self.needed), 3),          # ration: eaten / needed over the year
                    left=round(self.total()), spoiled_by={k: round(v) for k, v in self.spoiled_by.items()})

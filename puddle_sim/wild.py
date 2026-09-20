"""Wild food (D-037): what the map offers to someone who walks instead of tills.

Nothing here is a population. Every item is a *readback* of state the world already keeps:

  蕨    fern shoots      flora kind FERN (shade + damp + decomposer soil), adults, spring window
  香菇  shade mushroom   deadwood in the slow-rot regime (shade + damp; 茨 D-013's high-yield regime),
                         colonised (wood_age), cool, spring and autumn
  草菇  sun mushroom     deadwood in the fast-rot regime (sun + rain), warm, summer
  松茸  pine mushroom    an old tree on poor dry upland soil, one host in eight, autumn only, cool

So mushrooms cost zero bits per cell (1852k): the log, its moisture and the season are already there.
The only state this module owns is `picked` -- a cooldown after a harvest -- and a tally of what the
map offered each tick (the potential a forager could turn into food; farmer.py's `forage` does that).
"""
from __future__ import annotations

import numpy as np

from .config import Config
from .flora import BARE, FERN, TREE, CHESTNUT, PERSIMMON, MUSSEL

NONE, FERN_SHOOT, SHIITAKE, STRAW, MATSUTAKE, NUT, FRUIT, CLAM = 0, 1, 2, 3, 4, 5, 6, 7
NAMES = {FERN_SHOOT: "蕨", SHIITAKE: "香菇", STRAW: "草菇", MATSUTAKE: "松茸", NUT: "栗", FRUIT: "柿", CLAM: "河蚌"}
# Food per pick, in the one unit everything eats in (D-043 step 6). The unit is rice's: a paddy
# cell (4x4 blocks = 16 m2) at fitness 1 yields base 0.02/tick x 12000 = 240 units a year; a real
# paddy gives ~8 kg rice per 16 m2, so 1 unit = 33 g rice = ~120 kcal. From there, by kcal:
#   chestnut  15 kg nuts/tree x 2000 kcal = 30 Mcal -> 250 units/tree, ~1 tree per 3 cells -> ~80/cell-year -> 3 picks of 25
#   persimmon 50 kg fruit/tree x 700 kcal = 35 Mcal -> ~100/cell-year -> 3 picks of 35 (and it does not keep)
#   mussels   ~2 kg meat per 16 m2 bed x 800 kcal = 1.6 Mcal -> 13, once (the bed is gone)
#   shiitake  one flush ~0.5 kg x 300 kcal -> 1.2 ; straw mushroom 0.8 ; fern shoots 0.3 kg x 300 -> 0.8
#   matsutake 0.1 kg -> 0.5 food. Its worth is not food (D-040: that is the other market).
YIELD = {FERN_SHOOT: 0.8, SHIITAKE: 1.2, STRAW: 0.8, MATSUTAKE: 0.5, NUT: 25.0, FRUIT: 35.0, CLAM: 13.0}
REGROW = {FERN_SHOOT: 37, SHIITAKE: 50, STRAW: 30, MATSUTAKE: 150, NUT: 125, FRUIT: 87, CLAM: 10 ** 6}   # ticks (~4.5/6/3.6/18/15/10.6 days at 8.2 ticks a day, D-049); a dug bed is gone; it comes back only by the flora rule
KEEPS = {NUT: True, FRUIT: False, CLAM: False}   # a nut goes in the store; a persimmon has to be eaten now (forage.py, later)
DESTRUCTIVE = {CLAM}                              # picking removes the plant itself


def _in(season: float, lo: float, hi: float) -> bool:
    return lo <= season < hi


class Wild:
    def __init__(self, cfg: Config, shape: tuple):
        self.cfg = cfg
        H, W = shape
        self.kind = np.zeros((H, W), dtype=np.int8)
        self.picked = np.zeros((H, W), dtype=np.int32)      # tick until which a picked cell offers nothing
        ys, xs = np.mgrid[0:H, 0:W]
        self.host = ((ys * 73856093) ^ (xs * 19349663)) % 8 == 0   # which old trees carry the pine mushroom
        self.offered = {k: 0 for k in NAMES}                # ripe cells this tick
        self.offered_total = {k: 0.0 for k in NAMES}        # food the map offered, summed over ticks (potential, not harvest)
        self.eaten_by_boar = 0                              # mast the boars got first (D-042)

    def step(self, world, tick: int) -> None:
        cfg = self.cfg
        fl, so = world.flora, world.soil
        season = world.climate.season
        T = world.temp
        m = so.m
        land = ~world.water
        kind = np.zeros_like(self.kind)

        # 蕨: adult fern, spring shoots.
        if _in(season, 0.05, 0.22):
            fern = (fl.kind == FERN) & (fl.age >= fl.table[FERN].adult_age)
            kind[fern] = FERN_SHOOT

        # Mushrooms on deadwood: the same regime split soil.py uses for rot (D-013).
        wood = (so.wood > 0.05) & land
        soaked = m >= cfg.wood_soaked
        dry = m <= cfg.wood_dry
        sun = 1.0 - world.cover()
        rotting = wood & ~soaked & ~dry
        colonised = so.wood_age >= 200          # (biological clock: a log is colonised after ~200 ticks)
        if _in(season, 0.02, 0.22) or _in(season, 0.55, 0.80):
            shade = rotting & colonised & (sun < 0.4) & (m >= 0.35) & (T >= 8) & (T <= 22)
            kind[shade & (kind == NONE)] = SHIITAKE
        if _in(season, 0.15, 0.45):
            sunny = rotting & (sun >= 0.4) & (m >= 0.3) & (T >= 20) & (T <= 34)
            kind[sunny & (kind == NONE)] = STRAW

        # 松茸: no log at all. An old tree, poor dry soil, the right host, autumn.
        if _in(season, 0.55, 0.75):
            old = (fl.kind == TREE) & (fl.age >= 2 * fl.table[TREE].adult_age)
            pine = old & (so.h < cfg.h_mid) & (m < 0.35) & (T >= 6) & (T <= 18) & self.host
            kind[pine & (kind == NONE)] = MATSUTAKE

        # 果樹: an adult chestnut in its nut window, an adult persimmon in its fruit window.
        if _in(season, 0.58, 0.74):
            nut = (fl.kind == CHESTNUT) & (fl.age >= fl.table[CHESTNUT].adult_age)
            kind[nut & (kind == NONE)] = NUT
        if _in(season, 0.66, 0.84):
            fr = (fl.kind == PERSIMMON) & (fl.age >= fl.table[PERSIMMON].adult_age)
            kind[fr & (kind == NONE)] = FRUIT

        # 河蚌: an adult bed, any season; digging it removes it.
        clam = (fl.kind == MUSSEL) & (fl.age >= fl.table[MUSSEL].adult_age)
        kind[clam & (kind == NONE)] = CLAM

        kind[self.picked > tick] = NONE
        self.kind = kind
        for k in NAMES:
            n = int((kind == k).sum())
            self.offered[k] = n
            self.offered_total[k] += n * YIELD[k] / (REGROW[k] if k not in DESTRUCTIVE else 2000)   # a cell offers its yield once per regrow window (a bed: once per ~generation)

    # ------------------------------------------------------------------ harvest

    def pick(self, world, y: int, x: int, tick: int) -> float:
        """Take what this cell offers; it offers nothing again until it has regrown."""
        k = int(self.kind[y, x])
        if k == NONE:
            return 0.0
        if k in DESTRUCTIVE:
            world.flora.kind[y, x] = BARE
            world.flora.age[y, x] = 0
            if k == CLAM:
                world.shells += 1
        else:
            self.picked[y, x] = tick + REGROW[k]
        self.kind[y, x] = NONE
        return YIELD[k]

    def counts(self) -> dict:
        return {NAMES[k]: v for k, v in self.offered.items()}

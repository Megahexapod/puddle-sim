"""Industries: the plants humans plant (D-024/D-025).

An industry is a row like a plant's: a fitness map over the world's fields (where it can run and
how well), a build cost paid from the settlement's ledger, inputs it eats per tick, outputs it
makes per tick times fitness, and effects it writes back into the same fields the plants and
creatures read. No special cases: a salt pan salinises its neighbours, a kiln eats the forest,
a water wheel is a weir the fish cannot pass.

Ledger resources: wood, food, salt, charcoal, power.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple

import numpy as np

from .config import Config
from .flora import BARE, TREES
from .gen import ROCK

SALTPAN, KILN, WHEEL = 1, 2, 3


@dataclass
class Industry:
    key: str
    name: str
    glyph: str
    cost: Dict[str, float]
    fitness: Callable                 # (world) -> 0..1 map
    inputs: Dict[str, float] = field(default_factory=dict)    # per tick at fitness 1
    outputs: Dict[str, float] = field(default_factory=dict)   # per tick at fitness 1
    on_water: bool = False            # the site cell itself is water (wheel) or land


def _nbr_max(a):
    p = np.pad(a, 1, mode="edge")
    out = np.full_like(a, -np.inf)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            out = np.maximum(out, p[1 + dy: 1 + dy + a.shape[0], 1 + dx: 1 + dx + a.shape[1]])
    return out


def _window(a, r):
    m = a.astype(np.float32)
    p = np.pad(m, r)
    out = np.zeros_like(m)
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            out += p[r + dy: r + dy + m.shape[0], r + dx: r + dx + m.shape[1]]
    return out


def saltpan_fitness(w) -> np.ndarray:
    """Land next to salty water, flat, sunlit, not rock: salt = sun x salinity."""
    salty = _nbr_max(np.where(w.water, w.salt_ema, 0.0))
    land = ~w.water & (w.substrate != ROCK) & ~w.farm & (w.flora.kind == BARE)
    gy, gx = np.gradient(w.hydro.b)
    slope = np.sqrt(gy ** 2 + gx ** 2)
    flat = np.clip(1.0 - slope / 0.4, 0.0, 1.0)
    sun = np.clip((w.light - 9) / 6.0, 0.0, 1.0)
    return (land * np.clip(salty / 0.6, 0.0, 1.0) * flat * sun).astype(np.float32)


def kiln_fitness(w) -> np.ndarray:
    """Land with forest around: fitness = trees in the 7x7 / 12."""
    land = ~w.water & (w.substrate != ROCK) & ~w.farm & ~w.tree
    return (land * np.clip(_window(w.tree, 3) / 12.0, 0.0, 1.0)).astype(np.float32)


def wheel_fitness(w) -> np.ndarray:
    """A water cell with current and depth, next to land to stand on."""
    hy = w.hydro
    cur = np.clip((hy.speed_ema - 0.5) / 1.0, 0.0, 1.0)
    deep = np.clip(hy.d_ema / 0.3, 0.0, 1.0)
    bank = _nbr_max((~w.water).astype(np.float32)) > 0
    return (w.water * bank * cur * deep).astype(np.float32)


TABLE: Dict[int, Industry] = {
    SALTPAN: Industry("saltpan", "鹽田", "鹽", cost={"wood": 2.0}, fitness=saltpan_fitness,
                      outputs={"salt": 0.02}),
    KILN: Industry("kiln", "木炭窯", "炭", cost={"wood": 3.0}, fitness=kiln_fitness,
                   inputs={"wood": 0.01}, outputs={"charcoal": 0.006}),
    WHEEL: Industry("wheel", "水車", "車", cost={"wood": 4.0}, fitness=wheel_fitness,
                    outputs={"power": 0.02}, on_water=True),
}


class Industries:
    """All built industries on the map, whoever owns them."""

    def __init__(self, cfg: Config, shape: tuple):
        self.cfg = cfg
        self.kind = np.zeros(shape, dtype=np.int8)
        self.owner = np.full(shape, -1, dtype=np.int16)
        self.fit = np.zeros(shape, dtype=np.float32)      # last fitness, for display

    def sites(self) -> List[Tuple[int, int, int, int]]:
        ys, xs = np.nonzero(self.kind)
        return [(int(y), int(x), int(self.kind[y, x]), int(self.owner[y, x])) for y, x in zip(ys, xs)]

    def blocked_water(self) -> np.ndarray:
        """Weirs: creatures cannot pass a wheel cell."""
        return self.kind == WHEEL

    def try_build(self, world, farmer, kind: int, radius: int) -> bool:
        """Build one of `kind` near the farmer's home if the ledger can pay and a site fits."""
        ind = TABLE[kind]
        for res, amt in ind.cost.items():
            if farmer.ledger.get(res, 0.0) < amt:
                return False
        f = ind.fitness(world) * (self.kind == 0)
        hy_, hx_ = farmer.home
        H, W = f.shape
        y0, y1 = max(0, hy_ - radius), min(H, hy_ + radius + 1)
        x0, x1 = max(0, hx_ - radius), min(W, hx_ + radius + 1)
        win = f[y0:y1, x0:x1]
        if win.max() < self.cfg.industry_min_fit:
            return False
        iy, ix = np.unravel_index(int(np.argmax(win)), win.shape)
        y, x = y0 + int(iy), x0 + int(ix)
        for res, amt in ind.cost.items():
            farmer.ledger[res] -= amt
        self.kind[y, x] = kind
        self.owner[y, x] = farmer.index
        if not ind.on_water:
            world.flora.kind[y, x] = BARE
            world.farm[y, x] = False
        farmer.built.append((y, x, kind))
        farmer._walk(y, x, trips=3, kind="industry")
        return True

    def step(self, world, farmers: list) -> None:
        """Run every industry: consume, produce, and write effects into the fields."""
        cfg = self.cfg
        if not self.kind.any():
            return
        fits = {k: ind.fitness(world) for k, ind in TABLE.items()}
        for y, x, k, owner in self.sites():
            ind = TABLE[k]
            f = float(fits[k][y, x])
            self.fit[y, x] = f
            if owner < 0 or owner >= len(farmers):
                continue
            farmer = farmers[owner]
            if world.tick_now % cfg.build_every == 0 and farmer.can(cfg.hours_industry):   # a working site is visited (D-028 tread); it costs hours (D-050)
                farmer.spend(cfg.hours_industry, "industry")
                farmer._walk(y, x, kind="industry")
            # A site that no longer fits (the river moved, the forest is gone) produces nothing.
            if f <= 0.0:
                if k == KILN:
                    # ... but a kiln with an empty forest around it is simply abandoned.
                    self.kind[y, x] = 0; self.owner[y, x] = -1
                continue
            # Inputs from the ledger; a kiln without stocked wood fells the nearest tree itself.
            scale = f
            for res, amt in ind.inputs.items():
                need = amt * f
                have = farmer.ledger.get(res, 0.0)
                if have >= need:
                    farmer.ledger[res] = have - need
                elif k == KILN:
                    if not self._fell_near(world, y, x, 3):
                        scale = 0.0
                else:
                    scale = min(scale, have / max(need, 1e-9))
                    farmer.ledger[res] = 0.0
            for res, amt in ind.outputs.items():
                farmer.ledger[res] = farmer.ledger.get(res, 0.0) + amt * scale
            # Effects.
            if k == SALTPAN and scale > 0:
                y0, y1 = max(0, y - 1), min(self.kind.shape[0], y + 2)
                x0, x1 = max(0, x - 1), min(self.kind.shape[1], x + 2)
                world.ground.Gs[y0:y1, x0:x1] += cfg.saltpan_salinise * scale
                world.ground.Gs[y, x] = max(float(world.ground.Gs[y, x]), 1.0)

    @staticmethod
    def _fell_near(world, y, x, r) -> bool:
        H, W = world.tree.shape
        best, bd = None, 1e9
        for yy in range(max(0, y - r), min(H, y + r + 1)):
            for xx in range(max(0, x - r), min(W, x + r + 1)):
                if world.flora.kind[yy, xx] in TREES:
                    d = (yy - y) ** 2 + (xx - x) ** 2
                    if d < bd:
                        best, bd = (yy, xx), d
        if best is None:
            return False
        world.flora.kind[best] = BARE
        return True

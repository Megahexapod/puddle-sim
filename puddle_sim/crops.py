"""Crops: what the farmer plants, chosen per field by the same band logic as everything else.

DRY     旱作  — moist but not flooded ground, sun, warmth; feeds on soil h.
RICE    水稻  — a flooded fresh field (the paddy), full sun, warm; leaks more nutrients; snails eat it;
                when the paddy dries the farmer irrigates from the nearest water (D-027).
CALTROP 菱角  — floats on still shallow fresh water (ponds), sun, warm; feeds on the water's
                nutrients; shades its cell.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import Config

NONE, DRY, RICE, CALTROP = 0, 1, 2, 3


@dataclass
class Crop:
    key: str
    name: str
    on_water: bool
    d: tuple            # inundation band (d_ema)
    m: tuple            # soil moisture band (land crops)
    light: int
    t_lo: float
    base: float         # food per tick at fitness 1 and h_mid (set from base_year by set_clock, D-049)
    runoff: float       # multiplier on runoff pollution
    color: tuple
    base_year: float = 0.0   # food per year at fitness 1: dry 120 (2.5 t/ha), rice 240 (5 t/ha), caltrop 144


TABLE = {
    DRY: Crop("dry", "旱作", False, d=(0.0, 0.03), m=(0.15, 0.8), light=10, t_lo=8.0, base=0.010, base_year=120.0, runoff=1.0,
              color=(0.45, 0.30, 0.17)),
    RICE: Crop("rice", "水稻", False, d=(0.02, 0.35), m=(0.0, 1.0), light=12, t_lo=15.0, base=0.020, base_year=240.0, runoff=1.5,
               color=(0.55, 0.62, 0.20)),
    CALTROP: Crop("caltrop", "菱角", True, d=(0.15, 1.2), m=(0.0, 1.0), light=11, t_lo=15.0, base=0.012, base_year=144.0, runoff=0.3,
                  color=(0.15, 0.40, 0.22)),
}


def set_clock(year_ticks: int) -> None:
    """Yields are written per year; the clock says how many ticks that is (D-049)."""
    for c in TABLE.values():
        if c.base_year > 0:
            c.base = c.base_year / year_ticks


set_clock(3000)


def _geo(factors):
    """Weighted geometric mean of (factor, weight) pairs, factors clipped to 0..1 -- the same Liebig
    combination creatures.py has used since D-003. The scarcest factor decides; five factors at 0.6
    give 0.6, not 0.08 (D-046: the product left the fields at a twentieth of a real farm)."""
    tot = sum(w for _, w in factors)
    out = np.ones_like(factors[0][0], dtype=np.float32)
    for f, w in factors:
        out = out * np.power(np.clip(f, 0.0, 1.0), w)
    return out ** (1.0 / tot)


def fitness(world, k: int) -> np.ndarray:
    """0..1(+) -- how well crop k would do on every cell right now (ignores who owns it).
    Gates (land/water, salt, the inundation and moisture bands) zero a cell; the continuous factors
    (soil, light, warmth, paddy water, nutrients) combine by weighted geometric mean; decomposer
    soil is a bonus on top."""
    cfg: Config = world.cfg
    c = TABLE[k]
    hy = world.hydro
    d = hy.d_ema
    h = world.soil.h
    soil = np.where(world.soil.eroding, 0.0, np.clip((h - cfg.h_min) / (cfg.h_mid - cfg.h_min), 0.0, 1.0))
    light = np.clip((world.light - c.light + 3) / 3.0, 0.0, 1.0)
    warm = np.clip((world.temp - c.t_lo) / 6.0, 0.0, 1.0)
    if k == CALTROP:
        gate = ((d >= c.d[0]) & (d <= c.d[1])) & (world.salt_ema < 0.2)
        still = np.clip((world.stillness() - 0.3) / 0.5, 0.0, 1.0)
        fresh = np.clip((0.2 - world.salt_ema) / 0.15, 0.0, 1.0)
        nutrients = np.clip(0.3 + 0.7 * (1.0 - world.quality), 0.0, 1.0)     # nutrients feed it
        f = _geo([(still, 1.0), (fresh, 0.5), (nutrients, 1.0), (light, 1.0), (warm, 1.0)])
    elif k == RICE:
        # The paddy holds its own water (world.paddy), filled by irrigation from the river.
        gate = (~world.water) & (~world.saline())
        water = np.clip((world.paddy - 0.15) / 0.25, 0.0, 1.0)
        soil_salt = np.clip((0.15 - world.soil_salinity()) / 0.1, 0.0, 1.0)
        f = _geo([(water, 1.5), (soil, 1.0), (soil_salt, 0.5), (light, 1.0), (warm, 1.0)])
    else:  # DRY
        m = world.soil.m
        gate = ((d >= c.d[0]) & (d <= c.d[1])) & (~world.water) & ((m >= c.m[0]) & (m <= c.m[1])) & (~world.saline())
        f = _geo([(soil, 1.0), (light, 1.0), (warm, 1.0)])
    f = f * gate
    f = np.where(world.soil.rich & gate, f * 1.5, f)                          # decomposer soil: the old x2, tempered
    return f.astype(np.float32)


def choose(world, y: int, x: int) -> int:
    """The crop for this cell: caltrop on water; on land a paddy if the river is close, flat and
    warm (the farmer will irrigate it), else a dry field."""
    if world.water[y, x]:
        return CALTROP
    hy = world.hydro
    H, W = hy.d.shape
    y0, y1 = max(0, y - 2), min(H, y + 3)
    x0, x1 = max(0, x - 2), min(W, x + 3)
    near_water = bool((hy.d_ema[y0:y1, x0:x1] >= 0.2).any())
    gy, gx = np.gradient(hy.b[max(0, y - 1): y + 2, max(0, x - 1): x + 2])
    flat = float(np.sqrt(gy ** 2 + gx ** 2).mean()) < 0.35
    warm = float(world.temp[y, x]) >= TABLE[RICE].t_lo
    fresh = float(world.soil_salinity()[y, x]) < 0.15
    if near_water and flat and warm and fresh:
        return RICE
    return DRY

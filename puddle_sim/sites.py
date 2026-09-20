"""Where would anyone live here? One scoring field for the player's start and for NPC settlements.

Liebig again: a site with no fresh water nearby is worth nothing however much wood it has. The
hard gates are water, dryness and soil; the soft factors are flatness, farmable land, wood and
food. Confluences score high on their own (two directions of water and forest).
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

from .config import Config
from .gen import SOIL


def _window_sum(a: np.ndarray, r: int) -> np.ndarray:
    m = a.astype(np.float32)
    p = np.pad(m, r)
    out = np.zeros_like(m)
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            out += p[r + dy: r + dy + m.shape[0], r + dx: r + dx + m.shape[1]]
    return out


def score(world) -> np.ndarray:
    from .world import chebyshev_distance_to
    cfg: Config = world.cfg
    hy = world.hydro
    land = ~world.water

    # Hard gates.
    fresh = (hy.d_ema >= cfg.d_min) & (world.salt_ema < 0.2)
    dist_fresh = chebyshev_distance_to(fresh, 3)
    water_f = np.where(dist_fresh <= 2, 1.0, np.where(dist_fresh == 3, 0.6, 0.0))
    dry = land & (hy.d_max < cfg.d_min) & ~world.saline() & (hy.d_ema < cfg.d_min)
    soil = world.substrate == SOIL

    # Soft factors.
    gy, gx = np.gradient(hy.b)
    slope = np.sqrt(gy ** 2 + gx ** 2)
    flat = np.clip(1.0 - (slope - 0.3) / 0.5, 0.0, 1.0)
    farmable = land & (world.water_dist <= cfg.water_reach) & (world.water_dist > 0) & soil & ~world.saline()
    farm_f = np.clip(_window_sum(farmable, 6) / 40.0, 0.0, 1.0)
    wood_f = np.clip(_window_sum(world.tree, 8) / 30.0, 0.0, 1.0) * np.where(world.light >= 12, 1.0, 0.3)
    from .creatures import SPECIES
    hab = np.zeros(land.shape, dtype=np.float32)
    for key in ("shrimp", "medaka"):
        hab = np.maximum(hab, np.where(world.water, SPECIES[key].habitat(world), 0.0))
    food_f = np.clip(_window_sum(hab, 6) / (0.6 * 30.0), 0.0, 1.0)

    s = (np.power(np.maximum(flat, 1e-3), 0.5) * np.power(np.maximum(farm_f, 1e-3), 1.0)
         * np.power(np.maximum(wood_f, 1e-3), 1.0) * np.power(np.maximum(food_f, 1e-3), 0.5)) ** (1.0 / 3.0)
    s = s * water_f * dry * soil
    return s.astype(np.float32)


def pick(s: np.ndarray, k: int, spacing: int) -> List[Tuple[int, int]]:
    """Greedy: best site, then the best at least `spacing` away, and so on."""
    s = s.copy()
    out = []
    H, W = s.shape
    for _ in range(k):
        i = int(np.argmax(s))
        if s.flat[i] <= 0:
            break
        y, x = divmod(i, W)
        out.append((y, x))
        y0, y1 = max(0, y - spacing), min(H, y + spacing + 1)
        x0, x1 = max(0, x - spacing), min(W, x + spacing + 1)
        s[y0:y1, x0:x1] = 0.0
    return out

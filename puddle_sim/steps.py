"""Footsteps from Minecraft into the tread field (D-030).

The mod writes `puddle_steps.csv` (tick,player,x,y,z,onGround) once a second. Here it becomes
tread: four blocks are one cell (D-023), consecutive samples are joined so a second of walking
does not leave holes, and every crossing compacts the cell like the farmer's trips do. The origin
maps the MC world onto the grid: give it, or let the walk's own bounding box choose it.

Why this exists: the scripted farmer walks straight lines. People walk around puddles. The
questionnaire's second section (do routes appear without being asked for) is measured with these.
"""
from __future__ import annotations

import csv
from typing import Dict, Optional, Tuple

import numpy as np


def load(path: str) -> Dict[str, np.ndarray]:
    """Per player: (n, 3) array of (tick, x, z), ground samples only."""
    out: Dict[str, list] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("onGround", "1") not in ("1", "true", "True"):
                continue
            out.setdefault(row["player"], []).append((float(row["tick"]), float(row["x"]), float(row["z"])))
    return {k: np.array(v, dtype=np.float64) for k, v in out.items() if v}


def fold(world, path: str, origin: Optional[Tuple[float, float]] = None, blocks_per_cell: float = 4.0,
         amount: float = None, flip_z: bool = False) -> dict:
    """Tread the world with a step log. Returns what happened, for the caption."""
    g = world.ground
    H, W = g.pack.shape
    walks = load(path)
    if not walks:
        return {"players": 0, "samples": 0, "cells": 0}
    allxy = np.concatenate([w[:, 1:3] for w in walks.values()])
    if origin is None:
        # Centre the walk's bounding box on the grid.
        cx, cz = allxy.mean(axis=0)
        origin = (cx - W * blocks_per_cell / 2.0, cz - H * blocks_per_cell / 2.0)
    ox, oz = origin
    before = g.pack.copy()
    samples = 0
    for name, w in walks.items():
        cells = np.stack([((w[:, 2] - oz) / blocks_per_cell), ((w[:, 1] - ox) / blocks_per_cell)], axis=1)
        if flip_z:
            cells[:, 0] = H - 1 - cells[:, 0]
        cells = np.floor(cells).astype(int)
        prev = None
        for i, (cy, cx_) in enumerate(cells):
            samples += 1
            if prev is not None and (i == 0 or w[i, 0] - w[i - 1, 0] <= 60):   # a gap of >3 s is a teleport, not a walk
                g.tread(int(prev[0]), int(prev[1]), int(cy), int(cx_), world, amount=amount)
            elif 0 <= cy < H and 0 <= cx_ < W and not world.water[cy, cx_]:
                g.pack[cy, cx_] = min(1.0, g.pack[cy, cx_] + (g.cfg.pack_step if amount is None else amount))
            prev = (cy, cx_)
    return {"players": len(walks), "samples": samples, "cells": int((g.pack > before).sum()),
            "origin": (float(ox), float(oz))}

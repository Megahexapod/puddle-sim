"""Ground: bedrock, ground water, infiltration, tread (D-028 ①, D-029).

Every cell is a column: bedrock at `b_rock`, surface at `hydro.b`, so the soil is `b - b_rock`
thick. The column holds water up to porosity x thickness; `G` is how much it holds (depth units).
The water table stands at `b_rock + G / porosity`.

Rain on land splits: an unsaturated, untrodden column takes a share (by substrate) into G, the rest
becomes surface water and runs off. Water in the ground moves sideways toward a lower water table
(Darcy, four neighbours, same shape as the pipes in hydro.py). A column that is fuller than it can
hold exfiltrates the excess to the surface — that is a spring at a valley foot, baseflow into the
river in the dry season, and a pit dug below the water table filling with water. Evapotranspiration
takes from the store by temperature; a canopy takes more.

Salt rides in the ground water (D-021 recast): `Gs` is the salt mass in the column. Rain dilutes it,
salty surface water recharges it, lateral flow carries it at the source's concentration, evaporation
takes the water and leaves the salt (that is how a basin salinises), exfiltration hands it back to
the surface. Ground salinity = Gs / G; above `salt_flip` the ground is 鹽化.

Tread (`pack`) is the reverse coupling: ground walked on compacts, compacted ground takes less rain,
rain that cannot enter stays on the path as a puddle.
"""
from __future__ import annotations

import numpy as np

from .config import Config


def _shift(a: np.ndarray, dy: int, dx: int, mode: str = "edge") -> np.ndarray:
    """a shifted so that out[y, x] = a[y + dy, x + dx]; edge-padded (walls) or zero-padded (no phantom inflow)."""
    p = np.pad(a, 1, mode=mode) if mode == "edge" else np.pad(a, 1)
    H, W = a.shape
    return p[1 + dy: 1 + dy + H, 1 + dx: 1 + dx + W]


class Ground:
    def __init__(self, cfg: Config, world):
        self.cfg = cfg
        H, W = cfg.height, cfg.width
        sub = world.substrate
        self.b_rock = (world.hydro.b - self.depth_table(cfg)[np.clip(sub, 0, 4)]).astype(np.float32)
        self.pack = np.zeros((H, W), dtype=np.float32)
        # Start half full on land, full under water: the rules find their own level in the spin-up.
        cap = self.capacity(world)
        self.G = np.where(world.water, cap, 0.5 * cap).astype(np.float32)
        self.Gs = np.zeros((H, W), dtype=np.float32)          # salt mass in the column
        self.exfil = np.zeros((H, W), dtype=np.float32)      # surface water the ground gave up this tick
        self.infil = np.zeros((H, W), dtype=np.float32)      # rain that entered this tick
        self.infil_frac = np.zeros((H, W), dtype=np.float32)

    # ------------------------------------------------------------------ tables

    @staticmethod
    def depth_table(cfg: Config) -> np.ndarray:
        return np.array([cfg.depth_soil, cfg.depth_rock, cfg.depth_sand, cfg.depth_mud, cfg.depth_mud], dtype=np.float32)

    def porosity(self, world) -> np.ndarray:
        cfg = self.cfg
        t = np.array([cfg.por_soil, cfg.por_rock, cfg.por_sand, cfg.por_mud, cfg.por_mud], dtype=np.float32)
        return t[np.clip(world.substrate, 0, 4)]

    def infil_rate(self, world) -> np.ndarray:
        cfg = self.cfg
        t = np.array([cfg.infil_soil, cfg.infil_rock, cfg.infil_sand, cfg.infil_mud, cfg.infil_mud], dtype=np.float32)
        return t[np.clip(world.substrate, 0, 4)]

    def thickness(self, world) -> np.ndarray:
        return np.maximum(world.hydro.b - self.b_rock, 0.02).astype(np.float32)

    def capacity(self, world) -> np.ndarray:
        return (self.porosity(world) * self.thickness(world)).astype(np.float32)

    def saturation(self, world) -> np.ndarray:
        return np.clip(self.G / np.maximum(self.capacity(world), 1e-6), 0.0, 1.0).astype(np.float32)

    def water_table(self, world) -> np.ndarray:
        return (self.b_rock + self.G / np.maximum(self.porosity(world), 1e-3)).astype(np.float32)

    def salinity(self) -> np.ndarray:
        """Salt per unit of ground water, 0..1. A little bound water keeps a bone-dry trace from reading as brine."""
        return np.clip(self.Gs / (self.G + 0.02), 0.0, 1.0).astype(np.float32)

    def add_salt(self, mask_or_idx, amount) -> None:
        self.Gs[mask_or_idx] = np.minimum(self.Gs[mask_or_idx] + amount, 1.0)

    # ------------------------------------------------------------------ per tick

    def split_rain(self, world, rain: np.ndarray) -> np.ndarray:
        """Take the ground's share of this tick's rain into G; return what stays on the surface."""
        cfg = self.cfg
        per_tick = cfg.hydro_dt * cfg.hydro_substeps
        S = self.saturation(world)
        land = ~world.water
        f = self.infil_rate(world) * (1.0 - self.pack) * (1.0 - S)
        f = np.where(land, np.clip(f, 0.0, 1.0), 0.0).astype(np.float32)
        self.infil_frac = f
        self.infil = (rain * per_tick * f).astype(np.float32)
        self.G += self.infil
        return (rain * (1.0 - f)).astype(np.float32)

    def step(self, world) -> None:
        cfg = self.cfg
        hy = world.hydro
        cap = self.capacity(world)
        por = self.porosity(world)
        G = self.G

        Gs = self.Gs

        # The river feeds its banks (and the lake its shore): wet cells fill their own column — with
        # the water's salt.
        wet = hy.d >= cfg.d_min
        take = np.where(wet, cfg.gw_recharge * np.clip(cap - G, 0.0, None), 0.0)
        take = np.minimum(take, hy.d * 0.5).astype(np.float32)
        hy.d -= take
        G = G + take
        Gs = Gs + take * world.salinity

        # Sideways: toward the lower water table, four neighbours, never more than the cell holds.
        # Under open water the table IS the water surface (the river feeds its banks; a lake sets
        # the level of the ground around it). Salt travels at the sender's concentration.
        wt = self.b_rock + G / np.maximum(por, 1e-3)
        wt = np.where(wet, np.maximum(wt, hy.b + hy.d), wt)
        c = Gs / (G + 1e-6)
        flows = []
        total = np.zeros_like(G)
        for dy, dx in ((0, -1), (0, 1), (-1, 0), (1, 0)):
            q = cfg.gw_flow * np.clip(wt - _shift(wt, dy, dx), 0.0, None) * por
            flows.append((dy, dx, q))
            total += q
        K = np.where(total > 0, np.minimum(1.0, G / (total + 1e-9)), 0.0)
        out = np.zeros_like(G)
        inc = np.zeros_like(G)
        inc_s = np.zeros_like(G)
        for dy, dx, q in flows:
            q = q * K
            out += q
            inc += _shift(q, -dy, -dx, mode="zero")        # what the neighbour sent us
            inc_s += _shift(q * c, -dy, -dx, mode="zero")
        Gs = Gs - out * c + inc_s
        G = G - out + inc

        # Evapotranspiration: warmth takes the water and leaves the salt; a canopy takes more.
        tf = np.clip(0.4 + world.temp / 20.0, 0.0, 2.5)
        canopy = world.canopy()
        G = G - cfg.gw_et * tf * (1.0 + 0.5 * canopy) * (G > 0)

        # Fuller than the column holds: the excess comes out on top (spring, seep, pit, baseflow),
        # taking its salt into the surface water.
        G = np.clip(G, 0.0, None)
        excess = np.clip(G - cap, 0.0, None)
        c = Gs / (G + 1e-6)
        d_old = hy.d.copy()
        hy.d += excess.astype(np.float32)
        if excess.any():
            world.salinity = np.where(excess > 0, (world.salinity * d_old + c * excess) / (hy.d + 1e-6),
                                      world.salinity).astype(np.float32)
        Gs = Gs - excess * c
        self.exfil = excess.astype(np.float32)
        self.G = (G - excess).astype(np.float32)
        self.Gs = np.clip(Gs, 0.0, None).astype(np.float32)

        # Tread relaxes where roots work the ground; hard bare ground loses its life slowly.
        r = world.soil.r
        self.pack = np.clip(self.pack - cfg.pack_relax * (1.0 + r), 0.0, 1.0).astype(np.float32)
        hard = (self.pack > cfg.pack_grass) & ~world.water
        world.soil.h[hard] = np.maximum(0.0, world.soil.h[hard] - cfg.pack_h_loss)

    # ------------------------------------------------------------------ mutation

    def tread(self, y0: int, x0: int, y1: int, x1: int, world, amount: float = None) -> None:
        """Walk a straight line (Bresenham) from (y0,x0) to (y1,x1); land cells crossed compact."""
        amount = self.cfg.pack_step if amount is None else amount
        H, W = self.pack.shape
        dy, dx = abs(y1 - y0), abs(x1 - x0)
        sy, sx = (1 if y1 >= y0 else -1), (1 if x1 >= x0 else -1)
        err = dx - dy
        y, x = y0, x0
        while True:
            if 0 <= y < H and 0 <= x < W and not world.water[y, x]:
                self.pack[y, x] = min(1.0, self.pack[y, x] + amount)
            if y == y1 and x == x1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy; x += sx
            if e2 < dx:
                err += dx; y += sy

    def dig(self, y: int, x: int, world, amount: float) -> None:
        """Lower the surface; digging through the soil takes the rock with it."""
        hy = world.hydro
        hy.b[y, x] -= amount
        if hy.b[y, x] < self.b_rock[y, x]:
            self.b_rock[y, x] = hy.b[y, x]
        self.pack[y, x] = 0.0

    def fill(self, y: int, x: int, world, amount: float, pack: float = 0.3) -> None:
        world.hydro.b[y, x] += amount
        self.pack[y, x] = pack

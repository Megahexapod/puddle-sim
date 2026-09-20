"""Water physics: a height field with the virtual-pipe shallow-water model (Mei, Decaudin, Hu 2007).

Every cell has terrain height b and water depth d. Water head H = b + d. Each cell owns four
outflow pipes (to its W/E/N/S neighbours); a pipe's flux accelerates with the head difference,
is scaled down so a cell never sends more water than it holds, and the depth update is just
inflow minus outflow. Velocity is the net flux through the cell divided by depth.

Why this model: it is O(cells) per step, entirely local (a cell only talks to its 4 neighbours),
gives a real velocity field, fills basins into ponds and overflows them at the lowest saddle
without any special-casing, and erosion/deposition (the next layer) are a two-line add-on in the
same paper. Local + cheap is the property that has to survive the trip back to MC.

Boundaries: west/north/south edges are walls; the east edge is open (water leaves the map).
A spring at the head of the valley feeds the river; a uniform light rain feeds everything else;
evaporation takes a thin film back so plateaus stay dry.
"""
from __future__ import annotations

import numpy as np

from .config import Config


class Hydro:
    def __init__(self, cfg: Config, terrain: np.ndarray, springs: list, pond: tuple = None):
        self.cfg = cfg
        self.b = terrain.astype(np.float32)
        self.pond = pond
        self.springs = springs            # [(y, x, flow), ...]
        self.spring = (springs[0][0], springs[0][1])
        H, W = self.b.shape
        self.d = np.zeros((H, W), dtype=np.float32)
        self.fL = np.zeros((H, W), dtype=np.float32)
        self.fR = np.zeros((H, W), dtype=np.float32)
        self.fT = np.zeros((H, W), dtype=np.float32)
        self.fB = np.zeros((H, W), dtype=np.float32)
        self.vx = np.zeros((H, W), dtype=np.float32)
        self.vy = np.zeros((H, W), dtype=np.float32)
        self.speed = np.zeros((H, W), dtype=np.float32)
        # Per-tick accumulated volumes through each pipe (for advecting scalars once per tick).
        self.QL = np.zeros((H, W), dtype=np.float32)
        self.QR = np.zeros((H, W), dtype=np.float32)
        self.QT = np.zeros((H, W), dtype=np.float32)
        self.QB = np.zeros((H, W), dtype=np.float32)
        self.outflow_tick = 0.0   # volume that left through the east edge this tick
        self.d_prev = self.d.copy()
        self.time = 0.0
        self.fIn = np.zeros(H, dtype=np.float32)      # the sea's pipe into the east-edge column
        self.Q_sea = np.zeros(H, dtype=np.float32)    # volume that came in from the sea this tick
        self.d_ema = np.zeros((H, W), dtype=np.float32)
        self.speed_ema = np.zeros((H, W), dtype=np.float32)
        self.d_max = np.zeros((H, W), dtype=np.float32)      # flood mark: the deepest this cell has been
        self.sea_head = float(cfg.sea_level)
        self.rain_now = float(cfg.rain)
        self.evap_now = float(cfg.evap)
        self.surge = 0.0                    # storm surge on the sea head (world sets it while a storm is on, D-039)
        self.spring_mult = 1.0

    # ------------------------------------------------------------------ public

    @property
    def water(self) -> np.ndarray:
        return self.d >= self.cfg.d_min

    def step(self) -> None:
        """One sim tick = several small physics substeps."""
        self.QL[:] = 0; self.QR[:] = 0; self.QT[:] = 0; self.QB[:] = 0
        self.Q_sea[:] = 0
        self.outflow_tick = 0.0
        self.d_prev = self.d.copy()      # depth before this tick: the basis for scalar transport
        for _ in range(self.cfg.hydro_substeps):
            self._substep(self.cfg.hydro_dt)
        a = 1.0 / self.cfg.ema_ticks
        self.d_max = np.maximum(self.d_max, self.d)
        self.d_ema = ((1 - a) * self.d_ema + a * self.d).astype(np.float32)
        self.speed_ema = ((1 - a) * self.speed_ema + a * self.speed).astype(np.float32)

    def volume(self) -> float:
        return float(self.d.sum())

    # ------------------------------------------------------------------ physics

    def _substep(self, dt: float) -> None:
        cfg = self.cfg
        b, d = self.b, self.d
        H, W = b.shape

        # Sources (the climate sets today's rain, evaporation and thaw).
        d += dt * self.rain_now
        for sy, sx, q in self.springs:
            d[sy, sx] += dt * q * self.spring_mult
        if self.pond is not None:
            d[self.pond] += dt * cfg.pond_q

        # The sea breathes (inland maps: the east edge is just a low, still outlet).
        self.time += dt
        period = cfg.tide_period * cfg.hydro_dt * cfg.hydro_substeps
        if cfg.sea:
            slow_period = cfg.sea_slow_period * cfg.hydro_dt * cfg.hydro_substeps
            self.sea_head = float(cfg.sea_level + cfg.tide_amp * np.sin(2 * np.pi * self.time / period)
                                  + cfg.sea_slow_amp * np.sin(2 * np.pi * self.time / slow_period) + self.surge)
        elif cfg.basin:
            self.sea_head = float(b[:, -1].max() + 5.0)      # a wall: the east edge never drains
        else:
            self.sea_head = float(b[:, -1].min() - 0.5)
        # Sea water comes in over the east edge when the sea stands higher than the cell.
        damp = 1.0 - cfg.hydro_friction
        k = dt * cfg.gravity
        if cfg.sea:
            self.fIn = np.maximum(0.0, self.fIn * damp + k * (self.sea_head - (b[:, -1] + d[:, -1]))).astype(np.float32)
            d[:, -1] += self.fIn * dt
            self.Q_sea += self.fIn * dt

        head = b + d

        # Neighbour heads. Walls: neighbour head = own head (no flow). East edge: the sea.
        hL = np.empty_like(head); hL[:, 1:] = head[:, :-1]; hL[:, 0] = head[:, 0]
        hR = np.empty_like(head); hR[:, :-1] = head[:, 1:]; hR[:, -1] = self.sea_head
        hT = np.empty_like(head); hT[1:, :] = head[:-1, :]; hT[0, :] = head[0, :]
        hB = np.empty_like(head); hB[:-1, :] = head[1:, :]; hB[-1, :] = head[-1, :]

        fL = np.maximum(0.0, self.fL * damp + k * (head - hL))
        fR = np.maximum(0.0, self.fR * damp + k * (head - hR))
        fT = np.maximum(0.0, self.fT * damp + k * (head - hT))
        fB = np.maximum(0.0, self.fB * damp + k * (head - hB))

        # Never send more than the cell holds.
        total = fL + fR + fT + fB
        K = np.where(total > 0, np.minimum(1.0, d / (total * dt + 1e-9)), 0.0).astype(np.float32)
        fL *= K; fR *= K; fT *= K; fB *= K
        total = fL + fR + fT + fB

        # Inflows = neighbours' outflows pointed at us.
        inL = np.zeros_like(d); inL[:, 1:] = fR[:, :-1]
        inR = np.zeros_like(d); inR[:, :-1] = fL[:, 1:]
        inT = np.zeros_like(d); inT[1:, :] = fB[:-1, :]
        inB = np.zeros_like(d); inB[:-1, :] = fT[1:, :]
        inflow = inL + inR + inT + inB

        d_old = d.copy()
        d_new = d + dt * (inflow - total)
        d_new -= dt * self.evap_now * cfg.evap_mult
        d_new = np.maximum(0.0, d_new)

        # Velocity from net flux through the cell (positive x = east, positive y = south).
        dmean = np.maximum(0.5 * (d_old + d_new), cfg.d_min)
        vx = 0.5 * ((inL - fL) + (fR - inR)) / dmean
        vy = 0.5 * ((inT - fT) + (fB - inB)) / dmean
        wet = d_new >= cfg.d_min
        self.vx = np.where(wet, vx, 0.0).astype(np.float32)
        self.vy = np.where(wet, vy, 0.0).astype(np.float32)
        self.speed = np.minimum(np.sqrt(self.vx ** 2 + self.vy ** 2), cfg.speed_cap).astype(np.float32)

        self.fL, self.fR, self.fT, self.fB = fL, fR, fT, fB
        self.d = d_new.astype(np.float32)

        self.QL += fL * dt; self.QR += fR * dt; self.QT += fT * dt; self.QB += fB * dt
        self.outflow_tick += float(fR[:, -1].sum() * dt)

    # ------------------------------------------------------------------ scalar transport

    def advect(self, c: np.ndarray, edge_conc: float = 0.0) -> np.ndarray:
        """Move a substance with this tick's pipe volumes (upwind, mass-conserving).

        c is a concentration of a SUBSTANCE (per unit water): pollution, detritus, sediment, salt.
        Fresh water (spring, rain) carries none of it, so it dilutes — which is why quality is
        transported as pollution = 1 - quality, not as quality itself.
        edge_conc: concentration of the substance in the sea water that came in over the east edge.
        """
        cfg = self.cfg
        d0, d1 = self.d_prev, self.d
        QL, QR, QT, QB = self.QL, self.QR, self.QT, self.QB
        out = QL + QR + QT + QB
        # A cell cannot export more than it held at the start of the tick.
        scale = np.minimum(1.0, np.maximum(d0, 0.0) / (out + 1e-9))
        QL = QL * scale; QR = QR * scale; QT = QT * scale; QB = QB * scale
        out = QL + QR + QT + QB

        mass = c * d0
        inc = np.zeros_like(mass)
        inc[:, 1:] += QR[:, :-1] * c[:, :-1]
        inc[:, :-1] += QL[:, 1:] * c[:, 1:]
        inc[1:, :] += QB[:-1, :] * c[:-1, :]
        inc[:-1, :] += QT[1:, :] * c[1:, :]
        if edge_conc:
            inc[:, -1] += self.Q_sea * edge_conc
        mass = np.maximum(0.0, mass - out * c + inc)
        # Keep the substance even in very shallow cells (bank edges flicker around d_min);
        # zeroing them there leaked ~30% of a blob in 60 ticks. Ecology masks by `water` anyway.
        c_new = mass / np.maximum(d1, 1e-4)
        return c_new.astype(np.float32)

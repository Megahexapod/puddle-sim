"""Soil: one number per land cell, three visible states, and the deadwood loop.

h (0..1) = organic health.       h < h_min          -> 流失中 (eroding)
                                 h_min .. h_rich    -> 健康 (plain soil)
                                 h > h_rich         -> 分解者土壤 (rich / decomposer soil)
Litter keeps h around h_mid ("repairs, never creates": it cannot push h above h_mid).
Wood lying on the ground pushes the 3x3 around it above h_rich. Crops pull h down.
h diffuses slowly between land cells: a healthy patch spreads, an exhausted one drags.

Moisture m comes from the water physics (film on the cell + neighbouring water depth).
Roots r come from what grows on the cell and next to it. A cell is eroding when it is too wet for
its roots to hold (m > m_crit * (1 + r)) or too poor (h < h_min).

Eroding cells that touch moving water accumulate scour = speed * (1 - r); past a threshold the
bank collapses: terrain drops to the neighbouring bed, the cell becomes water, its soil becomes
sediment in the river. Sediment travels with the flow and settles where the water is slow,
raising the bed; when the bed climbs out of the water a new, poor, eroding cell appears.

Deadwood (茨 D-012/D-013): wood mass on a cell rots at a rate set by its moisture regime —
soaked and bone-dry do not rot; sun + rain rots fast but yields little; shade + damp rots slowly
and yields most. Yield = h pushed into the ring. When the wood is gone the surplus h above h_rich
is eaten back down (the ferns on the ground), and the rich patch fades unless fed more wood.
"""
from __future__ import annotations

import numpy as np

from .config import Config
from .gen import ROCK, SAND, MUD


def _nbr_max(a: np.ndarray) -> np.ndarray:
    """Max over the 8 neighbours (excluding self)."""
    p = np.pad(a, 1, mode="edge")
    out = np.full_like(a, -np.inf)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            out = np.maximum(out, p[1 + dy : 1 + dy + a.shape[0], 1 + dx : 1 + dx + a.shape[1]])
    return out


def _nbr_sum(a: np.ndarray) -> np.ndarray:
    p = np.pad(a, 1)
    out = np.zeros_like(a)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            out += p[1 + dy : 1 + dy + a.shape[0], 1 + dx : 1 + dx + a.shape[1]]
    return out


def _nbr_min_masked(a: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Min over 8 neighbours where mask is True; +inf where no masked neighbour."""
    big = np.where(mask, a, np.inf).astype(np.float32)
    p = np.pad(big, 1, constant_values=np.inf)
    out = np.full_like(big, np.inf)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            out = np.minimum(out, p[1 + dy : 1 + dy + a.shape[0], 1 + dx : 1 + dx + a.shape[1]])
    return out


class Soil:
    def __init__(self, cfg: Config, shape: tuple, rng: np.random.Generator):
        self.cfg = cfg
        H, W = shape
        self.h = np.full((H, W), cfg.h_mid, dtype=np.float32)
        self.wood = np.zeros((H, W), dtype=np.float32)     # deadwood mass lying on the cell
        self.wood_age = np.zeros((H, W), dtype=np.int32)   # ticks since placed (epiphyte stage)
        self.scour = np.zeros((H, W), dtype=np.float32)
        self.m = np.zeros((H, W), dtype=np.float32)
        self.r = np.zeros((H, W), dtype=np.float32)
        self.eroding = np.zeros((H, W), dtype=bool)
        self.rich = np.zeros((H, W), dtype=bool)
        self.bund = np.zeros((H, W), dtype=bool)     # a bunded paddy: wet by design, only poverty makes it erode (round 4)
        self.flooded = np.zeros((H, W), dtype=bool)  # paddy with standing water: anoxic soil (D-036)
        self.deferred = np.zeros((H, W), dtype=np.float32)   # decomposition the anoxia held back; paid when the paddy drains
        # bookkeeping
        self.collapsed_total = 0
        self.emerged_total = 0
        self.rng = rng

    # ------------------------------------------------------------------ derived

    def phase_update(self) -> None:
        cfg = self.cfg
        too_wet = (self.m > cfg.m_crit * (1.0 + self.r)) & ~self.bund
        self.eroding = too_wet | (self.h < cfg.h_min)
        self.rich = self.h > cfg.h_rich

    # ------------------------------------------------------------------ per tick

    def step(self, world) -> dict:
        """Advance soil one tick. Returns a dict of events for logging."""
        cfg = self.cfg
        hy = world.hydro
        land = ~world.water
        H, W = self.h.shape

        sub = world.substrate
        rock, sand, mud = sub == ROCK, sub == SAND, sub == MUD

        # --- moisture from the physics: own film + wettest neighbour (mud holds more) ---
        film = np.clip(hy.d / cfg.d_min, 0.0, 1.0)
        nb_depth = np.clip(_nbr_max(hy.d) / 0.5, 0.0, 1.0)
        # Ground water from the nearest water body: damp within a few cells, dry beyond (the
        # climate knob: m_reach wide = a green map, narrow = forest hugs the rivers).
        terrain = cfg.m_reach_gain * np.clip(1.0 - world.water_dist_far / cfg.m_reach, 0.0, 1.0)
        gw = cfg.m_gw * world.ground.saturation(world)              # a full column is damp ground (D-029)
        self.m = np.clip(cfg.m_base + terrain + gw + cfg.mud_moist * mud + 0.6 * film + 0.45 * nb_depth, 0.0, 1.0).astype(np.float32)

        # --- roots: own plant + half of the best neighbour; sand holds less ---
        own = np.where(world.tree, cfg.root_tree, np.where(world.farm, cfg.root_crop, cfg.root_grass))
        own = np.where(self.eroding & ~world.tree & ~world.farm, cfg.root_bare, own)  # bare eroding ground
        anchors = np.where(world.tree, cfg.root_tree, np.where(world.mangrove, 1.0, 0.0))
        self.r = np.clip((own + 0.5 * _nbr_max(anchors)) * np.where(sand, cfg.sand_root, 1.0), 0.0, 1.5).astype(np.float32)

        if cfg.mode == "mc":                       # D-030: bits and random ticks from here on
            return world.ground.soil_tick(world)

        # --- organic health ---
        h = self.h
        trees3 = world.count3x3_tree()
        # Litter repairs toward h_mid but never above it.
        h = h + cfg.litter_soil * trees3 * np.clip(cfg.h_mid - h, 0.0, None)
        # Crops drain. Under standing water the soil is anoxic and decomposes at a fraction of the
        # aerobic rate; what is held back is not gone, it is deferred: a drained paddy pays it out
        # (the dry-wet flush). A paddy kept flooded accumulates -- that is why the same field has
        # grown rice for millennia (D-036).
        self.flooded = self.bund & (world.paddy >= cfg.paddy_flooded)
        drain = cfg.crop_drain * np.where(self.flooded, cfg.anox_drain, 1.0)
        h = h - drain * world.farm
        self.deferred = np.where(self.flooded, self.deferred + cfg.crop_drain * (1.0 - cfg.anox_drain), self.deferred)
        pay = np.minimum(self.deferred, cfg.flush_rate) * (~self.flooded)
        h = h - pay
        self.deferred = ((self.deferred - pay) * (1.0 - cfg.humify)).astype(np.float32)   # part of it becomes humus for good
        # Wood on the ground: rot and yield by moisture regime.
        wet = self.m
        sun = 1.0 - world.cover()
        soaked = wet >= cfg.wood_soaked
        dry = wet <= cfg.wood_dry
        rot_mult = np.where(soaked | dry, 0.0, 1.0 + cfg.wood_sun_rot * sun)
        yield_mult = np.where(soaked | dry, 0.0, 1.0 - cfg.wood_sun_waste * sun)
        has_wood = self.wood > 0
        rot = np.where(has_wood, cfg.wood_rot * rot_mult, 0.0)
        rot = np.minimum(rot, self.wood)
        push = rot * yield_mult * cfg.wood_to_soil          # h delivered this tick
        # Most of it lands under the log (a rich node you can dig up), the rest in the ring.
        h = h + cfg.wood_under * push + (1.0 - cfg.wood_under) * _nbr_sum(push) / 8.0
        self.wood = (self.wood - rot).astype(np.float32)
        self.wood_age = np.where(has_wood, self.wood_age + 1, 0).astype(np.int32)
        # Ferns eat the surplus when no wood is nearby.
        wood_near = (_nbr_sum(has_wood.astype(np.float32)) + has_wood) > 0
        surplus = np.clip(h - cfg.h_rich, 0.0, None)
        h = h - np.where(~wood_near, cfg.fern_eat * surplus, 0.0)
        # Slow diffusion between land cells (the "phase spreads to neighbours").
        hl = np.where(land, h, np.nan)
        p = np.pad(hl, 1, mode="edge")
        acc = np.zeros_like(h); cnt = np.zeros_like(h)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                v = p[1 + dy : 1 + dy + H, 1 + dx : 1 + dx + W]
                ok = ~np.isnan(v)
                acc += np.where(ok, v, 0.0); cnt += ok
        nbr_mean = np.where(cnt > 0, acc / np.maximum(cnt, 1), h)
        h = h + cfg.h_diffusion * (nbr_mean - h)
        cap = np.where(sand, cfg.sand_h_cap, cfg.h_cap)
        h = np.where(rock, 0.0, h)                                   # rock has no soil to speak of
        self.h = np.where(land, np.clip(h, 0.0, cap), self.h).astype(np.float32)

        self.phase_update()
        self.eroding &= ~rock                                        # rock neither erodes nor heals
        self.rich &= ~rock

        # --- scour and collapse ---
        speed_nb = _nbr_max(np.where(world.water, hy.speed, 0.0))
        touching = land & (speed_nb > 0)
        scour_mult = np.where(sand, cfg.sand_scour, np.where(mud, cfg.mud_scour, 1.0))
        bed = world.flora.field("bed")                     # seagrass beds calm the water next to them
        bed = np.where(bed > 0, bed, 1.0)
        bed_nb = np.minimum(bed, _nbr_min_masked(bed, np.ones_like(rock)))
        # Still water does not scour (D-029): only flow above scour_v0 — or waves — pull at a bank.
        drive = cfg.scour_rate * np.clip(speed_nb - cfg.scour_v0, 0.0, None) + cfg.wave_rate * world.wave_exposure
        touching = (land & (speed_nb > cfg.scour_v0)) | (world.wave_exposure > 0)
        self.scour = np.where(self.eroding & touching & ~rock,
                              self.scour + drive * scour_mult * bed_nb * np.clip(1.0 - self.r, 0.0, 1.0),
                              self.scour * (1.0 - cfg.scour_heal)).astype(np.float32)
        collapse = land & (self.scour >= cfg.scour_crit) & ~rock
        n_collapse = int(collapse.sum())
        if n_collapse:
            bed = _nbr_min_masked(hy.b, world.water)
            floor = world.ground.b_rock
            ys, xs = np.nonzero(collapse)
            for y, x in zip(ys, xs):
                if not np.isfinite(bed[y, x]):
                    continue
                # Terrain drops to the neighbouring bed — never below its own bedrock (D-029). Hitting
                # the floor exposes rock, which neither erodes nor grows soil; the soil goes downstream.
                target = max(float(bed[y, x]), float(floor[y, x]))
                lost = float(hy.b[y, x]) - target
                if lost <= 0.0:
                    self.scour[y, x] = 0.0
                    if target <= float(floor[y, x]) + 1e-4:
                        world.substrate[y, x] = ROCK
                    continue
                hy.b[y, x] = target
                world.sediment_in[y, x] += lost * cfg.soil_to_sediment
                world.flora.kind[y, x] = 0
                world.farm[y, x] = False
                world.substrate[y, x] = ROCK if target <= float(floor[y, x]) + 1e-4 else MUD
                self.wood[y, x] = 0.0
                self.scour[y, x] = 0.0
                self.h[y, x] = cfg.h_silt
            self.collapsed_total += n_collapse
        return {"collapsed": n_collapse}

    def on_emerged(self, mask: np.ndarray) -> None:
        """Cells the river just built: poor silt, eroding until healed."""
        n = int(mask.sum())
        if n:
            self.h[mask] = self.cfg.h_silt
            self.scour[mask] = 0.0
            self.emerged_total += n

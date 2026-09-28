"""MC mode (D-030, knife 1852k): the same world with the land layers stored as a few bits each and
updated the way Minecraft updates blocks — on events and on random ticks — instead of as
continuous fields integrated every tick.

What is compressed, and how MC does the same thing:

  ground water  2 bit  dry/damp/wet/spring   lookup on "distance to water" + rain class + substrate,
                                             recomputed only when the water mask changes (event)
                                             — farmland's MOISTURE 0..7 is exactly this
  soil          2 bit  eroding/healthy/rich  random-tick transitions (litter heals, wood enriches,
                                             crops drain, hard ground dies) — grass/dirt/mycelium spread
  tread         2 bit  none/trodden/path/road counter on step events, relaxes on random tick — dirt_path
  rain          2 bit  region class          quantised pattern, fixed between recomputes — biome.hasRain
  salt          1 bit  saline                set when salty water dries on the cell, cleared by rain
                                             on random tick
  collapse      event                        an eroding bank next to fast water falls with a
                                             probability when random-ticked — gravel/sand falling
  plants        random tick                  the same table, but only the sampled cells roll, with
                                             probabilities scaled by the sampling ratio

The water physics is shared with the continuous mode: it is generation-time material here, and in
the shipped game the engine's own water (MC's 0..7 level automaton) stands in for it.

The point is the comparison: run both modes on the same map, same farmers, and see whether the
shrimp still live, the paths still appear, the lake still sits in its basin. If yes, these bits
are what goes back into blocks. If not, the sim says which bit is missing.
"""
from __future__ import annotations

import numpy as np

from .gen import MUD, ROCK, SAND
from .ground import Ground
from .world import chebyshev_distance_to

DRY, DAMP, WET, SPRING = 0, 1, 2, 3
ERODING, HEALTHY, RICH = 0, 1, 2


class GroundMC(Ground):
    """Same interface as Ground; bits and random ticks inside."""

    def __init__(self, cfg, world):
        super().__init__(cfg, world)
        H, W = cfg.height, cfg.width
        self.level = np.zeros((H, W), dtype=np.int8)        # ground water 0..3
        self.rain_class = np.ones((H, W), dtype=np.int8)     # 0..3
        self.tread_count = np.zeros((H, W), dtype=np.int16)  # step events; level = count // pack_per_level
        self.soil_state = np.full((H, W), HEALTHY, dtype=np.int8)
        self.saline_bit = np.zeros((H, W), dtype=bool)
        self.prev_water = world.water.copy()
        self.prev_salinity = np.zeros((H, W), dtype=np.float32)
        self.rt_per_section = cfg.mc_random_ticks            # MC: 3 per 16x16x16 section per tick
        self.n_sections = max(1, (H * W) // 256)
        self.n_samples = self.rt_per_section * self.n_sections
        self.scale = (H * W) / float(self.n_samples)          # probability multiplier for sampled cells
        self.mask = np.zeros((H, W), dtype=bool)
        self.G[:] = 0.0                                       # unused in this mode
        self.ticks = 0
        self.events = 0
        self.was_spring = np.zeros((H, W), dtype=bool)
        self._recompute_levels(world)

    # ------------------------------------------------------------------ readbacks (same names as Ground)

    def saturation(self, world) -> np.ndarray:
        # Calibrated to the continuous column: its banks sit around 0.3-0.5, not full.
        return np.array([0.0, 0.25, 0.5, 1.0], dtype=np.float32)[self.level]

    def salinity(self) -> np.ndarray:
        return self.saline_bit.astype(np.float32)

    @property
    def pack_level(self) -> np.ndarray:
        return np.minimum(3, self.tread_count // self.cfg.pack_per_level).astype(np.int8)

    # ------------------------------------------------------------------ events

    def _recompute_levels(self, world, region: np.ndarray = None) -> None:
        """Ground water from a lookup: distance to water, rain class, substrate. And the pit rule:
        a cell standing below the surface of water within 4 cells is a spring (it fills).
        `region`: only these cells are rewritten (a neighbour update reaches 4-5 cells, not the map)."""
        cfg = self.cfg
        hy = world.hydro
        water = hy.d >= cfg.d_min
        dist = chebyshev_distance_to(water, 8)
        base = np.where(dist <= 1, WET, np.where(dist <= 4, DAMP, DRY)).astype(np.int8)
        base = base + (self.rain_class >= 3).astype(np.int8) - (self.rain_class <= 0).astype(np.int8)
        base = base - (world.substrate == SAND).astype(np.int8)
        base = np.where(world.substrate == ROCK, DRY, base)
        base = np.clip(base, DRY, WET)
        # Pit rule (MC style "water within 4"): the highest water surface within 4 cells, minus a
        # little per cell of distance, is the local water table.
        head = np.where(water, hy.b + hy.d, -np.inf)
        p = np.pad(head, 4, mode="constant", constant_values=-np.inf)
        best = np.full(head.shape, -np.inf, dtype=np.float32)
        H, W = head.shape
        for dy in range(-4, 5):
            for dx in range(-4, 5):
                if dy == 0 and dx == 0:
                    continue
                d = max(abs(dy), abs(dx))
                best = np.maximum(best, p[4 + dy: 4 + dy + H, 4 + dx: 4 + dx + W] - cfg.mc_table_drop * d)
        spring = (~water) & (hy.b < best - 0.02) & (world.substrate != ROCK)
        new = np.where(spring, SPRING, base).astype(np.int8)
        self.level = new if region is None else np.where(region, new, self.level).astype(np.int8)
        self.events += 1

    def set_rain_class(self, pattern: np.ndarray) -> None:
        self.rain_class = np.digitize(pattern, [0.4, 0.8, 1.4]).astype(np.int8)   # 0..3

    # ------------------------------------------------------------------ per tick

    def split_rain(self, world, rain: np.ndarray) -> np.ndarray:
        cfg = self.cfg
        mult = np.array([0.25, 0.75, 1.25, 2.0], dtype=np.float32)[self.rain_class]
        r = rain.mean() * mult                                  # the sky is one number; the class spreads it
        infil = np.array([cfg.infil_soil, cfg.infil_rock, cfg.infil_sand, cfg.infil_mud, cfg.infil_mud], dtype=np.float32)[np.clip(world.substrate, 0, 4)]
        f = infil * (1.0 - self.pack_level / 3.0) * (1.0 - np.minimum(self.level, 2) / 2.0)
        f = np.round(f * 4.0) / 4.0                             # quarter steps: a lookup, not a float
        f = np.where(world.water, 0.0, f).astype(np.float32)
        self.infil_frac = f
        self.infil = (r * cfg.hydro_dt * cfg.hydro_substeps * f).astype(np.float32)
        return (r * (1.0 - f)).astype(np.float32)

    def step(self, world) -> None:
        cfg = self.cfg
        hy = world.hydro
        self.ticks += 1
        H, W = self.level.shape
        rng = world.rng
        # --- random tick sample for this tick (MC: 3 cells per section) ---
        idx = rng.integers(0, H * W, size=self.n_samples)
        self.mask[:] = False
        self.mask.ravel()[idx] = True
        mask = self.mask

        # --- events: the water moved ---
        water = hy.d >= cfg.d_min
        changed = water ^ self.prev_water
        if self.ticks % cfg.mc_gw_every == 0:
            self._recompute_levels(world)                       # the season: everyone looks again
        elif changed.any():
            region = chebyshev_distance_to(changed, 5) <= 5      # a neighbour update, five cells out
            self._recompute_levels(world, region)
        if changed.any():
            dried = self.prev_water & ~water & (self.prev_salinity > 0.35)
            self.saline_bit |= dried
        self.prev_water = water.copy()
        self.prev_salinity = world.salinity.copy()

        # --- springs give water once, when the lookup flips them (the pit fills; no perpetual source) ---
        spring = (self.level == SPRING) & ~water
        new_spring = spring & ~self.was_spring
        if new_spring.any():
            hy.d[new_spring] = np.maximum(hy.d[new_spring], cfg.d_min * 1.5)
        self.exfil = np.where(new_spring, cfg.d_min, 0.0).astype(np.float32)
        self.was_spring = spring

        # --- random-ticked transitions ---
        rnd = rng.random((H, W))
        # tread relaxes; rain leaches salt
        relax = mask & (self.tread_count > 0) & (rnd < cfg.pack_relax * 3 * cfg.pack_per_level * self.scale * (1.0 + world.soil.r))
        self.tread_count[relax] -= 1
        leach = mask & self.saline_bit & (self.rain_class >= 2) & (rnd < cfg.mc_leach_p)
        self.saline_bit[leach] = False
        self.pack = (self.pack_level / 3.0).astype(np.float32)

    def soil_tick(self, world) -> dict:
        """Soil state transitions and bank collapse, on the sampled cells only."""
        cfg = self.cfg
        hy = world.hydro
        so = world.soil
        H, W = self.level.shape
        mask = self.mask
        rng = world.rng
        rnd = rng.random((H, W))
        land = ~world.water
        st = self.soil_state
        trees3 = world.count3x3_tree()
        rock = world.substrate == ROCK
        s = self.scale
        # Transition rates derived from the continuous rule: a state is 0.3 of h, so the chance a
        # random-ticked cell steps is (dh/dt / 0.3) x the sampling ratio. Nothing is tuned by hand.
        step = 0.3
        # litter heals eroding -> healthy
        heal = mask & land & (st == ERODING) & (trees3 >= 1) & (rnd < cfg.litter_soil * trees3 * 0.3 / step * s)
        # wood on or next to the cell enriches healthy -> rich; the log is spent when its own cell turns
        from .soil import _nbr_sum as _ns
        wood_nb = (_ns((so.wood > 0).astype(np.float32)) > 0)
        enrich = mask & land & (st == HEALTHY) & ((so.wood > 0) | wood_nb) & (rnd < cfg.wood_rot * cfg.wood_to_soil / step * s)
        # crops drain healthy/rich one step down
        drain = mask & land & world.farm & (st >= HEALTHY) & (rnd < cfg.crop_drain / step * s)
        # no wood near: rich -> healthy (the ferns)
        wood_near = (_ns((so.wood > 0).astype(np.float32)) + (so.wood > 0)) > 0
        fern = mask & land & (st == RICH) & ~wood_near & (rnd < cfg.fern_eat * 0.2 / step * s)
        # hard bare ground loses its life
        hard = mask & land & (self.pack_level >= 2) & (st >= HEALTHY) & (rnd < cfg.pack_h_loss / step * s)
        st = np.where(heal, HEALTHY, st)
        st = np.where(enrich, RICH, st)
        st = np.where(drain | hard, np.maximum(ERODING, st - 1), st)
        st = np.where(fern, HEALTHY, st)
        st = np.where(rock, ERODING, st)
        so.wood[enrich & (so.wood > 0)] = 0.0
        self.soil_state = st.astype(np.int8)
        so.h = np.where(land, np.array([0.1, 0.5, 0.9], dtype=np.float32)[st], so.h).astype(np.float32)
        too_wet = so.m > cfg.m_crit * (1.0 + so.r)
        so.eroding = ((st == ERODING) | too_wet) & ~rock
        so.rich = (st == RICH) & ~rock & ~too_wet

        # --- collapse: an eroding bank next to fast water falls when ticked, with a probability ---
        from .soil import _nbr_max, _nbr_min_masked
        speed_nb = _nbr_max(np.where(world.water, hy.speed, 0.0))
        p_fall = cfg.scour_rate * np.clip(speed_nb - cfg.scour_v0, 0.0, None) * np.clip(1.0 - so.r, 0.0, 1.0) * s / cfg.scour_crit
        fall = mask & land & so.eroding & ~rock & (rnd < p_fall)
        n = int(fall.sum())
        if n:
            bed = _nbr_min_masked(hy.b, world.water)
            for y, x in zip(*np.nonzero(fall)):
                if not np.isfinite(bed[y, x]):
                    continue
                target = max(float(bed[y, x]), float(self.b_rock[y, x]))
                lost = float(hy.b[y, x]) - target
                if lost <= 0:
                    world.substrate[y, x] = ROCK
                    continue
                hy.b[y, x] = target
                world.sediment_in[y, x] += lost * cfg.soil_to_sediment
                world.flora.kind[y, x] = 0
                world.farm[y, x] = False
                world.substrate[y, x] = ROCK if target <= float(self.b_rock[y, x]) + 1e-4 else MUD
                so.wood[y, x] = 0.0
                self.soil_state[y, x] = ERODING
            so.collapsed_total += n
        return {"collapsed": n}

    # ------------------------------------------------------------------ mutation

    def tread(self, y0, x0, y1, x1, world, amount=None) -> None:
        """A crossing is one count; the level is the count in thirds (pack_per_level)."""
        H, W = self.tread_count.shape
        dy, dx = abs(y1 - y0), abs(x1 - x0)
        sy, sx = (1 if y1 >= y0 else -1), (1 if x1 >= x0 else -1)
        err = dx - dy
        y, x = y0, x0
        n = 1 if amount is None else max(1, int(round(amount / self.cfg.pack_step)))
        while True:
            if 0 <= y < H and 0 <= x < W and not world.water[y, x]:
                self.tread_count[y, x] = min(3 * self.cfg.pack_per_level, self.tread_count[y, x] + n)
            if y == y1 and x == x1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy; x += sx
            if e2 < dx:
                err += dx; y += sy
        self.pack = (self.pack_level / 3.0).astype(np.float32)

    def dig(self, y, x, world, amount) -> None:
        super().dig(y, x, world, amount)
        self.tread_count[y, x] = 0
        self._recompute_levels(world)

    def fill(self, y, x, world, amount, pack=0.3) -> None:
        world.hydro.b[y, x] += amount
        self.tread_count[y, x] = self.cfg.pack_per_level
        self._recompute_levels(world)

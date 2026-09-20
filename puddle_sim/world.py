"""Grid world: a valley through forested land, water that actually flows (hydro.py), soil that
remembers (soil.py), plants that read the fields (flora.py).

Fields are numpy arrays, one per property. Water is wherever the physics put it (depth >= d_min);
vegetation is wherever the rules let it live; biomes are what you see afterwards.
"""
from __future__ import annotations

import numpy as np

from .climate import Climate
from .config import Config
from .crops import CALTROP, DRY, NONE, RICE, TABLE as CROPS, choose as choose_crop, fitness as crop_fitness
from .flora import BARE, GRASS, MANGROVE, SWAMP, TREE, TREES, VETIVER, XERO, Flora
from .gen import MUD, ROCK, SALT, SAND, build as build_skeleton, check as check_skeleton, smooth_noise  # noqa: F401
from .ground import Ground
from .hydro import Hydro
from .soil import Soil
from .wild import Wild


def count_window(a: np.ndarray, r: int) -> np.ndarray:
    """Sum over the (2r+1)^2 neighbourhood of every cell (including itself). Bool -> count."""
    m = a.astype(np.int16) if a.dtype == bool else a
    p = np.pad(m, r)
    out = np.zeros_like(m)
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            out += p[r + dy : r + dy + m.shape[0], r + dx : r + dx + m.shape[1]]
    return out


def count3x3(a: np.ndarray) -> np.ndarray:
    return count_window(a, 1)


def nbr_max(a: np.ndarray) -> np.ndarray:
    p = np.pad(a, 1, mode="edge")
    out = np.full_like(a, -np.inf)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            out = np.maximum(out, p[1 + dy : 1 + dy + a.shape[0], 1 + dx : 1 + dx + a.shape[1]])
    return out


def chebyshev_distance_to(mask: np.ndarray, max_d: int) -> np.ndarray:
    """Chebyshev distance from every cell to the nearest True cell, capped at max_d + 1."""
    dist = np.full(mask.shape, max_d + 1, dtype=np.int16)
    dist[mask] = 0
    cur = mask.copy()
    for d in range(1, max_d + 1):
        p = np.pad(cur, 1)
        grown = np.zeros_like(cur)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                grown |= p[1 + dy : 1 + dy + cur.shape[0], 1 + dx : 1 + dx + cur.shape[1]]
        newly = grown & ~cur
        dist[newly] = d
        cur = grown
    return dist


class World:
    def __init__(self, cfg: Config, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng
        H, W = cfg.height, cfg.width
        from .crops import set_clock as _crop_clock
        _crop_clock(cfg.year_ticks)                    # D-049: yields per year -> per tick
        if cfg.basin and not cfg.sea and cfg.arid_preset:
            # A closed basin only stays a lake (not a map-wide flood) under a dry sky: little rain,
            # strong evaporation, and a river that drains a salty catchment.
            cfg.rain = min(cfg.rain, 0.0002)
            cfg.evap_mult = max(cfg.evap_mult, 1.6)
            cfg.salt_bg = max(cfg.salt_bg, 0.15)
        self.farm = np.zeros((H, W), dtype=bool)
        self.quality = np.ones((H, W), dtype=np.float32)
        self.detritus = np.zeros((H, W), dtype=np.float32)
        self.light = np.full((H, W), cfg.max_light, dtype=np.int16)
        self.salinity = np.zeros((H, W), dtype=np.float32)
        self.salt_ema = np.zeros((H, W), dtype=np.float32)   # what plants and crabs read (rides over tides)
        self.dry_ticks = np.zeros((H, W), dtype=np.int32)
        self.crab_count = np.zeros((H, W), dtype=np.int16)
        self.prey_count = np.zeros((H, W), dtype=np.int16)     # shrimp + medaka, for the beetle
        self.animal_count = np.zeros((H, W), dtype=np.int16)   # everyone, for respiration
        self.wave_exposure = np.zeros((H, W), dtype=np.float32)
        self.sheltered = np.zeros((H, W), dtype=bool)
        self.mg_built = 0
        self.algae = np.zeros((H, W), dtype=np.float32)
        self.oxygen = np.ones((H, W), dtype=np.float32)
        self.crop = np.zeros((H, W), dtype=np.int8)
        self.paddy = np.zeros((H, W), dtype=np.float32)      # water held in a rice field
        self.trapped = 0.0                                    # pollution load irrigated into paddies (D-036)
        self.shells = 0                                       # mussels dug up (D-039; lime later)
        self.bird_risk = np.zeros((H, W), dtype=np.float32)   # herons: shallow open water (D-043)
        self.eaten_by_birds = 0
        self.prev_salinity = np.zeros((H, W), dtype=np.float32)
        self.prev_depth = np.zeros((H, W), dtype=np.float32)
        self.salt_crust_total = 0
        self.tick_now = 0
        self.step_count = 0            # monotonic (tick can be -1 during spin-up); keys per-tick caches

        # Skeleton -> water to quiet -> invariants; re-roll the skeleton if the platter's promise
        # is broken (a river that does not reach the sea, a dry pond, a tributary that never joins).
        self.gen_report = {}
        self.flora = Flora(cfg, (H, W), rng)
        self.wild = Wild(cfg, (H, W))               # D-037: what the map offers a forager (readbacks, no state)
        for attempt in range(cfg.gen_attempts):
            gen_rng = np.random.default_rng(cfg.seed + 1000 * attempt)
            self.skeleton = build_skeleton(cfg, gen_rng)
            self.valley_center = self.skeleton.center
            self.pond_center = self.skeleton.pond_center
            self.hydro = Hydro(cfg, self.skeleton.b.copy(), self.skeleton.springs, self.pond_center if cfg.pond else None)
            for _ in range(cfg.spinup_ticks):          # let the river find itself first
                self.hydro.step()
            self.water = self.hydro.water
            self.water_dist = chebyshev_distance_to(self.water, cfg.water_reach)
            self.water_dist_far = chebyshev_distance_to(self.hydro.d_ema >= cfg.d_min, cfg.m_reach)
            self.substrate = self.skeleton.substrate.copy()
            self.gen_report = check_skeleton(self)
            self.gen_report["attempt"] = attempt
            if self.gen_report["ok"]:
                break
        self.coast_x0 = next((x0 for k, x0, x1 in self.skeleton.segments if k == "delta"), W)

        self.recompute_light()
        if cfg.mode == "mc":                  # D-030: the same layers as bits + events + random ticks
            from .mcmode import GroundMC
            self.ground = GroundMC(cfg, self)
        else:
            self.ground = Ground(cfg, self)   # bedrock, ground water, tread (D-029)
        self.climate = Climate(cfg, self)
        self.temp = self.climate.temp
        self.quality[:] = 1.0
        self.detritus[self.water] = 0.5
        self._farm3 = count3x3(self.farm)
        self._farm5 = count_window(self.farm, 2)

        # Let the salt find its reach before anything lives.
        for _ in range(cfg.tide_period * 3 if cfg.sea else 0):
            self.hydro.step()
            self._salt_step()
        self.water = self.hydro.water
        self.soil = Soil(cfg, (H, W), rng)
        self.sediment = np.zeros((H, W), dtype=np.float32)      # suspended, concentration per unit water
        self.sediment_in = np.zeros((H, W), dtype=np.float32)   # mass injected this tick (bank collapse)
        self.prev_water = self.water.copy()
        self.was_water_recently = self.water.copy()
        self.food_tick = 0.0
        self.food_total = 0.0
        self.food_cell = np.zeros((H, W), dtype=np.float32)
        self.grazer_count = np.zeros((H, W), dtype=np.int16)
        self.snail_count = np.zeros((H, W), dtype=np.int16)
        self.boar_count = np.zeros((H, W), dtype=np.int16)
        self.soil.bund = self.farm & (self.crop == RICE)
        self.soil.step(self)          # moisture, roots, phases once, so the plants can read them
        self.soil.phase_update()

        # Generation = the runtime rules run to quiet: scatter every kind where it fits, then step.
        self.flora.seed(self, 0)
        zeros = np.zeros((H, W), dtype=np.int16)
        for t in range(cfg.flora_spinup):
            self.step(zeros, -1)
        self.soil.h[:] = np.where(count3x3(self.tree) >= 3, np.maximum(self.soil.h, cfg.h_mid + 0.1), self.soil.h)
        self.soil.collapsed_total = 0
        self.soil.emerged_total = 0
        self.flora.built_land = 0
        self.food_total = 0.0

    # ------------------------------------------------------------------ vegetation readbacks

    @property
    def tree(self) -> np.ndarray:
        """Land trees (upland, swamp, desert riparian). Read-only view: write through flora.kind."""
        return self.flora.is_kind(*TREES)

    @property
    def mangrove(self) -> np.ndarray:
        return self.flora.kind == MANGROVE

    def canopy(self) -> np.ndarray:
        return self.flora.canopy()

    def count3x3_tree(self) -> np.ndarray:
        return count3x3(self.tree)

    def prey_count3(self) -> np.ndarray:
        return count3x3(self.prey_count)

    def crab3(self) -> np.ndarray:
        return count3x3(self.crab_count)

    # ------------------------------------------------------------------ derived fields

    def recompute_light(self) -> None:
        cfg = self.cfg
        shade = count3x3(self.canopy()) * cfg.tree_shade
        self.light = np.clip(cfg.max_light - shade, 1, cfg.max_light).astype(np.int16)

    def cover(self) -> np.ndarray:
        """0..1 — how shaded a tile is. 3 bank trees => light 6 => cover 1.0. Water caltrop shades its cell."""
        c = (self.cfg.max_light - self.light) / 9.0 + self.cfg.caltrop_shade * (self.crop == CALTROP)
        return np.clip(c, 0.0, 1.0)

    def food(self) -> np.ndarray:
        return np.clip(self.detritus / self.cfg.food_sat, 0.0, 1.0)

    def farm3(self) -> np.ndarray:
        return self._farm3

    def farm5(self) -> np.ndarray:
        return self._farm5

    def stillness(self) -> np.ndarray:
        return np.clip(1.0 - self.hydro.speed / self.cfg.still_speed, 0.0, 1.0).astype(np.float32)

    def soil_salinity(self) -> np.ndarray:
        """D-021: salt per unit of ground water (ground.py). > salt_flip means salinised ground."""
        return self.ground.salinity()

    def saline(self) -> np.ndarray:
        return (self.soil_salinity() > self.cfg.salt_flip) & ~self.water

    # ------------------------------------------------------------------ salt

    def _salt_step(self) -> None:
        cfg, hy = self.cfg, self.hydro
        if not cfg.sea and not cfg.basin:
            return
        S = hy.advect(self.salinity, edge_conc=1.0 if cfg.sea else 0.0)
        for sy, sx, _q in hy.springs:
            S[sy, sx] = max(S[sy, sx], cfg.salt_bg)
        if cfg.sea:
            # The ocean is bigger than the map (D-039): the shelf's outer columns are sea water, and
            # deep water anywhere mixes back toward it. Without this the river fills the 12-column
            # "sea" like a bathtub and the whole coast goes fresh.
            S[:, -2:] = 1.0
            deep = hy.water & (hy.b < cfg.sea_level - cfg.tide_amp - cfg.closure_depth)
            S = np.where(deep, S + cfg.ocean_mix * (1.0 - S), S)
        w = hy.water
        Sw = np.where(w, S, 0.0)
        nbr = count3x3(Sw) - Sw
        n = count3x3(w) - w.astype(np.int16)
        mean = np.where(n > 0, nbr / np.maximum(n, 1), S)
        S = S + cfg.salt_diffusion * (mean - S)
        self.salinity = np.clip(S, 0.0, 1.0).astype(np.float32)
        a = 1.0 / cfg.ema_ticks
        self.salt_ema = np.where(w, (1 - a) * self.salt_ema + a * self.salinity, self.salt_ema).astype(np.float32)


    # ------------------------------------------------------------------ per-tick physics

    def build_dam(self) -> None:
        cfg = self.cfg
        x = cfg.dam_x
        cy = int(round(self.valley_center[x]))
        y0, y1 = max(0, cy - 4), min(cfg.height, cy + 5)
        self.hydro.b[y0:y1, x] += cfg.dam_height

    def step(self, shrimp_count: np.ndarray, tick: int = 0) -> None:
        cfg = self.cfg
        hy = self.hydro
        self.step_count += 1
        if cfg.dam_tick >= 0 and tick == cfg.dam_tick:      # (the spin-up passes tick -1: it used to build the dam 1200 times)
            self.build_dam()
        if tick % 10 == 0:
            self.climate.update(self, max(tick, 0))
            self.temp = self.climate.temp
        # D-039 storm surge: while a storm is on, the sea stands higher. Over a young beach ridge it
        # goes -- into the hollows behind, salt and all, by the same pipes as everything else. What
        # cannot drain back stays, evaporates, and is a salt lagoon; dried out, a salt pan.
        hy.surge = cfg.storm_surge if (cfg.sea and self.climate.storm is not None) else 0.0
        # Rain: the ground takes its share first (infiltration), the rest is the sky's gift to the pipes.
        hy.rain_now = self.ground.split_rain(self, self.climate.rain_field)
        hy.step()
        self.ground.step(self)                # ground water moves, leaks out where the column is full
        self.water = hy.water
        self.water_dist = chebyshev_distance_to(self.water, cfg.water_reach)
        if tick % 20 == 0:
            self.water_dist_far = chebyshev_distance_to(hy.d_ema >= cfg.d_min, cfg.m_reach)
        self._salt_step()

        # Floods take fields whose crop cannot stand that much water.
        d_hi = np.zeros_like(hy.d)
        for k, c in CROPS.items():
            d_hi[self.crop == k] = c.d[1]
        flooded = self.farm & (hy.d >= d_hi + 0.2)
        if flooded.any():
            self.farm &= ~flooded
            self.crop[flooded] = NONE
        # Paddies: their water evaporates and seeps; a thirsty paddy is irrigated from the river (D-027).
        paddies = self.farm & (self.crop == RICE)
        self.paddy = np.where(paddies, np.clip(self.paddy - hy.evap_now * cfg.evap_mult * 4 - cfg.paddy_seep, 0.0, 1.0), 0.0).astype(np.float32)
        thirsty = paddies & (self.paddy < 0.5)
        if thirsty.any():
            self._irrigate(thirsty)

        w = self.water
        trees3 = count3x3(self.tree)
        farms3 = count3x3(self.farm)
        self._farm3 = farms3
        self._farm5 = count_window(self.farm, 2)
        runoff_w = np.zeros_like(hy.d)
        for k, c in CROPS.items():
            runoff_w[self.farm & (self.crop == k)] = c.runoff

        # Water quality: pollution is carried; runoff adds it, time, bank trees and algae remove it.
        q = 1.0 - np.clip(hy.advect(1.0 - self.quality), 0.0, 1.0)
        q = q - cfg.runoff_per_farm * count3x3(runoff_w)
        q = q + (cfg.quality_recover + cfg.riparian_recover * trees3 + self.flora.field("filter", adults_only=True)) * (1.0 - q)   # mussel beds filter (D-039)
        self.quality = np.clip(q, 0.0, 1.0).astype(np.float32)

        # Detritus: carried; every plant's water litter falls in (bank trees into neighbours).
        litter_w = self.flora.field("litter_water", adults_only=True)
        d = hy.advect(self.detritus)
        d = d + np.where(w, litter_w + count3x3(litter_w) * 0.5, 0.0)
        d = d * (1.0 - cfg.detritus_decay)
        d = d - cfg.detritus_eat * shrimp_count
        self.detritus = np.clip(d, 0.0, 3.0).astype(np.float32)

        # --- water chemistry (D-027): algae grow on light, warmth and nutrients; dead algae rot;
        # rotting eats oxygen; moving water and living algae put it back; animals breathe. ---
        light_f = np.clip((self.light - 6) / 9.0, 0.0, 1.0) * (1.0 - self.cfg.caltrop_shade * (self.crop == CALTROP))
        warm = self.climate.temp_factor(8.0, 38.0, soft=6.0)
        nutrients = 1.0 - self.quality
        A = hy.advect(self.algae)
        A = np.maximum(A, np.where(w, cfg.algae_bg, 0.0))
        growth = cfg.algae_grow * A * light_f * warm * (0.15 + 0.85 * nutrients)
        dead = cfg.algae_die * A
        A = A + growth - dead - cfg.algae_eat * self.grazer_count
        self.algae = np.where(w, np.clip(A, 0.0, 3.0), 0.0).astype(np.float32)
        self.quality = np.clip(self.quality + cfg.algae_uptake * growth, 0.0, 1.0).astype(np.float32)
        self.detritus = np.clip(self.detritus + dead, 0.0, 3.0).astype(np.float32)
        O = hy.advect(self.oxygen)
        O = O + cfg.o2_reaer * (1.0 - O) * (0.2 + hy.speed)
        O = O + cfg.o2_photo * self.algae * light_f
        O = O - cfg.o2_bod * (self.detritus * cfg.detritus_decay * 10.0 + dead) * (0.3 + 0.7 * warm)
        O = O - cfg.o2_resp * self.animal_count
        self.oxygen = np.where(w, np.clip(O, 0.0, 1.0), 1.0).astype(np.float32)

        # --- plants live and die by the table; dead ones leave wood (MC mode: only the sampled cells roll) ---
        if cfg.mode == "mc":
            self.flora.step(self, tick, mask=self.ground.mask, scale=self.ground.scale)
        else:
            self.flora.step(self, tick)
        self.soil.wood += self.flora.dead_wood
        self.mg_built = self.flora.built_land
        if tick % 5 == 0:
            self.recompute_light()
        if tick >= 0 and tick % 5 == 0:
            self.wild.step(self, tick)
            # birds (D-043): shallow water under open sky; a wading bird cannot fish deep or under trees
            shallow = np.clip((1.2 - hy.d_ema) / 1.0, 0.0, 1.0) * w          # a heron sees a red fish at half a metre too
            self.bird_risk = (shallow * (1.0 - self.cover())).astype(np.float32)

        # --- coast: waves gnaw at unprotected shore (transgression), dune grass builds sand ---
        salty_nb = nbr_max(np.where(w, self.salt_ema, 0.0))
        storm_on = self.climate.storm is not None
        # D-039 fetch: waves come from the open sea. A shore is exposed only if water runs east from it
        # to deep water without land in between; behind a bar, a spit, a headland or the far side of
        # the ridge it is sheltered -- no swash, no scour, mud settles, and mangroves get their place.
        if cfg.sea:
            deep = w & (hy.b < cfg.sea_level - cfg.tide_amp - cfg.closure_depth) & (self.salt_ema > 0.3)
            open_ = deep.copy()
            for x in range(w.shape[1] - 2, -1, -1):
                open_[:, x] |= w[:, x] & open_[:, x + 1]
            exposed_nb = nbr_max(open_.astype(np.float32)) > 0
        else:
            open_ = np.zeros_like(w); exposed_nb = np.ones_like(w)
        # Fair weather builds the beach, storms cut it (the summer/winter profile): waves scour
        # hard in a storm and reach higher; the rest of the time the swash band only receives.
        reach = cfg.sea_level + cfg.tide_amp + (cfg.storm_reach if storm_on else cfg.beach_top)   # how high a wave gets
        self.wave_exposure = ((~w) & (salty_nb > 0.3) & exposed_nb & (hy.b < reach)).astype(np.float32) * cfg.tide_amp * (cfg.storm_wave if storm_on else 1.0)
        self.sheltered = w & (self.salt_ema > 0.15) & ~open_          # salty water the waves do not reach (readback: where mud and mangroves belong)
        if cfg.sea and cfg.beach_rate > 0:
            # D-038 swash deposition: where a wave runs up a shore and dies it drops its sand -- on
            # the band between low tide and a little above high tide, where no current is running
            # (a river mouth keeps itself open by flowing). The sand is borrowed from the seabed
            # next door, so the beach grows and the inshore deepens: a ridge along the contour.
            H, W = hy.b.shape
            speed_nb = nbr_max(np.where(w, hy.speed_ema, 0.0))
            swash = (self.wave_exposure > 0) & (hy.b >= cfg.sea_level - cfg.tide_amp) \
                & (hy.b < cfg.sea_level + cfg.tide_amp + cfg.beach_top) & (speed_nb < cfg.scour_v0) \
                & (self.substrate != ROCK)
            if storm_on:
                swash &= False
            self.wave_exposure[swash] = 0.0                                   # no scour where the swash is dropping sand
            if swash.any():
                dep = cfg.beach_rate * cfg.tide_amp
                hy.b[swash] += dep
                self.substrate[swash] = SAND                              # waves sort: mud winnowed, sand stays
                # the sand comes from the surf zone: every salty water neighbour that still has sand
                # to give (above the shoreface floor) pays an equal share
                donor = open_ & (hy.b > cfg.sea_level - cfg.tide_amp - cfg.closure_depth)   # only the surf zone moves sand; sheltered water keeps its bed
                n_don = count3x3(donor) - donor.astype(np.int16)           # donors around each cell (not itself)
                want = np.where(swash & (n_don > 0), dep / np.maximum(n_don, 1), 0.0).astype(np.float32)
                take = np.where(donor, count3x3(want) - want, 0.0)          # what each donor owes its swash neighbours
                hy.b -= take.astype(np.float32)
        dune = self.flora.field("dune", adults_only=True)
        if dune.any():
            room = hy.b < cfg.sea_level + cfg.tide_amp + cfg.dune_max
            hy.b += (dune * room).astype(np.float32)

        # --- soil: health, phases, bank collapse (writes hydro.b and sediment_in) ---
        self.soil.bund = self.farm & (self.crop == RICE)   # the paddy's bund holds its soil
        self.soil.step(self)

        # --- sediment: injected by collapses, carried by the flow, dropped where it is slow ---
        sd = hy.advect(self.sediment)
        sd = sd + self.sediment_in / np.maximum(hy.d, cfg.d_min)
        for sy, sx, _q in hy.springs:
            sd[sy, sx] = max(sd[sy, sx], cfg.sediment_bg)
        self.sediment_in[:] = 0.0
        slow = np.clip(1.0 - hy.speed / cfg.v_dep, 0.0, 1.0)
        trap_f = self.flora.field("trap", adults_only=True)
        trap = np.maximum(1.0, np.maximum(trap_f, nbr_max(trap_f)))
        dep_frac = np.minimum(0.9, cfg.dep_rate * slow * trap) * w
        deposited = dep_frac * sd * hy.d
        hy.b += deposited.astype(np.float32)
        self.sediment = (sd * (1.0 - dep_frac)).astype(np.float32)
        # Salt stays where the water went (D-027): a drying cell with salty water leaves a crust.
        dried = self.prev_water & ~w
        if dried.any():
            mass = self.prev_salinity * self.prev_depth                                 # salt is a mass, not a colour (D-039)
            self.ground.Gs[dried] += mass[dried]                                        # the water went, the salt stayed
            crust = dried & (self.prev_salinity > 0.35) & (mass > cfg.crust_mass) & (self.substrate != ROCK)   # a surge film salts the soil; only a drying pool leaves a crust
            self.substrate[crust] = SALT
            self.salt_crust_total += int(crust.sum())
        self.prev_salinity = self.salinity.copy()
        self.prev_depth = hy.d.copy()
        # Land the river built: cells that stayed dry for a whole tidal cycle after being water.
        self.dry_ticks = np.where(w, 0, self.dry_ticks + 1).astype(np.int32)
        emerged = (self.dry_ticks == cfg.tide_period) & self.was_water_recently
        self.soil.on_emerged(emerged)
        self.substrate[emerged] = MUD
        self.was_water_recently = np.where(w, True, np.where(emerged, False, self.was_water_recently))
        self.prev_water = w.copy()

        # --- the farmer's harvest: every field yields by its crop's own fitness (crops.py) ---
        food = np.zeros_like(hy.d)
        # Insolation follows the same yearly sine as the temperature (D-036): day length x sun angle
        # makes midwinter worth `winter_sun` of midsummer for every crop, whatever the thermometer says.
        sun = cfg.winter_sun + (1.0 - cfg.winter_sun) * 0.5 * (1.0 + np.sin(2 * np.pi * self.climate.season))
        for k, c in CROPS.items():
            sel = self.farm & (self.crop == k)
            if not sel.any():
                continue
            f = crop_fitness(self, k)
            if k == RICE:
                f = f * np.clip(1.0 - cfg.snail_damage * count3x3(self.snail_count), 0.0, 1.0)
            if k == DRY:
                f = f * np.clip(1.0 - cfg.boar_damage * count3x3(self.boar_count), 0.0, 1.0)   # D-042: the raid
            food[sel] = c.base * f[sel] * sun
        self.food_cell = food.astype(np.float32)
        self.food_tick = float(self.food_cell.sum())
        self.food_total += self.food_tick

    # ------------------------------------------------------------------ mutation by agents

    def _irrigate(self, dry_paddy: np.ndarray) -> None:
        """Move water from the nearest wet cell (within 4) into each dry paddy. The river pays."""
        hy, cfg = self.hydro, self.cfg
        H, W = dry_paddy.shape
        wet = hy.d >= cfg.irrigate_min_depth
        ys, xs = np.nonzero(dry_paddy)
        for y, x in zip(ys, xs):
            y0, y1 = max(0, y - 4), min(H, y + 5)
            x0, x1 = max(0, x - 4), min(W, x + 5)
            win = wet[y0:y1, x0:x1]
            if not win.any():
                continue
            iy, ix = np.unravel_index(int(np.argmax(hy.d[y0:y1, x0:x1] * win)), win.shape)
            sy, sx = y0 + iy, x0 + ix
            take = min(cfg.irrigate_depth, float(hy.d[sy, sx]) * 0.5)
            hy.d[sy, sx] -= take
            self.paddy[y, x] += take
            # Irrigation water brings whatever the river carries: its salt into the ground (bad), and
            # upstream's runoff into the paddy's soil (good) -- the paddy is the stillest water on
            # the map, so the load settles here (D-036 nutrient trap). Same pipe, both signs.
            self.ground.G[y, x] += take * 0.5                                    # the paddy seeps into its column
            self.ground.Gs[y, x] += take * 0.5 * self.salinity[sy, sx]           # ... with the river's salt
            load = take * (1.0 - float(self.quality[sy, sx]))
            if load > 0:
                self.soil.h[y, x] = min(cfg.h_cap, self.soil.h[y, x] + cfg.trap_to_h * load)
                self.trapped += load

    def plant_cover(self, y: int, x: int) -> None:
        """A human-planted erosion hedge (護坡草): roots without a canopy; costs the cell."""
        if self.water[y, x] or self.substrate[y, x] == ROCK:
            return
        self.farm[y, x] = False
        self.crop[y, x] = NONE
        self.flora.kind[y, x] = VETIVER
        self.flora.age[y, x] = self.flora.table[VETIVER].adult_age

    def plant_tree(self, y: int, x: int) -> None:
        self.farm[y, x] = False
        self.crop[y, x] = NONE
        self.flora.kind[y, x] = TREE
        self.flora.age[y, x] = self.flora.table[TREE].adult_age
        self.recompute_light()

    def cut_tree(self, y: int, x: int) -> float:
        if not self.tree[y, x]:
            return 0.0
        k = int(self.flora.kind[y, x])
        self.flora.kind[y, x] = BARE
        self.recompute_light()
        return float(self.flora.table[k].wood)

    def lay_wood(self, y: int, x: int, mass: float) -> None:
        self.soil.wood[y, x] += mass

    def till(self, y: int, x: int, crop: int = None) -> None:
        """Farmer makes a field here (a paddy, a dry field, or a caltrop pond) and clears the canopy
        around it for light."""
        cfg = self.cfg
        if self.substrate[y, x] == ROCK:
            return
        self.farm[y, x] = True
        self.crop[y, x] = crop if crop is not None else choose_crop(self, y, x)
        if not self.water[y, x]:
            self.flora.kind[y, x] = BARE
        self.clear_canopy(y, x)

    def clear_canopy(self, y: int, x: int) -> int:
        """Fell the trees within `clear_radius` of a field (at tilling, and again on every tending
        trip — the forest grows back and a paddy needs light 12+). Returns trees felled."""
        cfg = self.cfg
        r = cfg.clear_radius
        y0, y1 = max(0, y - r), min(cfg.height, y + r + 1)
        x0, x1 = max(0, x - r), min(cfg.width, x + r + 1)
        patch = self.flora.kind[y0:y1, x0:x1]
        hit = np.isin(patch, TREES)
        n = int(hit.sum())
        if n:
            patch[hit] = BARE
            self.recompute_light()
        return n

    # ------------------------------------------------------------------ queries

    def walkable(self, amphibious: bool) -> np.ndarray:
        # Memoised per tick: every creature asked for this every tick (290k calls per 400 ticks = a
        # fifth of the whole run) and the answer only changes when the world steps.
        cache = getattr(self, "_walk_cache", None)
        if cache is None or cache["tick"] != self.step_count:
            cache = self._walk_cache = {"tick": self.step_count}
        key = amphibious if isinstance(amphibious, str) else ("amphibious" if amphibious else "water")
        m = cache.get(key)
        if m is None:
            m = cache[key] = self._walkable(key)
        return m

    def _walkable(self, medium: str) -> np.ndarray:
        if medium == "land":                                   # D-042: anything shallower than a ford, not rock
            return (self.hydro.d_ema < 0.3) & (self.substrate != ROCK)
        amphibious = medium == "amphibious"
        m = self.water | (self.hydro.d_ema >= self.cfg.mg_d_lo) if amphibious else self.water
        if amphibious:
            m = m | (self.farm & (self.crop == RICE) & (self.paddy > 0.2))   # snails walk into the paddies
        ind = getattr(self, "industries", None)
        if ind is not None:
            m = m & ~ind.blocked_water()
        return m

    def water_neighbours(self, y: int, x: int, mask: np.ndarray = None):
        H, W = self.cfg.height, self.cfg.width
        m = self.water if mask is None else mask
        out = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                ny, nx = y + dy, x + dx
                if 0 <= ny < H and 0 <= nx < W and m[ny, nx]:
                    out.append((ny, nx))
        return out

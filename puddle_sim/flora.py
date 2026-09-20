"""Flora: every plant is a row in one table; one set of rules reads the table (D-019).

A plant establishes where the fields say it can (substrate, moisture, inundation, salinity, light,
slope), spreads from neighbours (or drifts in rarely), ages, and dies when it falls outside its
band for too long, is shaded past what it tolerates, or grows old. Its effects on the world are
read back as fields: canopy (light), roots (bank stability), litter (soil h or water detritus),
sediment trapping (land building), wood on death, bed stabilisation (seagrass), and — for the
dune grass — slow sand accretion. Biomes are what you see when you look at where the rows ended up.

Kinds: 0 bare, 1 tree, 2 swamp tree, 3 mangrove, 4 salt marsh, 5 dune grass, 6 grass, 7 seagrass,
8 rock algae, 9 vetiver (planted), 10 riparian desert tree (胡楊), 11 fern (蕨, D-037: the thing that was
already eating the surplus of decomposer soil in soil.py, made visible and edible). Crops are the
farmer's, not a row here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from .config import Config
from .gen import MUD, ROCK, SAND, SOIL

BARE, TREE, SWAMP, MANGROVE, MARSH, DUNE, GRASS, SEAGRASS, ALGAE, VETIVER, XERO, FERN, CHESTNUT, PERSIMMON, MUSSEL = range(15)
TREES = (TREE, SWAMP, XERO, CHESTNUT, PERSIMMON)   # land trees: roots, canopy, wood, felled by farmers
CALTROP_LIKE = (SEAGRASS, MARSH, SWAMP)     # water vegetation a diving beetle counts as shelter


@dataclass
class Plant:
    key: str
    name: str
    sub: Dict[int, float]         # substrate weights {SOIL:..,ROCK:..,SAND:..,MUD:..}
    m: tuple = (0.0, 1.0)         # soil moisture band (land plants)
    d: tuple = (0.0, 0.0)         # inundation band on d_ema; (0,0) = land only (d_ema < d_min)
    s: tuple = (0.0, 0.25)        # salinity band (land: soil salinity; water: salt_ema)
    t: tuple = (-5.0, 40.0)       # temperature band (current temperature; long winters kill via out_ticks)
    light: int = 8                # seedlings need at least this
    light_max: int = 99           # ... and no more than this (shade plants: a fern in full sun does not establish)
    rich: bool = False            # only on decomposer soil (h > h_rich)
    speed_max: float = 99.0       # aquatic: current faster than this and it cannot hold (mussel beds)
    o2_min: float = 0.0           # aquatic: needs at least this much oxygen
    host: bool = False            # spreads only where fish are (a mussel's larvae ride on a fish's gills), not from its own neighbours
    filter: float = 0.0           # water quality recovered per tick per adult (mussels filter)
    shade_die: int = 99           # adults die if light drops below this (canopy intolerance); 99 = never
    slope: float = 9.0            # max slope
    root: float = 0.0
    canopy: bool = False
    litter_land: float = 0.0      # h repair per tick (toward h_mid) per plant in the 3x3
    litter_water: float = 0.0     # detritus per tick into the cell if it is water
    trap: float = 1.0             # deposition multiplier on the 3x3
    wood: float = 0.0             # deadwood mass on death
    p_est: float = 0.01           # establishment per tick per eligible cell with a neighbour
    p_far: float = 0.0            # ... without one
    adult_age: int = 300
    p_die: float = 0.00004
    out_ticks: int = 600          # ticks outside the band before death
    bed: float = 1.0              # scour multiplier on neighbouring banks (seagrass < 1)
    dune: float = 0.0             # sand accretion per tick (dune grass)
    aquatic: bool = False         # lives in water (uses salt_ema, not soil salinity)
    color: tuple = (0.3, 0.5, 0.3)


def default_table() -> Dict[int, Plant]:
    return {
        TREE: Plant("tree", "樹", {SOIL: 1.0, ROCK: 0.0, SAND: 0.3, MUD: 0.4}, m=(0.16, 1.0), d=(0, 0),
                    s=(0.0, 0.25), t=(2.0, 40.0), light=7, slope=1.4, root=0.6, canopy=True, litter_land=0.0006,
                    litter_water=0.015, trap=1.0, wood=1.0, p_est=0.0008, p_far=0.00002, adult_age=400,
                    p_die=0.00004, out_ticks=900, color=(0.16, 0.36, 0.20)),
        SWAMP: Plant("swamp", "沼澤林", {SOIL: 0.8, ROCK: 0.0, SAND: 0.2, MUD: 1.0}, m=(0.5, 1.0), d=(0.02, 0.5),
                     s=(0.0, 0.2), t=(4.0, 40.0), light=5, slope=0.9, root=0.8, canopy=True, litter_land=0.0004,
                     litter_water=0.02, trap=2.0, wood=0.8, p_est=0.001, p_far=0.00003, adult_age=400,
                     p_die=0.00004, out_ticks=900, color=(0.12, 0.30, 0.24)),
        MANGROVE: Plant("mangrove", "紅樹林", {SOIL: 0.4, ROCK: 0.0, SAND: 1.0, MUD: 1.5}, m=(0.0, 1.0), d=(0.03, 0.45),
                        s=(0.15, 1.0), t=(12.0, 40.0), light=8, slope=0.8, root=1.0, canopy=True, litter_water=0.02,
                        trap=8.0, wood=0.6, p_est=0.02, p_far=0.0005, adult_age=400, p_die=0.00005,
                        out_ticks=600, aquatic=True, color=(0.20, 0.55, 0.45)),
        MARSH: Plant("marsh", "鹽沼草", {SOIL: 0.3, ROCK: 0.0, SAND: 0.6, MUD: 1.5, 4: 0.6}, m=(0.0, 1.0), d=(0.01, 0.25),
                     s=(0.1, 1.0), t=(-5.0, 32.0), light=11, shade_die=10, slope=0.3, root=0.7, litter_water=0.01,
                     trap=4.0, p_est=0.04, p_far=0.001, adult_age=100, p_die=0.0002, out_ticks=300,
                     aquatic=True, color=(0.55, 0.62, 0.32)),
        DUNE: Plant("dune", "濱草", {SOIL: 0.2, ROCK: 0.0, SAND: 1.5, MUD: 0.2}, m=(0.0, 0.95), d=(0, 0),
                    s=(0.0, 0.8), t=(0.0, 40.0), light=11, shade_die=10, slope=0.5, root=0.5, litter_land=0.0002,
                    p_est=0.03, p_far=0.0008, adult_age=100, p_die=0.0002, out_ticks=300, dune=0.0003,
                    color=(0.62, 0.68, 0.40)),
        GRASS: Plant("grass", "草", {SOIL: 1.0, ROCK: 0.0, SAND: 0.6, MUD: 0.6}, m=(0.02, 0.8), d=(0, 0),
                     s=(0.0, 0.3), t=(-3.0, 40.0), light=10, shade_die=9, slope=1.2, root=0.25, litter_land=0.0002,
                     p_est=0.08, p_far=0.003, adult_age=40, p_die=0.0003, out_ticks=200,
                     color=(0.55, 0.62, 0.35)),
        SEAGRASS: Plant("seagrass", "海草", {SOIL: 0.3, ROCK: 0.0, SAND: 1.2, MUD: 1.0}, m=(0.0, 1.0), d=(0.25, 1.2),
                        s=(0.3, 1.0), t=(6.0, 34.0), light=9, slope=0.3, root=0.0, litter_water=0.015, trap=2.0,
                        p_est=0.03, p_far=0.0006, adult_age=150, p_die=0.0001, out_ticks=400, bed=0.5,
                        aquatic=True, color=(0.20, 0.50, 0.40)),
        VETIVER: Plant("vetiver", "護坡草", {SOIL: 1.0, ROCK: 0.0, SAND: 1.0, MUD: 1.0}, m=(0.0, 1.0), d=(0, 0),
                       s=(0.0, 0.6), t=(0.0, 40.0), light=6, slope=1.5, root=0.9, litter_land=0.0003,
                       p_est=0.0, p_far=0.0, adult_age=60, p_die=0.00002, out_ticks=600,
                       color=(0.70, 0.72, 0.30)),
        # 胡楊／檉柳: the desert's river tree. Lives off the water table (ground.py) — damp ground
        # only, but takes salt the upland tree cannot, and heat. Spreads along the shore, slowly.
        XERO: Plant("xero", "胡楊", {SOIL: 1.0, ROCK: 0.0, SAND: 0.8, MUD: 0.7, 4: 0.3}, m=(0.12, 1.0), d=(0, 0),
                    s=(0.0, 0.6), t=(2.0, 46.0), light=8, slope=1.2, root=0.6, canopy=True, litter_land=0.0004,
                    litter_water=0.01, trap=1.0, wood=0.6, p_est=0.0006, p_far=0.00001, adult_age=400,
                    p_die=0.00004, out_ticks=900, color=(0.45, 0.50, 0.22)),
        # 蕨 (D-037): shade, damp, decomposer soil. soil.py's `fern_eat` term has been eating the h surplus
        # above h_rich all along -- this row is its visible body. Spring shoots are food (wild.py).
        FERN: Plant("fern", "蕨", {SOIL: 1.0, ROCK: 0.0, SAND: 0.2, MUD: 0.8}, m=(0.3, 1.0), d=(0, 0),
                    s=(0.0, 0.2), t=(0.0, 30.0), light=1, light_max=9, slope=1.5, root=0.2, litter_land=0.0003,
                    rich=True, p_est=0.03, p_far=0.001, adult_age=60, p_die=0.0003, out_ticks=300,
                    color=(0.30, 0.55, 0.28)),
        # 果樹 (D-038). Two canopy trees that also feed people (wild.py reads their season):
        # 栗 chestnut -- forest edge and upland, drier ground, wants more light than the forest tree,
        #                nuts in autumn (they keep);
        # 柿 persimmon -- low warm damp ground, river terraces, fruit in late autumn (it does not keep).
        CHESTNUT: Plant("chestnut", "栗", {SOIL: 1.0, ROCK: 0.0, SAND: 0.2, MUD: 0.2}, m=(0.14, 0.65), d=(0, 0),
                        s=(0.0, 0.15), t=(0.0, 32.0), light=9, slope=1.6, root=0.6, canopy=True, litter_land=0.0006,
                        litter_water=0.01, trap=1.0, wood=1.0, p_est=0.0004, p_far=0.00002, adult_age=3000,   # a year to bear (D-051: a grove is not replaceable in a season)
                        p_die=0.00002, out_ticks=900, color=(0.30, 0.40, 0.16)),
        PERSIMMON: Plant("persimmon", "柿", {SOIL: 1.0, ROCK: 0.0, SAND: 0.1, MUD: 0.8}, m=(0.25, 0.95), d=(0, 0),
                         s=(0.0, 0.15), t=(6.0, 38.0), light=10, slope=1.0, root=0.5, canopy=True, litter_land=0.0005,
                         litter_water=0.012, trap=1.0, wood=0.8, p_est=0.0003, p_far=0.00002, adult_age=2500,
                         p_die=0.00003, out_ticks=800, color=(0.42, 0.36, 0.12)),
        # 河蚌 (D-039): sessile, so a row in this table, not a creature. Sand or mud bottom in slow fresh
        # water with oxygen; its larvae need a fish to travel on, so beds appear where fish are; adults
        # filter the water clean and armour the bed. Dug up by a forager (wild.py), shells kept.
        MUSSEL: Plant("mussel", "河蚌", {SOIL: 0.5, ROCK: 0.0, SAND: 1.0, MUD: 1.0}, m=(0.0, 1.0), d=(0.10, 1.0),
                      s=(0.0, 0.10), t=(2.0, 34.0), light=1, slope=0.6, root=0.0, litter_water=0.0,
                      trap=1.5, p_est=0.001, p_far=0.0, adult_age=3000, p_die=0.00003, out_ticks=300, bed=0.7,   # a unionid takes years to grow: a dug bed is gone for a season (D-043 ruler run 1: 13/pick x fast regrowth = 3700 food/yr, a jackpot that is not real)
                      aquatic=True, speed_max=0.6, o2_min=0.4, host=True, filter=0.0015, color=(0.45, 0.40, 0.50)),
        ALGAE: Plant("algae", "岩灘藻", {SOIL: 0.0, ROCK: 1.5, SAND: 0.0, MUD: 0.0}, m=(0.0, 1.0), d=(0.02, 0.6),
                     s=(0.3, 1.0), light=8, slope=9.0, root=0.0, litter_water=0.01, p_est=0.05,
                     p_far=0.002, adult_age=60, p_die=0.0005, out_ticks=200, aquatic=True,
                     color=(0.35, 0.45, 0.30)),
    }


class Flora:
    def __init__(self, cfg: Config, shape: tuple, rng: np.random.Generator, table: Dict[int, Plant] = None):
        self.cfg = cfg
        self.rng = rng
        self.table = table or default_table()
        H, W = shape
        self.kind = np.zeros((H, W), dtype=np.int8)
        self.age = np.zeros((H, W), dtype=np.int32)
        self.out = np.zeros((H, W), dtype=np.int32)      # ticks outside tolerance
        self.dead_wood = np.zeros((H, W), dtype=np.float32)   # wood dropped this tick (world picks it up)
        self.built_land = 0

    # ------------------------------------------------------------------ readbacks

    def is_kind(self, *kinds) -> np.ndarray:
        return np.isin(self.kind, kinds)

    def canopy(self) -> np.ndarray:
        """Adults of canopy species cast shade."""
        m = np.zeros(self.kind.shape, dtype=bool)
        for k, p in self.table.items():
            if p.canopy:
                m |= (self.kind == k) & (self.age >= p.adult_age)
        return m

    def field(self, attr: str, adults_only: bool = False) -> np.ndarray:
        out = np.zeros(self.kind.shape, dtype=np.float32)
        for k, p in self.table.items():
            v = float(getattr(p, attr))
            if v == 0.0:
                continue
            sel = self.kind == k
            if adults_only:
                sel &= self.age >= p.adult_age
            out[sel] = v
        return out

    def counts(self) -> Dict[str, int]:
        return {p.key: int((self.kind == k).sum()) for k, p in self.table.items()}

    # ------------------------------------------------------------------ rules

    def _fits(self, k: int, p: Plant, world, slope: np.ndarray, soil_sal: np.ndarray) -> np.ndarray:
        """Where this plant's bands are satisfied right now (not counting light or neighbours)."""
        sub = world.substrate
        w_sub = np.zeros(sub.shape, dtype=np.float32)
        for s_id, wgt in p.sub.items():
            w_sub[sub == s_id] = wgt
        ok = w_sub > 0
        d = world.hydro.d_ema
        if p.d == (0, 0):
            ok &= d < self.cfg.d_min
            ok &= (world.soil.m >= p.m[0]) & (world.soil.m <= p.m[1])
        else:
            ok &= (d >= p.d[0]) & (d <= p.d[1])
        sal = world.salt_ema if p.aquatic else soil_sal
        ok &= (sal >= p.s[0]) & (sal <= p.s[1])
        ok &= slope <= p.slope
        T = world.temp
        ok &= (T >= p.t[0]) & (T <= p.t[1])
        if p.rich:
            ok &= world.soil.h > self.cfg.h_rich
        if p.speed_max < 90:
            ok &= world.hydro.speed_ema <= p.speed_max
        if p.o2_min > 0:
            ok &= world.oxygen >= p.o2_min
        return ok, w_sub

    def step(self, world, tick: int, mask: np.ndarray = None, scale: float = 1.0) -> None:
        """mask/scale (MC mode, D-030): only the sampled cells roll, with probabilities x scale, and
        out-of-band death is a roll of scale/out_ticks instead of a counter."""
        cfg = self.cfg
        H, W = self.kind.shape
        gy, gx = np.gradient(world.hydro.b)
        slope = np.sqrt(gy ** 2 + gx ** 2)
        soil_sal = world.soil_salinity()
        light = world.light
        rnd = self.rng.random((H, W))
        if mask is not None:
            rnd = np.where(mask, rnd / scale, 2.0)      # rnd < p*scale  <=>  rnd/scale < p; unsampled never roll
        self.dead_wood[:] = 0.0

        empty = self.kind == BARE
        # Process kinds in priority order: canopy species claim cells over ground cover.
        order = [TREE, CHESTNUT, PERSIMMON, SWAMP, XERO, MANGROVE, MARSH, SEAGRASS, MUSSEL, ALGAE, DUNE, VETIVER, FERN, GRASS]
        new_kind = self.kind.copy()
        for k in order:
            p = self.table[k]
            fits, w_sub = self._fits(k, p, world, slope, soil_sal)
            # --- establishment on bare cells (and ground cover may be overgrown by canopy species) ---
            claimable = (new_kind == BARE) | (np.isin(new_kind, (GRASS, FERN)) & p.canopy) | ((new_kind == MARSH) & (k == MANGROVE))
            claimable &= new_kind != VETIVER
            if p.canopy:
                # Trees seed into a neglected dry field (the tending trip cuts them back); a flooded
                # paddy they cannot enter -- the water does the weeding (D-036, derived version).
                claimable &= ~(world.farm & (world.paddy >= cfg.paddy_flooded))
            else:
                claimable &= ~world.farm               # ground cover / marsh do not claim a worked field
            near = (world.prey_count3() > 0) if p.host else (self._near(k) > 0)   # glochidia ride fish, not currents
            prob = np.where(near, p.p_est, p.p_far) * w_sub
            if k == MANGROVE:
                prob = prob * np.clip(1.0 - cfg.crab_pressure * world.crab3(), 0.0, 1.0)
            est = fits & claimable & (light >= p.light) & (light <= p.light_max) & (rnd < prob)
            if not p.canopy:
                est &= world.ground.pack <= cfg.pack_grass       # a path stays bare (D-028)
            new_kind[est] = k
            self.age[est] = 0
            self.out[est] = 0
            # --- death of this kind ---
            here = self.kind == k
            outside = here & ~fits
            if mask is None:
                self.out = np.where(here, np.where(outside, self.out + 1, 0), self.out)
                out_die = here & (self.out >= p.out_ticks)
                old = here & (self.rng.random((H, W)) < p.p_die)
            else:
                # A stress counter that only advances when the cell is random-ticked (a few bits of
                # block state): transients do not kill, a season outside the band does.
                self.out = np.where(here & mask, np.where(outside, self.out + int(scale), 0), self.out)
                out_die = here & (self.out >= p.out_ticks)
                rnd2 = np.where(mask, self.rng.random((H, W)) / scale, 2.0)
                old = here & (rnd2 < p.p_die)
            if p.shade_die < 90:
                shaded = here & (light < p.shade_die) & (self.age >= p.adult_age)
                if mask is not None:
                    shaded &= mask
            else:
                shaded = np.zeros_like(here)
            if p.light_max < 90:                       # a shade plant burnt out by a clearing
                sunburnt = here & (light > p.light_max + 2) & (self.age >= p.adult_age)
                if mask is not None:
                    sunburnt &= mask
                shaded = shaded | sunburnt
            die = out_die | shaded | old
            if die.any():
                if p.wood > 0:
                    self.dead_wood[die] += p.wood
                # Succession: a mangrove that dried out leaves the land to a tree (rich peat).
                if k == MANGROVE:
                    dried = die & (world.hydro.d_ema < p.d[0])
                    if dried.any():
                        world.soil.h[dried] = cfg.h_rich + 0.2
                        new_kind[dried] = TREE
                        self.age[dried] = self.table[TREE].adult_age
                        self.built_land += int(dried.sum())
                        die &= ~dried
                new_kind[die & (new_kind == k)] = BARE
        self.kind = new_kind
        self.age = np.where(self.kind != BARE, self.age + 1, 0).astype(np.int32)

    def _near(self, k: int) -> np.ndarray:
        m = (self.kind == k).astype(np.int16)
        p = np.pad(m, 2)
        out = np.zeros_like(m)
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                out += p[2 + dy: 2 + dy + m.shape[0], 2 + dx: 2 + dx + m.shape[1]]
        return out

    # ------------------------------------------------------------------ seeding (generation = rules to quiet)

    def seed(self, world, ticks: int) -> None:
        """Generation: scatter a few of every kind where they fit, then run the rules to quiet."""
        cfg = self.cfg
        gy, gx = np.gradient(world.hydro.b)
        slope = np.sqrt(gy ** 2 + gx ** 2)
        soil_sal = world.soil_salinity()
        for k, p in self.table.items():
            fits, w_sub = self._fits(k, p, world, slope, soil_sal)
            cand = np.argwhere(fits & (self.kind == BARE))
            if len(cand) == 0:
                continue
            n = min(len(cand), max(6, int(len(cand) * (0.25 if k in (TREE, GRASS) else 0.03))))
            if k == XERO and world.cfg.rain > 0.0003:
                n = max(3, n // 8)                              # a wet map seeds few; the upland tree wins there
            picks = cand[self.rng.choice(len(cand), size=n, replace=False)]
            self.kind[picks[:, 0], picks[:, 1]] = k
            self.age[picks[:, 0], picks[:, 1]] = p.adult_age
        world.recompute_light()
        for t in range(ticks):
            self.step(world, t)
            if t % 20 == 0:
                world.recompute_light()
        world.recompute_light()

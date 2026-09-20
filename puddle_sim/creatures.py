"""Creatures. Two in so far: the shrimp (蝦子) and the medaka (鱂魚, 稻田魚).

Behaviour skeleton, lifted from how MC keeps mobs cheap:
- evaluate habitat only every `eval_every` ticks (MC sensors: every 20), staggered per creature
- unhappy -> look up the precomputed score map within `search_radius` (the 'POI index'),
  move toward the best tile only if it beats the current one by `hysteresis`,
  and claim it on the index (MC POI ticket) so the herd spreads instead of piling up
- nothing better nearby -> wander (random step with `wander_prob`, MC: 1/120 per tick)
- resource output per tick = yield_rate * satisfaction  (linear, D-003.3)

A species is a name plus a habitat function over the world (0..1 per water tile, Liebig-style:
any factor at zero zeroes the tile). Movement / breeding / starvation rules are shared.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from .config import Config
from .life import LifeTable
from .world import World, count3x3, count_window

Pos = Tuple[int, int]


def _geo(factors: List[Tuple[np.ndarray, float]]) -> np.ndarray:
    """Weighted geometric mean of (factor, weight) pairs. Weights need not sum to 1."""
    tot = sum(w for _, w in factors)
    out = np.ones_like(factors[0][0], dtype=np.float32)
    for f, w in factors:
        out = out * np.power(np.clip(f, 0.0, 1.0), w)
    return out ** (1.0 / tot)


# ---------------------------------------------------------------------------- species

@dataclass
class Species:
    key: str
    name: str
    marker: str
    n_start: int
    habitat: Callable[[World], np.ndarray]   # 0..1 on water tiles (crowding added later)
    eats_detritus: bool = False
    eats_algae: bool = False
    predator: bool = False                   # presses on the breeding of shrimp/medaka around it
    pressure: float = 0.2                    # ... by this much per individual in the 3x3 (D-041: a carp is not a beetle)
    amphibious: bool = False                 # may stand on the tidal mudflat, not only in water
    terrestrial: bool = False                # lives on land (D-042 boar): walks anything shallower than a ford, not rock
    mass: float = 1.0                        # D-032: body mass, shrimp = 1
    strategy: float = 0.0                    # D-032: -1 (r) .. +1 (K)
    egg: bool = False                        # egg-layer: tiny births, many of them
    passive: str = ""                        # the passive product's name ("" = none)
    food: float = 0.0                        # food units in one grown animal (D-043: by kcal, 1 unit = 33 g rice; a boar is 60 kg, not 40 shrimp)
    immigrates: bool = True                  # a near-extinct population gets newcomers (eggs on birds' feet). Not a boar: nothing flies it in
    genes: tuple = ()                        # heritable traits, each 0..1 (D-043): ("color",) for the carp
    wild_genes: tuple = ()                   # the wild mean of each gene (a founder is born near it)

    @property
    def medium(self) -> str:
        return "land" if self.terrestrial else ("amphibious" if self.amphibious else "water")

    def life(self, cfg: Config) -> LifeTable:
        lt = getattr(self, "_life", None)
        if lt is None:
            lt = LifeTable.derive(cfg, self.mass, self.strategy, self.egg)
            object.__setattr__(self, "_life", lt)
        return lt


def _o2(w: World, need: float) -> np.ndarray:
    """0..1 — oxygen relative to what this animal needs (full at `need`, zero at half of it)."""
    return np.clip((w.oxygen - 0.5 * need) / (0.5 * need + 1e-6), 0.0, 1.0)


def _depth(w: World, lo: float, hi: float, soft: float = 0.05) -> np.ndarray:
    """0..1 — how far inside the depth band (lo..hi) the tide-smoothed depth is (D-028 ③). A film
    counts as water for the map but not for an animal that needs room to swim."""
    d = w.hydro.d_ema
    return np.clip(np.minimum((d - lo) / soft + 1.0, (hi - d) / soft + 1.0), 0.0, 1.0)


def shrimp_habitat(w: World) -> np.ndarray:
    """蝦子: clean, oxygenated water, shaded banks, leaf litter to eat, not too cold. All, or nothing."""
    return _geo([(w.quality, 0.50), (w.cover(), 0.25), (w.food(), 0.25), (w.climate.temp_factor(4.0, 32.0), 0.3),
                 (_o2(w, 0.55), 0.5), (_depth(w, 0.15, 1.5), 0.4)])


def medaka_habitat(w: World) -> np.ndarray:
    """鱂魚 (稻田魚): sun on the water, still water, insects — which paddies and sunlight breed.
    Tolerates dirty water (satisfied from quality 0.4 up)."""
    sun = 1.0 - w.cover()
    q_tol = np.clip(w.quality / 0.4, 0.0, 1.0)
    paddy = np.clip(w.farm5() / 8.0, 0.0, 1.0)         # farmland within 2 tiles: insects fly
    # A wild baseline of insects everywhere; sun and paddies multiply it. Marginal before farming,
    # blooming with it — that is the role: the creature the farm breeds.
    insects = np.clip(0.3 + 0.3 * sun + 0.5 * paddy, 0.0, 1.0)
    still = np.clip(0.3 + w.stillness(), 0.0, 1.0)
    return _geo([(q_tol, 0.20), (sun, 0.30), (insects, 0.30), (still, 0.20), (w.climate.temp_factor(8.0, 36.0), 0.3),
                 (_o2(w, 0.3), 0.4), (_depth(w, 0.05, 0.6), 0.4)])


def crab_habitat(w: World) -> np.ndarray:
    """螃蟹: shallow brackish water among mangroves, eating litter. Eats mangrove propagules too
    (that side lives in World._mangrove_step via crab_count)."""
    brackish = np.clip(w.salt_ema / 0.3, 0.0, 1.0)
    from .world import count_window
    stand = np.clip(count_window(w.mangrove, 2) / 4.0, 0.0, 1.0)
    return _geo([(_depth(w, 0.0, 0.5, soft=0.1), 0.25), (brackish, 0.30), (0.2 + 0.8 * stand, 0.30), (w.food(), 0.15),
                 (w.climate.temp_factor(12.0, 36.0), 0.3)])


def snail_habitat(w: World) -> np.ndarray:
    """蘋果螺: still, warm, shallow water with something to eat (litter or algae); shrugs at low
    oxygen and dirty water. A paddy is heaven — and it eats the rice (crops.py)."""
    paddy = w.farm & (w.crop == 3 - 1)          # RICE
    food = np.clip(w.food() + w.algae / 0.5 + 0.8 * paddy, 0.0, 1.0)
    fresh = np.clip((0.3 - w.salt_ema) / 0.2, 0.0, 1.0)
    return _geo([(0.2 + 0.8 * w.stillness(), 0.3), (food, 0.4), (_depth(w, 0.03, 0.8), 0.3), (fresh, 0.3),
                 (w.climate.temp_factor(12.0, 38.0), 0.3), (_o2(w, 0.15), 0.2)])


def beetle_habitat(w: World) -> np.ndarray:
    """龍蝨: has opinions. Clean, oxygen-rich, still, vegetated fresh water, warm but not hot,
    prey around. Every factor counts, so it is the first to leave."""
    from .flora import CALTROP_LIKE
    veg = np.clip((count_window(w.flora.is_kind(*CALTROP_LIKE), 1) + (w.crop == 3)) / 2.0, 0.0, 1.0)
    prey = np.clip(count_window(w.prey_count, 2) / 6.0, 0.0, 1.0)
    fresh = np.clip((0.15 - w.salt_ema) / 0.1, 0.0, 1.0)
    return _geo([(w.quality, 1.0), (_o2(w, 0.75), 1.0), (0.1 + 0.9 * w.stillness(), 0.6), (0.2 + 0.8 * veg, 0.6),
                 (0.2 + 0.8 * prey, 0.5), (fresh, 0.5), (w.climate.temp_factor(12.0, 30.0), 0.5), (_depth(w, 0.1, 0.6), 0.5)])


def carp_habitat(w: World) -> np.ndarray:
    """鯽魚 (D-041): the pond fish. Needs room (depth), likes it still and fresh, shrugs at murk and low
    oxygen, eats what is on the bottom and whatever small thing swims past. Where it lives, shrimp
    and medaka breed less (Species.pressure) -- the first predator above the canary."""
    fresh = np.clip((0.15 - w.salt_ema) / 0.1, 0.0, 1.0)
    q_tol = np.clip(w.quality / 0.35, 0.0, 1.0)
    prey = np.clip(count_window(w.prey_count, 2) / 8.0, 0.0, 1.0)
    food = np.clip(0.25 + 0.35 * w.food() + 0.25 * np.clip(w.algae / 0.6, 0.0, 1.0) + 0.4 * prey, 0.0, 1.0)
    return _geo([(_depth(w, 0.30, 1.5, soft=0.1), 0.6), (0.2 + 0.8 * w.stillness(), 0.4), (fresh, 0.5), (q_tol, 0.2),
                 (food, 0.4), (w.climate.temp_factor(6.0, 34.0), 0.3), (_o2(w, 0.25), 0.3)])


def boar_habitat(w: World) -> np.ndarray:
    """野豬 (D-042): the first land animal. Wants cover to lie up in, something to dig for (roots in
    decomposer soil, fallen nuts and fruit in autumn, a dry field within a raid), water near, and
    no people -- a trodden path is a road, and roads mean hunters."""
    from .wild import NUT, FRUIT
    cfg = w.cfg
    cover = np.clip(count_window(w.tree, 1) / 6.0, 0.0, 1.0)
    roots = np.clip((w.soil.h - cfg.h_min) / (cfg.h_rich - cfg.h_min), 0.0, 1.0)
    mast = np.clip(count_window(np.isin(w.wild.kind, (NUT, FRUIT)), 2) / 3.0, 0.0, 1.0)
    crops = np.clip(count_window(w.farm & (w.crop == 1), 2) / 4.0, 0.0, 1.0)          # DRY fields: the raid
    food = np.clip(0.15 + 0.4 * roots + 0.5 * mast + 0.6 * crops, 0.0, 1.0)
    near_water = np.clip(1.0 - (w.water_dist_far - 2) / 8.0, 0.0, 1.0)
    quiet = 1.0 - np.clip((w.ground.pack - cfg.pack_grass) / (1.0 - cfg.pack_grass), 0.0, 1.0)
    return _geo([(0.2 + 0.8 * cover, 0.5), (food, 0.6), (near_water, 0.3), (0.3 + 0.7 * quiet, 0.3),
                 (w.climate.temp_factor(-5.0, 35.0), 0.3)])


def brine_habitat(w: World) -> np.ndarray:
    """鹵蟲: the salt lake's own. Hypersaline, still, warm, sunlit shallow water; nothing else lives there."""
    return _geo([(np.clip((w.salt_ema - 0.2) / 0.3, 0.0, 1.0), 1.0), (0.3 + 0.7 * w.stillness(), 0.4),
                 (_depth(w, 0.03, 1.0), 0.3), (w.climate.temp_factor(10.0, 40.0), 0.4),
                 (np.clip((w.light - 8) / 6.0, 0.0, 1.0), 0.3)])


SPECIES: Dict[str, Species] = {
    "shrimp": Species("shrimp", "蝦子", "o", 24, shrimp_habitat, eats_detritus=True, mass=1.0, strategy=-0.3, egg=True, passive="eggs", food=0.1),
    "medaka": Species("medaka", "鱂魚", "^", 24, medaka_habitat, mass=0.5, strategy=-0.6, egg=True, passive="eggs", food=0.05),
    "crab": Species("crab", "螃蟹", "s", 12, crab_habitat, eats_detritus=True, amphibious=True, mass=10.0, strategy=-0.5, egg=True, passive="shell", food=0.5),   # big body, r-strategy: thousands of larvae, most die
    "snail": Species("snail", "蘋果螺", "v", 24, snail_habitat, eats_detritus=True, eats_algae=True, amphibious=True, mass=2.0, strategy=-0.2, egg=True, food=0.1),
    "beetle": Species("beetle", "龍蝨", "D", 12, beetle_habitat, predator=True, pressure=0.2, mass=0.3, strategy=0.0, egg=True),
    "carp": Species("carp", "鯽魚", "f", 8, carp_habitat, eats_detritus=True, eats_algae=True, predator=True, pressure=0.5, mass=6.0, strategy=0.0, egg=True, food=2.0, genes=("color",), wild_genes=(0.05,)),
    "brine": Species("brine", "鹵蟲", "p", 24, brine_habitat, mass=0.01, strategy=-1.0, egg=True, passive="cysts"),
    "boar": Species("boar", "野豬", "B", 6, boar_habitat, terrestrial=True, mass=40.0, strategy=-0.8, egg=False, food=375.0, immigrates=False),
}


# ---------------------------------------------------------------------------- creature

class Creature:
    _next_id = 0

    def __init__(self, cfg: Config, species: Species, pos: Pos, tick: int, rng: np.random.Generator):
        self.cfg = cfg
        self.species = species
        self.id = Creature._next_id
        Creature._next_id += 1
        self.pos: Pos = pos
        self.target: Optional[Pos] = None
        self.satisfaction: float = 0.5
        self.starving: int = 0
        self.stranded: int = 0
        self.phase = int(rng.integers(0, cfg.eval_every))  # stagger evaluations
        lt = species.life(cfg)
        # Stagger breeding too, or everyone breeds on the same tick (a burst, not a population).
        self.last_breed: int = tick - int(rng.integers(0, lt.generation))
        self.alive = True
        # D-032 life: born small, grow by von Bertalanffy on satisfaction-weighted time, die of age.
        self.born: int = tick
        self.t_eff: float = float(-np.log(1.0 - lt.birth_frac ** (1.0 / 3.0)) / lt.k)   # effective age at birth size
        self.size: float = lt.size_at(self.t_eff)
        self.sat_ema: float = 0.5
        self.stock: float = 0.0            # passive product waiting to be collected
        self.pen: bool = False             # fenced in (D-032 tier 2)
        self.lifespan: int = int(lt.lifespan * (0.8 + 0.4 * rng.random()))
        self.lifespan0 = self.lifespan
        # D-043 heredity: a founder sits near the wild mean; children get their genes from inherit().
        self.genes = {g: float(np.clip(m + rng.normal(0.0, cfg.mutation), 0.0, 1.0)) for g, m in zip(species.genes, species.wild_genes)}
        self._apply_genes()

    def _apply_genes(self) -> None:
        """What the genes do to the body. color: conspicuous to birds (step) and frail (here)."""
        c = self.genes.get("color", 0.0)
        self.lifespan = int(self.lifespan0 * (1.0 - self.cfg.color_frailty * c))

    def inherit(self, mate: "Creature", rng: np.random.Generator) -> None:
        """Average of the two parents plus mutation (mate may be the parent itself: selfing)."""
        cfg = self.cfg
        for g in self.genes:
            mid = 0.5 * (mate.genes.get(g, 0.0) + self._parent_genes.get(g, 0.0))
            v = mid + rng.normal(0.0, cfg.mutation)
            if rng.random() < cfg.mutation_major_p:                   # a major gene flips
                v += cfg.mutation_major * (1.0 if rng.random() < 0.7 else -1.0)
            self.genes[g] = float(np.clip(v, 0.0, 1.0))
        self._apply_genes()

    # ------------------------------------------------------------------ per tick

    def step(self, tick: int, world: World, score: np.ndarray, rng: np.random.Generator) -> float:
        """Advance one tick. Returns resource produced this tick."""
        cfg = self.cfg

        # The water moved: a creature on a dry tile hops to a wet neighbour, or waits for the tide;
        # too long on dry ground and it is stranded.
        mask = world.walkable(self.species.medium)
        pen_mask = getattr(world, "pen_mask", None)
        if pen_mask is not None:
            mask = mask & (pen_mask if self.pen else ~pen_mask)
        if not mask[self.pos]:
            nb = world.water_neighbours(*self.pos, mask=mask)
            if nb:
                self.pos = nb[int(rng.integers(len(nb)))]
                self.stranded = 0
            else:
                self.stranded += 1
                if self.stranded > cfg.strand_ticks:
                    self.alive = False
                return 0.0
        else:
            self.stranded = 0

        if (tick + self.phase) % cfg.eval_every == 0:
            self._evaluate(world, score)

        if self.target is not None:
            self._step_toward_target(world)
        elif rng.random() < cfg.wander_prob:
            self._wander(world, rng)

        if self.satisfaction < cfg.starve_threshold:
            self.starving += 1
            if self.starving > cfg.starve_ticks:
                self.alive = False
        else:
            self.starving = 0

        # D-043 birds: a bright fish in open shallow water gets eaten. The risk field is the world's
        # (shallow x no cover), the exposure is the gene's, and a roofed pen has no sky.
        c = self.genes.get("color", 0.0) if self.genes else 0.0
        if c > 0.0 and not (self.pen and cfg.pen_roof):
            risk = world.bird_risk[self.pos]
            if risk > 0 and rng.random() < cfg.bird_base * risk * (0.15 + c):
                self.alive = False
                world.eaten_by_birds += 1
                return 0.0

        # Life (D-032): grow on good days, age every day, die when the clock says — sooner if life was bad.
        lt = self.species.life(cfg)
        self.sat_ema = 0.995 * self.sat_ema + 0.005 * self.satisfaction
        self.t_eff += self.satisfaction
        self.size = lt.size_at(self.t_eff)
        if tick - self.born > self.lifespan * (0.5 + 0.5 * self.sat_ema):
            self.alive = False
        # Passive product: only a content, grown animal lays / moults / gives.
        produced = 0.0
        if self.species.passive and self.satisfaction >= cfg.passive_min_sat and lt.is_mature(self.size):
            produced = cfg.yield_rate * self.satisfaction * (self.size / lt.mass)
            self.stock += produced
        return produced

    # ------------------------------------------------------------------ behaviour pieces

    def _evaluate(self, world: World, score: np.ndarray) -> None:
        cfg = self.cfg
        y, x = self.pos
        self.satisfaction = float(max(score[y, x], 0.0))
        if self.satisfaction >= cfg.move_threshold:
            self.target = None
            return
        r = cfg.search_radius
        y0, y1 = max(0, y - r), min(cfg.height, y + r + 1)
        x0, x1 = max(0, x - r), min(cfg.width, x + r + 1)
        win = score[y0:y1, x0:x1]
        iy, ix = np.unravel_index(int(np.argmax(win)), win.shape)
        best = float(win[iy, ix])
        if best > self.satisfaction + cfg.hysteresis:
            ty, tx = y0 + int(iy), x0 + int(ix)
            self.target = (ty, tx)
            # Claim it (MC POI "ticket"): knock the index down around the target right now.
            score[max(0, ty - 1) : ty + 2, max(0, tx - 1) : tx + 2] -= cfg.crowd_penalty
        else:
            self.target = None  # nothing better around -> just wander

    def _step_toward_target(self, world: World) -> None:
        ty, tx = self.target
        y, x = self.pos
        if (y, x) == (ty, tx):
            self.target = None
            return
        best, best_d = None, (ty - y) ** 2 + (tx - x) ** 2
        for ny, nx in world.water_neighbours(y, x, mask=self._mask(world)):
            d = (ty - ny) ** 2 + (tx - nx) ** 2
            if d < best_d:
                best, best_d = (ny, nx), d
        if best is None:
            self.target = None  # blocked -> give up, wander
            return
        self.pos = best

    def _mask(self, world: World) -> np.ndarray:
        m = world.walkable(self.species.medium)
        pen_mask = getattr(world, "pen_mask", None)
        if pen_mask is not None:
            m = m & (pen_mask if self.pen else ~pen_mask)
        return m

    def _wander(self, world: World, rng: np.random.Generator) -> None:
        nb = world.water_neighbours(*self.pos, mask=self._mask(world))
        if nb:
            self.pos = nb[int(rng.integers(len(nb)))]

    def can_breed(self, tick: int) -> bool:
        cfg = self.cfg
        lt = self.species.life(cfg)
        return (self.satisfaction >= cfg.breed_threshold and lt.is_mature(self.size)
                and tick - self.last_breed >= lt.generation)

    def litter(self, rng: np.random.Generator) -> int:
        """Offspring this brood: the life table's expectation x satisfaction, stochastically rounded."""
        n = self.species.life(self.cfg).litter * self.satisfaction
        return int(n) + (1 if rng.random() < n - int(n) else 0)


def species_score(cfg: Config, world: World, species: Species, occupancy: np.ndarray) -> np.ndarray:
    """Habitat index for one species: its habitat function minus crowding by its own kind."""
    s = species.habitat(world)
    if cfg.crowd_penalty > 0:
        crowd = cfg.crowd_penalty * count3x3(occupancy.astype(np.float32))
        pen = getattr(world, "pen_mask", None)
        if pen is not None and pen.any():
            crowd = np.where(pen, crowd / cfg.pen_density_mult, crowd)   # fed fish do not compete for the pond's food (D-043)
        s = s - crowd
    return np.where(world.walkable(species.medium), np.clip(s, 0.0, 1.0), -1.0).astype(np.float32)


def spawn(cfg: Config, world: World, species: Species, rng: np.random.Generator, n: int, tick: int = 0) -> List[Creature]:
    """Drop n creatures where their kind can actually live (top half of their habitat score)."""
    mask = world.walkable(species.medium)
    score = np.where(mask, species.habitat(world), -1.0)
    flat = np.argsort(score.ravel())[::-1][: max(n * 3, 40)]      # the best few dozen cells
    flat = flat[score.ravel()[flat] > 0]
    tiles = np.array(np.unravel_index(flat, score.shape)).T
    if len(tiles) == 0:
        tiles = np.argwhere(mask)
    picks = tiles[rng.choice(len(tiles), size=min(n, len(tiles)), replace=False)]
    out = []
    lt = species.life(cfg)
    for y, x in picks:
        c = Creature(cfg, species, (int(y), int(x)), tick, rng)
        # A founding population has all ages, not a nursery.
        age = int(rng.random() * lt.lifespan * 0.6)
        c.born = tick - age
        c.t_eff += age * 0.7
        c.size = lt.size_at(c.t_eff)
        out.append(c)
    return out

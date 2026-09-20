"""Scripted player: 'normal agriculture'. Not an AI — a policy we can reason about.

blind:    every `expand_every` ticks till one more tile — the closest untilled land tile to home
          within `water_reach` of water (MC farmland rule) and farther than `buffer` from water.
          Tilling clears trees in `clear_radius` so the crop gets light >= crop_min_light.
adaptive: same, but watches one species' yield (EMA). When it falls `drop_tol` below its peak,
          stop expanding and instead put a tree back on the bank tile nearest the newest field
          (retiring that field if it was one). Resume when the yield recovers halfway.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

from .config import Config
from .store import Store, WILD_KIND
from .world import World


class Farmer:
    def __init__(self, cfg: Config, world: World, rng: np.random.Generator, home: tuple = None, name: str = "player"):
        self.cfg = cfg
        self.world = world
        self.rng = rng
        self.name = name
        self.index = 0
        self.home = home if home is not None else self._pick_home()
        self.food_total = 0.0
        self.ledger = {"wood": 0.0, "food": 0.0, "salt": 0.0, "charcoal": 0.0, "power": 0.0}
        self.built: List[Tuple[int, int, int]] = []
        self.industries = None        # set by the sim
        self.fields: List[Tuple[int, int]] = []
        self.paddies: List[Tuple[int, int]] = []   # fields opened as paddies (rotation switches their crop by season)
        # D-043 step 6: labour. Every trip is walked and takes time; food is one number whatever it came from.
        self.tick = 0
        self.busy_until = 0
        self.walked = {"till": 0.0, "tend": 0.0, "wood": 0.0, "forage": 0.0, "hunt": 0.0, "industry": 0.0, "other": 0.0}
        self.foraged = {}                # wild kind name -> food units picked
        self.hunted_food = 0.0           # meat x edible
        self.store = Store(cfg.ticks_per_day)   # D-044 step 1: the larder; everything edible lands here and rots or is eaten
        # D-050 labour: hours flow in every tick (people x work_hours a day), jobs take them out, nothing is saved past a day.
        self.hours_bank = 0.0
        self.hours_by = {}               # kind -> hours worked
        self.tend_stamp = {}             # field -> tick last weeded
        self.last_hunt = -10 ** 9
        self.protected: set = set()   # tiles the adaptive farmer replanted: never till them again
        self.replanted = 0
        self.moved = 0.0
        self.exhausted = False
        # adaptive state
        self.watch_ema = 0.0
        self.watch_peak = 0.0
        self.holding = False

    def _pick_home(self) -> tuple:
        """A land tile next to water, near the centre of the map."""
        w = self.world
        H, W = self.cfg.height, self.cfg.width
        from .gen import ROCK
        cand = np.argwhere(~w.water & (w.water_dist == 1) & (w.substrate != ROCK))
        centre = np.array([H / 2, W / 2])
        d = np.linalg.norm(cand - centre, axis=1)
        y, x = cand[int(np.argmin(d))]
        return int(y), int(x)

    # ------------------------------------------------------------------ per tick

    def observe(self, watched_yield: float) -> None:
        """Adaptive farmer's only sense: one number per tick."""
        cfg = self.cfg
        self.watch_ema = (1 - cfg.ema) * self.watch_ema + cfg.ema * watched_yield
        self.watch_peak = max(self.watch_peak, self.watch_ema)

    # ------------------------------------------------------------------ labour (D-050)

    def busy(self, tick: int) -> bool:
        """Kept for callers: with an hours budget nobody is 'busy', they are out of hours."""
        return False

    def walk_hours(self, y: int, x: int, trips: int = 1) -> float:
        d = max(abs(int(y) - self.home[0]), abs(int(x) - self.home[1]))
        return 2.0 * d * trips * self.cfg.cell_m / 1000.0 / self.cfg.walk_kmh

    def can(self, hours: float) -> bool:
        return self.hours_bank >= hours

    def spend(self, hours: float, kind: str) -> None:
        self.hours_bank -= hours
        self.hours_by[kind] = self.hours_by.get(kind, 0.0) + hours

    def hours_total(self) -> float:
        return float(sum(self.hours_by.values()))

    def farms_on(self) -> bool:
        return self.cfg.expand_every < 10 ** 6

    def step(self, tick: int) -> None:
        cfg = self.cfg
        self.tick = tick
        # The larder rots and the household eats whether or not anyone is home (D-044 step 1).
        self.store.step(float(self.world.temp[self.home]), cfg.people * cfg.eat_per_person, cfg.decay_q10, cfg.decay_ref_t,
                        ema_ticks=cfg.ration_days * cfg.ticks_per_day)
        # Hours arrive; a day's worth at most is kept (you cannot bank sleep). A hungry household
        # works less -- nobody dies, the fields go to bush (D-051: the pressure lands on what you tend).
        strength = cfg.hunger_floor + (1.0 - cfg.hunger_floor) * float(np.clip(self.store.ration_ema, 0.0, 1.0))
        self.strength_min = min(getattr(self, "strength_min", 1.0), strength)   # the leanest day, for the ledger
        day_hours = cfg.people * cfg.work_hours * strength
        self.hours_bank = min(day_hours, self.hours_bank + day_hours / cfg.ticks_per_day)
        # Jobs by priority; each takes hours; the day ends when they run out. Opening new land is
        # last -- it is what spare labour does, and tending is what caps the farm (D-050).
        self._job_tend()
        if cfg.rotate and self.paddies:
            self._rotate()
        if cfg.forage_every > 0:
            self._job_forage()
        if cfg.harvest_every > 0:
            self._job_wood()
        if cfg.industries and self.industries is not None and tick > 0:
            self._job_industry()
        if cfg.feed_wood_every > 0 and tick % cfg.feed_wood_every == 0:
            self._feed_wood()
        if cfg.move_soil_every > 0 and tick % cfg.move_soil_every == 0:
            self._move_soil()
        if not self.farms_on():
            return
        if cfg.farmer_mode == "adaptive" and self.watch_peak > 0:
            if self.watch_ema < (1 - cfg.drop_tol) * self.watch_peak:
                self.holding = True
            elif self.watch_ema > (1 - cfg.drop_tol / 2) * self.watch_peak:
                self.holding = False
            if self.holding:
                self._replant_bank()
                return
        if self.can(cfg.hours_till):
            self._expand()

    def _job_tend(self) -> None:
        """Weed the block that has waited longest, if any field has waited more than tend_days.
        Cost = hours per field in the block + the walk. A farm bigger than a day's weeding goes to
        bush at the edges -- that is the cap on how much land one household can hold."""
        w, cfg = self.world, self.cfg
        if not self.fields:
            return
        due = cfg.tend_days * cfg.ticks_per_day
        oldest, oldest_t = None, None
        for f in self.fields:
            if not w.farm[f]:
                continue
            t0 = self.tend_stamp.get(f, self.tick - due - 1)
            if self.tick - t0 >= due and (oldest_t is None or t0 < oldest_t):
                oldest, oldest_t = f, t0
        if oldest is None:
            return
        fy, fx = oldest
        block = [(gy, gx) for (gy, gx) in self.fields
                 if w.farm[gy, gx] and abs(gy - fy) <= cfg.tend_radius and abs(gx - fx) <= cfg.tend_radius]
        cost = cfg.hours_tend_field * len(block) + self.walk_hours(fy, fx)
        if not self.can(cost):
            return
        self.spend(cost, "tend")
        self._walk(fy, fx, kind="tend")
        for (gy, gx) in block:
            self.ledger["wood"] += w.clear_canopy(gy, gx) * cfg.tree_wood * 0.5   # saplings, half a tree
            self.tend_stamp[(gy, gx)] = self.tick

    def _job_forage(self) -> None:
        cfg = self.cfg
        if self.tick % max(1, cfg.forage_every) != 0:
            return
        if not self.can(cfg.hours_forage):
            return
        if self._forage():
            self.spend(cfg.hours_forage, "forage")

    def _job_wood(self) -> None:
        cfg = self.cfg
        if self.ledger["wood"] >= cfg.wood_target:
            return
        if self.tick % max(1, cfg.harvest_every) != 0:
            return
        if not self.can(cfg.hours_wood):
            return
        before = self.ledger["wood"]
        self._harvest_wood()
        if self.ledger["wood"] > before:
            self.spend(cfg.hours_wood, "wood")

    def _job_industry(self) -> None:
        cfg = self.cfg
        if self.tick % cfg.build_every != 0:
            return
        if not self.can(cfg.hours_industry):
            return
        from .industry import KILN, SALTPAN, WHEEL
        have = {k for (_, _, k) in self.built}
        for kind in (WHEEL, SALTPAN, KILN):          # one of each, in this order of want
            if kind not in have:
                if self.industries.try_build(self.world, self, kind, cfg.industry_radius):
                    self.spend(cfg.hours_industry, "industry")
                    break

    # ------------------------------------------------------------------ actions

    def _walk(self, y: int, x: int, trips: int = 1, kind: str = "other") -> None:
        """Every job is a trip from home and back; the ground remembers (D-028 tread), the ledger
        remembers the distance, and the clock runs while you are out (D-043 step 6)."""
        g = self.world.ground
        for _ in range(trips):
            g.tread(self.home[0], self.home[1], int(y), int(x), self.world)
        d = max(abs(int(y) - self.home[0]), abs(int(x) - self.home[1]))
        self.walked[kind] = self.walked.get(kind, 0.0) + 2.0 * d * trips

    def _forage(self) -> bool:
        """Walk to the best ripe thing within reach (yield per cell of walking), pick it, come home.
        Returns True if a trip was made."""
        w, cfg = self.world, self.cfg
        from .wild import YIELD, NAMES
        wild = w.wild.kind
        r = cfg.forage_radius
        hy, hx = self.home
        y0, y1 = max(0, hy - r), min(cfg.height, hy + r + 1)
        x0, x1 = max(0, hx - r), min(cfg.width, hx + r + 1)
        win = wild[y0:y1, x0:x1]
        ys, xs = np.nonzero(win)
        if len(ys) == 0:
            return False
        ys = ys + y0; xs = xs + x0
        d = np.maximum(np.abs(ys - hy), np.abs(xs - hx))
        gain = np.array([YIELD[int(k)] for k in wild[ys, xs]], dtype=np.float32) / (2.0 * d + 1.0)
        i = int(np.argmax(gain))
        y, x = int(ys[i]), int(xs[i])
        k = int(wild[y, x])
        food = w.wild.pick(w, y, x, self.tick)
        self._walk(y, x, kind="forage")
        self.ledger["food"] += food
        self.foraged[NAMES[k]] = self.foraged.get(NAMES[k], 0.0) + food
        self.store.add(WILD_KIND.get(NAMES[k], "fruit"), food)
        return True

    def season_ledger(self) -> dict:
        """What was missed (D-051): fields lost to bush or flood, food that rotted, tending owed."""
        w, cfg = self.world, self.cfg
        live = sum(1 for f in self.fields if w.farm[f])
        due = cfg.tend_days * cfg.ticks_per_day
        overdue = sum(1 for f in self.fields if w.farm[f] and self.tick - self.tend_stamp.get(f, -10 ** 9) > 2 * due)
        return dict(fields_opened=len(self.fields), fields_live=live, fields_lost=len(self.fields) - live,
                    fields_overgrown=overdue, rotted=round(self.store.spoiled), strength=round(
                        cfg.hunger_floor + (1.0 - cfg.hunger_floor) * float(np.clip(self.store.ration_ema, 0.0, 1.0)), 2),
                    strength_min=round(getattr(self, "strength_min", 1.0), 2),          # snapshot vs the worst day: the run may end in harvest season
                    hungry_days=round(self.store.hungry_ticks / cfg.ticks_per_day))

    def food_all(self) -> float:
        """One stomach: what the fields yielded + what was picked + what was killed (x edible)."""
        return float(self.food_total) + float(sum(self.foraged.values())) + float(self.hunted_food)

    def walked_total(self) -> float:
        return float(sum(self.walked.values()))

    def _expand(self) -> None:
        if self.exhausted:
            return
        w, cfg = self.world, self.cfg
        if cfg.max_farms and sum(1 for f in self.fields if w.farm[f]) >= cfg.max_farms:   # live fields only
            return
        from .gen import ROCK, SAND
        from .crops import RICE as _RICE, DRY as _DRY
        # A mixed farm (D-036 round): paddies at the river up to `rice_cap`, then dry fields behind a
        # wider buffer -- the shape every river valley farmed for real.
        paddies_full = cfg.rice_cap > 0 and int((w.farm & (w.crop == _RICE)).sum()) >= cfg.rice_cap
        buffer = cfg.dry_buffer if (paddies_full and cfg.dry_buffer >= 0) else cfg.buffer
        ok = (~w.water) & (~w.farm) & (w.water_dist <= cfg.water_reach) & (w.water_dist > buffer) \
            & (w.substrate != ROCK)
        if cfg.water_fields:
            from .crops import CALTROP, fitness
            pond = w.water & (~w.farm) & (fitness(w, CALTROP) > 0.4)
            ok |= pond
        for (py, px) in self.protected:
            ok[py, px] = False
        cand = np.argwhere(ok)
        if len(cand) == 0:
            self.exhausted = True
            return
        force_crop = None
        if cfg.crop_pref == "rice" and not paddies_full:
            from .crops import RICE, choose
            rice = [c for c in cand if (not w.water[c[0], c[1]]) and choose(w, int(c[0]), int(c[1])) == RICE]
            if rice:
                cand = np.array(rice)
        elif paddies_full:
            land = [c for c in cand if not w.water[c[0], c[1]]]
            if land:
                cand = np.array(land)
                force_crop = _DRY
        home = np.array(self.home)
        d = np.linalg.norm(cand - home, axis=1)
        d = d + 12.0 * (w.substrate[cand[:, 0], cand[:, 1]] == SAND)   # sand only when nothing better is near
        y, x = cand[int(np.argmin(d))]
        w.till(int(y), int(x), crop=force_crop)
        self.spend(cfg.hours_till + self.walk_hours(y, x, 2), "till")
        self.fields.append((int(y), int(x)))
        self.tend_stamp[(int(y), int(x))] = self.tick
        if w.crop[int(y), int(x)] == _RICE:
            self.paddies.append((int(y), int(x)))
        self._walk(y, x, trips=2, kind="till")
        self.exhausted = False

    def _rotate(self) -> None:
        """稲麦二毛作 (D-037): inside `rice_season` a paddy holds rice; outside it the farmer pulls the
        bund, the field drains, and a dry crop is sown into the flush the anoxia deferred (soil.py).
        The debt becomes the winter crop's fertiliser -- or, if the field is poor, its grave."""
        w, cfg = self.world, self.cfg
        from .crops import RICE, DRY
        lo, hi = cfg.rice_season
        want = RICE if lo <= w.climate.season < hi else DRY
        for (y, x) in self.paddies:
            if not w.farm[y, x] or w.crop[y, x] == want:
                continue
            if not self.can(cfg.hours_rotate):
                break
            w.crop[y, x] = want
            if want == DRY:
                w.paddy[y, x] = 0.0                  # drained
            self.spend(cfg.hours_rotate, "tend")
            self._walk(y, x, kind="tend")

    def _harvest_wood(self) -> None:
        """Fell the nearest tree to home into the ledger."""
        w = self.world
        trees = np.argwhere(w.tree)
        if len(trees) == 0:
            return
        ty, tx = trees[int(np.argmin(np.linalg.norm(trees - np.array(self.home), axis=1)))]
        self.ledger["wood"] += w.cut_tree(int(ty), int(tx))
        self._walk(ty, tx, kind="wood")

    def _feed_wood(self) -> None:
        """Lay a log from the ledger where `feed_mode` says (fell one if the ledger is empty)."""
        w, cfg = self.world, self.cfg
        if not self.fields:
            return
        ref = np.array(self.fields[-1])
        if self.ledger["wood"] >= cfg.tree_wood:
            self.ledger["wood"] -= cfg.tree_wood
            mass = cfg.tree_wood
        else:
            trees = np.argwhere(w.tree)
            if len(trees) == 0:
                return
            ty, tx = trees[int(np.argmin(np.linalg.norm(trees - ref, axis=1)))]
            mass = w.cut_tree(int(ty), int(tx))
        if mass <= 0:
            return
        if cfg.feed_mode == "field":
            fy, fx = min(self.fields, key=lambda f: w.soil.h[f])
            w.lay_wood(fy, fx, mass)
            return
        # "shade": a land cell next to a remaining tree, as close to the fields as possible.
        near_tree = (~w.water) & (~w.farm) & (~w.tree) & (w.count3x3_tree() >= 1)
        cand = np.argwhere(near_tree)
        if len(cand) == 0:
            fy, fx = min(self.fields, key=lambda f: w.soil.h[f])
            w.lay_wood(fy, fx, mass)
            return
        cy, cx = cand[int(np.argmin(np.linalg.norm(cand - ref, axis=1)))]
        w.lay_wood(int(cy), int(cx), mass)
        self._walk(cy, cx, kind="wood")

    def _move_soil(self) -> None:
        """Carry rich soil from the shade to the sun: the transport loop of the MC experiment."""
        w, cfg = self.world, self.cfg
        if not self.fields:
            return
        src_ok = (~w.water) & (~w.farm) & (w.soil.h > cfg.h_rich)
        cand = np.argwhere(src_ok)
        if len(cand) == 0:
            return
        fy, fx = min(self.fields, key=lambda f: w.soil.h[f])
        # nearest rich cell to that field
        sy, sx = cand[int(np.argmin(np.linalg.norm(cand - np.array([fy, fx]), axis=1)))]
        amt = min(cfg.move_soil_amount, float(w.soil.h[sy, sx] - cfg.h_rich))
        w.soil.h[sy, sx] -= amt
        w.soil.h[fy, fx] = min(cfg.h_cap, w.soil.h[fy, fx] + amt)
        self.moved += amt
        self._walk(sy, sx, kind="tend")
        self._walk(fy, fx, kind="tend")

    def _replant_bank(self) -> None:
        """Put cover (a hedge or a tree) on the bare bank tile (water_dist == 1) closest to the newest field."""
        w, cfg = self.world, self.cfg
        ok = (~w.water) & (~w.tree) & (w.water_dist == 1)
        cand = np.argwhere(ok)
        if len(cand) == 0:
            return
        ref = np.array(self.fields[-1] if self.fields else self.home)
        d = np.linalg.norm(cand - ref, axis=1)
        y, x = cand[int(np.argmin(d))]
        if w.farm[y, x]:
            self.fields = [f for f in self.fields if f != (int(y), int(x))]
        if cfg.cover_plant == "vetiver":
            w.plant_cover(int(y), int(x))
        else:
            w.plant_tree(int(y), int(x))
        self.protected.add((int(y), int(x)))
        self.replanted += 1
        self.exhausted = False

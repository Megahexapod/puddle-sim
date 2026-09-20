"""The loop. World physics -> farmer -> creatures (per species) -> log."""
from __future__ import annotations

import csv
import json
import os
from typing import Dict, List

import numpy as np

from .config import Config
from .creatures import SPECIES, Creature, Species, spawn, species_score
from .world import count3x3
from .farmer import Farmer
from .industry import TABLE as IND_TABLE, Industries
from .sites import pick, score
from .world import World


class Simulation:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.world = World(cfg, self.rng)
        self.site_score = score(self.world)
        sites = pick(self.site_score, max(1, cfg.n_settlements), cfg.site_spacing)
        if not sites:
            sites = [None]
        self.farmers: List[Farmer] = [Farmer(cfg, self.world, self.rng, home=h, name=("player" if i == 0 else f"npc{i}"))
                                      for i, h in enumerate(sites)]
        self.farmer = self.farmers[0]
        self.industries = Industries(cfg, (cfg.height, cfg.width))
        for i, f in enumerate(self.farmers):
            f.index = i
            f.industries = self.industries
        self.world.industries = self.industries
        self.species: List[Species] = [SPECIES[k.strip()] for k in cfg.species.split(",") if k.strip()]
        self.pops: Dict[str, List[Creature]] = {
            sp.key: spawn(cfg, self.world, sp, self.rng, cfg.n_shrimp) for sp in self.species
        }
        self.scores: Dict[str, np.ndarray] = {}
        self.tick = 0
        self.yield_total: Dict[str, float] = {sp.key: 0.0 for sp in self.species}
        self.yield_tick: Dict[str, float] = {sp.key: 0.0 for sp in self.species}
        self.log: List[dict] = []
        self.log_fields = ((["tick", "n_farm", "n_trees", "n_trees_bank", "mean_quality", "mean_cover", "mean_food",
                            "n_eroding", "n_rich", "collapsed", "emerged", "wood", "food_tick", "food_total",
                            "n_mangrove", "mg_built", "mean_salt", "n_rock", "n_sand", "n_mud",
                            "n_saline", "n_grass", "n_swamp", "n_marsh", "n_dune", "n_seagrass", "n_algae", "sea_head"]
                           + [f"food_{i}" for i in range(max(1, cfg.n_settlements))]
                           + [f"{r}_{i}" for i in range(max(1, cfg.n_settlements)) for r in ("wood", "salt", "charcoal", "power")]
                           + ["n_industries", "temp_mean", "season", "snowpack", "spring_mult", "n_snow",
                              "mean_algae", "mean_o2", "min_o2", "n_dry", "n_rice", "n_caltrop", "n_vetiver", "n_crust"])
                           + [f"{k}_{sp.key}" for sp in self.species for k in ("n", "sat", "yield", "total")])
        self.predators3 = np.zeros((cfg.height, cfg.width), dtype=np.int16)
        # D-032: the pen (tier 2) and the ledger lines the farmer's harvest fills.
        self.pen_mask = np.zeros((cfg.height, cfg.width), dtype=bool)
        if cfg.pen == "pond" and self.world.pond_center is not None:      # D-043: a 5x5 on the pond
            py, px = int(self.world.pond_center[0]) - 2, int(self.world.pond_center[1]) - 2
            self.pen_mask[max(0, py):py + 5, max(0, px):px + 5] = self.world.water[max(0, py):py + 5, max(0, px):px + 5]
        elif cfg.pen:
            py, px, ph, pw = [int(v) for v in cfg.pen.split(",")]
            self.pen_mask[py:py + ph, px:px + pw] = True
        if self.pen_mask.any():
            for sp in self.species:
                for c in self.pops[sp.key]:
                    c.pen = bool(self.pen_mask[c.pos])
            if cfg.pen_stock > 0 and cfg.pen_species in self.pops:        # D-043: the keeper stocks the pen
                sp = SPECIES[cfg.pen_species]
                cells = np.argwhere(self.pen_mask)
                for i in range(cfg.pen_stock):
                    y, x = cells[int(self.rng.integers(len(cells)))]
                    c = Creature(cfg, sp, (int(y), int(x)), 0, self.rng)
                    c.pen = True
                    lt = sp.life(cfg); c.t_eff += 0.5 * lt.lifespan * 0.7; c.size = lt.size_at(c.t_eff)   # grown stock
                    self.pops[sp.key].append(c)
        self.world.pen_mask = self.pen_mask
        self.harvest: Dict[str, float] = {"meat": 0.0, "eggs": 0.0, "shell": 0.0, "cysts": 0.0, "kills": 0.0}
        self.pen_harvest: Dict[str, float] = {"meat": 0.0, "kills": 0.0}
        self._refresh_scores()

    @property
    def creatures(self) -> List[Creature]:
        return [c for pop in self.pops.values() for c in pop]

    # ------------------------------------------------------------------ one tick

    def _occupancy(self, key: str) -> np.ndarray:
        occ = np.zeros((self.cfg.height, self.cfg.width), dtype=np.int16)
        for c in self.pops[key]:
            occ[c.pos] += 1
        return occ

    def gene_stats(self, key: str = "carp", gene: str = "color") -> dict:
        """Mean / max of a gene, wild and in the pen (D-043 readback)."""
        pop = [c for c in self.pops.get(key, []) if c.alive]
        out = {}
        for tag, sel in (("wild", [c for c in pop if not c.pen]), ("pen", [c for c in pop if c.pen])):
            vals = [c.genes.get(gene, 0.0) for c in sel]
            out[tag] = dict(n=len(vals), mean=round(float(np.mean(vals)), 3) if vals else None,
                            max=round(float(np.max(vals)), 3) if vals else None,
                            red=sum(1 for v in vals if v >= 0.5))
        return out

    def _boar_effects(self, t: int) -> None:
        """D-042: a boar roots (ground cover goes), and eats the nuts and fruit under the trees it
        stands by -- the same cells a forager would have picked."""
        cfg, w = self.cfg, self.world
        from .flora import GRASS, FERN, BARE
        from .wild import NUT, FRUIT
        for c in self.pops["boar"]:
            y, x = c.pos
            if self.rng.random() < cfg.boar_root:
                if w.flora.kind[y, x] in (GRASS, FERN):
                    w.flora.kind[y, x] = BARE
            if w.wild.kind[y, x] in (NUT, FRUIT) and self.rng.random() < cfg.boar_mast:
                w.wild.pick(w, y, x, t)
                w.wild.eaten_by_boar += 1

    def _refresh_scores(self) -> None:
        for sp in self.species:
            self.scores[sp.key] = species_score(self.cfg, self.world, sp, self._occupancy(sp.key))

    def step(self) -> None:
        cfg, w = self.cfg, self.world
        t = self.tick
        w.tick_now = t

        grazers = np.zeros((cfg.height, cfg.width), dtype=np.int16)
        algae_eaters = np.zeros_like(grazers)
        animals = np.zeros_like(grazers)
        prey = np.zeros_like(grazers)
        predators = np.zeros(grazers.shape, dtype=np.float32)      # weighted by each predator's pressure (D-041)
        for sp in self.species:
            occ = self._occupancy(sp.key)
            animals += occ
            if sp.eats_detritus:
                grazers += occ
            if sp.eats_algae:
                algae_eaters += occ
            if sp.key in ("shrimp", "medaka"):
                prey += occ
            if sp.predator:
                predators += occ * sp.pressure
        w.crab_count = self._occupancy("crab") if "crab" in self.pops else np.zeros_like(grazers)
        w.snail_count = self._occupancy("snail") if "snail" in self.pops else np.zeros_like(grazers)
        w.boar_count = self._occupancy("boar") if "boar" in self.pops else np.zeros_like(grazers)
        w.grazer_count = algae_eaters
        w.animal_count = animals
        w.prey_count = prey
        self.predators3 = count3x3(predators)
        w.step(grazers, t)

        for f in self.farmers:
            f.observe(self.yield_tick.get(cfg.watch_species, 0.0))
            f.step(t)
        self.industries.step(w, self.farmers)

        if t % cfg.eval_every == 0:
            self._refresh_scores()

        for sp in self.species:
            pop = self.pops[sp.key]
            score = self.scores[sp.key]
            occ = self._occupancy(sp.key)
            produced = 0.0
            births: List[Creature] = []
            for c in pop:
                produced += c.step(t, w, score, self.rng)
                if c.alive and c.can_breed(t):
                    y, x = c.pos
                    if sp.key in ("shrimp", "medaka") and self.rng.random() < self.predators3[y, x]:
                        c.last_breed = t          # the beetle got the brood
                        continue
                    y0, y1 = max(0, y - 2), min(cfg.height, y + 3)
                    x0, x1 = max(0, x - 2), min(cfg.width, x + 3)
                    cap = sp.life(cfg).density_cap * (cfg.pen_density_mult if c.pen else 1.0)
                    if occ[y0:y1, x0:x1].sum() - 1 < cap:          # others in the 5x5, not counting the parent
                        walk = w.walkable(sp.medium) & (self.pen_mask if c.pen else ~self.pen_mask)
                        nb = w.water_neighbours(y, x, mask=walk)
                        if nb:
                            c.last_breed = t
                            mate = c
                            if sp.genes:                                   # D-043: a mate within the 5x5, else selfing
                                cands = [o for o in pop if o is not c and o.alive and o.pen == c.pen
                                         and abs(o.pos[0] - y) <= 2 and abs(o.pos[1] - x) <= 2
                                         and sp.life(cfg).is_mature(o.size)]
                                if cands:
                                    mate = cands[int(self.rng.integers(len(cands)))]
                            for _ in range(c.litter(self.rng)):
                                child_pos = nb[int(self.rng.integers(len(nb)))]
                                occ[child_pos] += 1
                                child = Creature(cfg, sp, child_pos, t, self.rng)
                                child.pen = c.pen
                                if sp.genes:
                                    child._parent_genes = c.genes
                                    child.inherit(mate, self.rng)
                                births.append(child)
            self.pops[sp.key] = [c for c in pop if c.alive] + births
            self.yield_tick[sp.key] = produced
            self.yield_total[sp.key] += produced
            if sp.key == "boar":
                self._boar_effects(t)

        # Immigration: a species down to a handful gets a few newcomers where it could live (birds carry eggs).
        if cfg.immigrate_every > 0 and t % cfg.immigrate_every == 0:
            for sp in self.species:
                if sp.immigrates and len(self.pops[sp.key]) < cfg.immigrate_min:
                    sc = self.scores.get(sp.key)
                    if sc is not None and float(sc.max()) >= cfg.immigrate_score:
                        self.pops[sp.key] += spawn(cfg, w, sp, self.rng, cfg.immigrate_n, tick=t)

        # D-032 tier 0: the forager. Every hunt_every ticks the player's farmer kills the n biggest of
        # hunt_species within hunt_radius of home (above the mesh size) and collects their stock.
        if cfg.hunt_every > 0 and t % cfg.hunt_every == 0 and cfg.hunt_species in self.pops and self.farmer.can(cfg.hours_hunt):
            sp = SPECIES[cfg.hunt_species]
            lt = sp.life(cfg)
            hy_, hx_ = self.farmer.home
            near = [c for c in self.pops[sp.key] if c.alive and not c.pen and c.size >= cfg.hunt_min_size * lt.mass
                    and max(abs(c.pos[0] - hy_), abs(c.pos[1] - hx_)) <= cfg.hunt_radius]
            near.sort(key=lambda c: -c.size)
            if near:
                self.farmer.spend(cfg.hours_hunt, "hunt")          # the trip is paid whether or not... no: only when there is something to go for
            for c in near[: cfg.hunt_n]:
                c.alive = False
                self.harvest["meat"] += c.size
                self.harvest["kills"] += 1
                self.farmer.hunted_food += sp.food * (c.size / lt.mass)
                self.farmer.store.add("meat", sp.food * (c.size / lt.mass))
                if sp.passive:
                    self.harvest[sp.passive] = self.harvest.get(sp.passive, 0.0) + c.stock
                self.farmer._walk(c.pos[0], c.pos[1], kind="hunt")
            self.pops[sp.key] = [c for c in self.pops[sp.key] if c.alive]
        # D-032 tier 2: the pen. Fed from the ledger (wood -> litter), dung fouls the water, the
        # stock inside is collected every round; the animals are harvested when they are grown.
        if self.pen_mask.any():
            fed = 0.0
            if self.farmer.ledger.get("wood", 0.0) > 0.0:
                cells = int(self.pen_mask.sum())
                cost = cfg.pen_feed * cells * 0.05
                if self.farmer.ledger["wood"] >= cost:
                    self.farmer.ledger["wood"] -= cost
                    w.detritus[self.pen_mask] = np.minimum(3.0, w.detritus[self.pen_mask] + cfg.pen_feed)
                    fed = cfg.pen_feed
            inside = [c for sp in self.species for c in self.pops[sp.key] if c.pen and c.alive]
            for c in inside:
                w.quality[c.pos] = max(0.0, w.quality[c.pos] - cfg.pen_waste)
            if cfg.hunt_every > 0 and t % cfg.hunt_every == 0:
                grown = [c for c in inside if c.size >= 0.9 * c.species.life(cfg).mass]
                keep = cfg.select_keep                             # breeding stock stays in the pen
                if cfg.select_trait:
                    # D-043: the breeder's sieve. Every harvest, the `keep` most colourful mature fish
                    # stay as breeding stock; every other grown fish is eaten. That is what a breeder
                    # does -- not one dull fish per month, all of them.
                    mature = [c for c in inside if c.species.life(cfg).is_mature(c.size)]
                    mature.sort(key=lambda c: -c.genes.get(cfg.select_trait, 0.0))
                    breeders = set(id(c) for c in mature[:keep])
                    cull = [c for c in grown if id(c) not in breeders]
                else:
                    grown.sort(key=lambda c: -c.size)
                    cull = grown[: max(0, min(cfg.hunt_n, len(inside) - keep))]
                for c in cull:
                    c.alive = False
                    self.pen_harvest["meat"] += c.size
                    self.pen_harvest["kills"] += 1
                    self.farmer.hunted_food += c.species.food * (c.size / c.species.life(cfg).mass)
                    self.farmer.store.add("meat", c.species.food * (c.size / c.species.life(cfg).mass))
                for c in inside:
                    if c.alive and c.stock > 0:
                        self.harvest[c.species.passive or "eggs"] = self.harvest.get(c.species.passive or "eggs", 0.0) + c.stock
                        c.stock = 0.0
                for sp in self.species:
                    self.pops[sp.key] = [c for c in self.pops[sp.key] if c.alive]

        for f in self.farmers:
            if f.fields:
                ys = np.array([p[0] for p in f.fields]); xs = np.array([p[1] for p in f.fields])
                got = float(w.food_cell[ys, xs].sum())
                f.food_total += got
                f.store.add("grain", got)          # (a continuous trickle for now; the harvest burst is step 2)
        if t % cfg.log_every == 0:
            self._log()
        self.tick += 1

    def run(self, on_snapshot=None) -> None:
        for _ in range(self.cfg.ticks):
            if on_snapshot is not None and self.tick % self.cfg.snap_every == 0:
                on_snapshot(self)
            self.step()
        if on_snapshot is not None:
            on_snapshot(self)

    # ------------------------------------------------------------------ logging

    def _log(self) -> None:
        w = self.world
        wm = w.water
        land = ~wm
        row = {
            "tick": self.tick,
            "n_farm": int(w.farm.sum()),
            "n_trees": int(w.tree.sum()),
            "n_trees_bank": int((w.tree & (w.water_dist == 1)).sum()),
            "mean_quality": float(w.quality[wm].mean()) if wm.any() else 1.0,
            "mean_cover": float(w.cover()[wm].mean()) if wm.any() else 0.0,
            "mean_food": float(w.food()[wm].mean()) if wm.any() else 0.0,
            "n_eroding": int((w.soil.eroding & land).sum()),
            "n_rich": int((w.soil.rich & land).sum()),
            "collapsed": w.soil.collapsed_total,
            "emerged": w.soil.emerged_total,
            "wood": float(w.soil.wood.sum()),
            "food_tick": w.food_tick,
            "food_total": w.food_total,
            "n_mangrove": int(w.mangrove.sum()),
            "mg_built": w.mg_built,
            "mean_salt": float(w.salinity[wm].mean()) if wm.any() else 0.0,
            "n_rock": int((w.substrate == 1).sum()),
            "n_sand": int((w.substrate == 2).sum()),
            "n_mud": int((w.substrate == 3).sum()),
            "n_saline": int(w.saline().sum()),
            "sea_head": round(float(w.hydro.sea_head), 3),
        }
        fc = w.flora.counts()
        for k in ("grass", "swamp", "marsh", "dune", "seagrass", "algae"):
            row[f"n_{k}"] = fc.get(k, 0)
        for i in range(max(1, self.cfg.n_settlements)):
            fm = self.farmers[i] if i < len(self.farmers) else None
            row[f"food_{i}"] = fm.food_total if fm else 0.0
            for r in ("wood", "salt", "charcoal", "power"):
                row[f"{r}_{i}"] = round(fm.ledger.get(r, 0.0), 2) if fm else 0.0
        row["n_industries"] = int((self.industries.kind > 0).sum())
        cl = w.climate
        row["temp_mean"] = round(float(cl.temp.mean()), 2)
        row["season"] = round(cl.season, 3)
        row["snowpack"] = round(cl.snowpack, 2)
        row["spring_mult"] = round(cl.spring_mult, 3)
        row["n_snow"] = int(cl.snow.sum())
        row["mean_algae"] = round(float(w.algae[wm].mean()) if wm.any() else 0.0, 3)
        row["mean_o2"] = round(float(w.oxygen[wm].mean()) if wm.any() else 1.0, 3)
        row["min_o2"] = round(float(w.oxygen[wm].min()) if wm.any() else 1.0, 3)
        row["n_dry"] = int((w.farm & (w.crop == 1)).sum())
        row["n_rice"] = int((w.farm & (w.crop == 2)).sum())
        row["n_caltrop"] = int((w.farm & (w.crop == 3)).sum())
        row["n_vetiver"] = int((w.flora.kind == 9).sum())
        row["n_crust"] = int((w.substrate == 4).sum())
        for sp in self.species:
            pop = self.pops[sp.key]
            row[f"n_{sp.key}"] = len(pop)
            row[f"sat_{sp.key}"] = float(np.mean([c.satisfaction for c in pop])) if pop else 0.0
            row[f"yield_{sp.key}"] = self.yield_tick[sp.key]
            row[f"total_{sp.key}"] = self.yield_total[sp.key]
        self.log.append(row)

    def save(self, out_dir: str) -> None:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "log.csv"), "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=self.log_fields)
            wr.writeheader()
            wr.writerows(self.log)
        with open(os.path.join(out_dir, "config.json"), "w") as f:
            json.dump(self.cfg.to_dict(), f, indent=2)

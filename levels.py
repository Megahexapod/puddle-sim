"""Levels (2026-09-16): a balanced starting world, a set of goals, a set of players (farmer policies).
Does any player reach the goals? Does the naive one fail? That is what makes it a level.

    python3 levels.py paddy            # run level "paddy" with every policy, print the pass/fail matrix
"""
from __future__ import annotations

import json
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from puddle_sim.config import Config  # noqa: E402
from puddle_sim.crops import RICE  # noqa: E402
from puddle_sim.sim import Simulation  # noqa: E402
from puddle_sim.world import count3x3  # noqa: E402

# D-043 step 7: the same handful of economies on three terrains. The level "passes" when the
# ranking flips -- the farmer wins the forest river, the forager / fisher / hunter win where the
# terrain keeps the plough away from the water (canyon) or the water on the fields (swamp forest).
ECONOMIES = {
    "farmer":         dict(crop_pref="rice", rice_cap=12, dry_buffer=2),
    "farmer_forage":  dict(crop_pref="rice", rice_cap=12, dry_buffer=2, forage_every=5, forage_radius=30),
    "farmer_hunter":  dict(crop_pref="rice", rice_cap=12, dry_buffer=2, hunt_every=15, hunt_n=1, hunt_radius=20, hunt_species="boar", hunt_min_size=0.5),
    "forager":        dict(expand_every=10 ** 9, industries=0, harvest_every=0, forage_every=3, forage_radius=30),
    "fisher_forager": dict(expand_every=10 ** 9, industries=0, harvest_every=0, forage_every=3, forage_radius=30, hunt_every=10, hunt_n=1, hunt_radius=24, hunt_species="carp", hunt_min_size=0.8),
    "hunter_forager": dict(expand_every=10 ** 9, industries=0, harvest_every=0, forage_every=3, forage_radius=30, hunt_every=15, hunt_n=1, hunt_radius=30, hunt_species="boar", hunt_min_size=0.5),
}
_BASE = dict(sea=0, tributary=0, n_settlements=1, t_north=17, t_south=23, season_amp=4.0, seed=2)

LEVELS = {
    "forest": dict(name="地形 1 森林河", ticks=8000, settle=375, policies=ECONOMIES,
                   map=dict(_BASE, seg_highland=0.1, seg_canyon=0.0, seg_hills=0.2, seg_floodplain=0.7)),
    "canyon": dict(name="地形 2 峽谷", ticks=8000, settle=375, policies=ECONOMIES,
                   map=dict(_BASE, seg_highland=0.15, seg_canyon=0.7, seg_hills=0.15, seg_floodplain=0.0, rock_slope_canyon=0.08)),
    "swamp":  dict(name="地形 3 湖沼林", ticks=8000, settle=375, policies=ECONOMIES,
                   map=dict(_BASE, seg_highland=0.05, seg_canyon=0.0, seg_hills=0.1, seg_floodplain=0.85, swamp=1, pond=1, rain=0.0012)),
    "paddy": dict(
        name="關卡 1 水田",
        map=dict(sea=0, tributary=0, seg_highland=0.1, seg_canyon=0.0, seg_hills=0.2, seg_floodplain=0.7,
                 n_settlements=1, t_north=17, t_south=23, season_amp=4.0, seed=2),   # warm enough for rice in summer, never above the shrimp's 32°
        ticks=8000,
        settle=375,           # ticks of nobody doing anything (~45 days): the balanced start (baseline measured here)
        policies={
            "naive":          dict(),
            "paddy":          dict(crop_pref="rice"),
            "paddy_buffer2":  dict(crop_pref="rice", buffer=2),
            "paddy_slow":     dict(crop_pref="rice", expand_every=15),
            "adaptive":       dict(crop_pref="rice", farmer_mode="adaptive", buffer=1, cover_plant="vetiver"),
            "paddy_wood":     dict(crop_pref="rice", feed_wood_every=15),
            "paddy_hunt":     dict(crop_pref="rice", hunt_every=12, hunt_n=1, hunt_radius=20),
            "no_industry":    dict(crop_pref="rice", industries=0),
            "paddy_cap25":    dict(crop_pref="rice", max_farms=25),       # round 4: stop before the river runs dry
            "paddy_clear2":   dict(crop_pref="rice", clear_radius=2),     # round 4: cut wider so rice gets light 15
            "mixed25_b2":     dict(crop_pref="rice", rice_cap=25, dry_buffer=2),   # D-036: 25 paddies at the river, dry fields behind a 2-cell bank
            "mixed12_b2":     dict(crop_pref="rice", rice_cap=12, dry_buffer=2),
            "paddy_buffer1":  dict(crop_pref="rice", buffer=1),                 # D-037 step 1: one row of bank trees kept, paddies still reach water
            "mixed12_rotate": dict(crop_pref="rice", rice_cap=12, dry_buffer=2, rotate=1),            # 稲麦二毛作 on the 12 paddies
            "mixed12_winter": dict(crop_pref="rice", rice_cap=12, dry_buffer=2, irrigate_min_depth=0.05),  # 冬期湛水: keep irrigating from a shallower river
            "mixed12_fish":   dict(crop_pref="rice", rice_cap=12, dry_buffer=2, hunt_every=10, hunt_n=1, hunt_radius=24, hunt_species="carp", hunt_min_size=0.5),   # D-041: a farmer who also fishes the pools
            "mixed12_fish8":  dict(crop_pref="rice", rice_cap=12, dry_buffer=2, hunt_every=10, hunt_n=1, hunt_radius=24, hunt_species="carp", hunt_min_size=0.8),   # ... with a wider mesh
            "mixed12_noboar": dict(crop_pref="rice", rice_cap=12, dry_buffer=2, species="shrimp,medaka,crab,snail,beetle,carp"),   # D-042 control: the same farm in a world without boar
            "mixed12_hunter": dict(crop_pref="rice", rice_cap=12, dry_buffer=2, hunt_every=15, hunt_n=1, hunt_radius=20, hunt_species="boar", hunt_min_size=0.5),   # D-042: shoots the raiders
            # D-043 step 6: the other economies, on the same ruler
            "forager":        dict(expand_every=10 ** 9, industries=0, harvest_every=0, forage_every=3, forage_radius=30),
            "fisher_forager": dict(expand_every=10 ** 9, industries=0, harvest_every=0, forage_every=3, forage_radius=30, hunt_every=10, hunt_n=1, hunt_radius=24, hunt_species="carp", hunt_min_size=0.8),
            "hunter_forager": dict(expand_every=10 ** 9, industries=0, harvest_every=0, forage_every=3, forage_radius=30, hunt_every=15, hunt_n=1, hunt_radius=30, hunt_species="boar", hunt_min_size=0.5),
            "mixed12_forage": dict(crop_pref="rice", rice_cap=12, dry_buffer=2, forage_every=5, forage_radius=30),
        },
    ),
}


def measure(sim) -> dict:
    w = sim.world
    rice = w.farm & (w.crop == RICE)
    snails3 = count3x3(w.snail_count)
    dmg = float(np.clip(1.0 - 0.25 * snails3[rice], 0.0, 1.0).mean()) if rice.any() else 1.0
    return dict(
        food=round(w.food_total), farms=int(w.farm.sum()), rice=int(rice.sum()),
        shrimp=len(sim.pops.get("shrimp", [])), medaka=len(sim.pops.get("medaka", [])), snail=len(sim.pops.get("snail", [])),
        quality=round(float(w.quality[w.water].mean()), 3), eroding_farms=int((w.farm & w.soil.eroding).sum()),
        rice_intact=round(dmg, 3), meat=round(sim.harvest["meat"], 1), trees=int(w.tree.sum()),
        h_farm=round(float(w.soil.h[w.farm].mean()), 2) if w.farm.any() else 0.0,
        flooded=int(w.soil.flooded.sum()), trapped=round(float(w.trapped), 2),
        carp=len(sim.pops.get("carp", [])), beetle=len(sim.pops.get("beetle", [])), boar=len(sim.pops.get("boar", [])),
        # the ruler (D-043 step 6): one stomach, one pair of legs
        food_all=round(sim.farmer.food_all()), foraged=round(sum(sim.farmer.foraged.values()), 1),
        hunted=round(sim.farmer.hunted_food, 1), walked=round(sim.farmer.walked_total()),
        fpw=round(sim.farmer.food_all() / max(1.0, sim.farmer.walked_total()), 4),
        hours=round(sim.farmer.hours_total()), fph=round(sim.farmer.food_all() / max(1.0, sim.farmer.hours_total()), 3),
        hours_by={k: round(v) for k, v in sim.farmer.hours_by.items()},
        walked_by={k: round(v) for k, v in sim.farmer.walked.items() if v},
        foraged_by={k: round(v, 1) for k, v in sim.farmer.foraged.items()},
        **{"house_" + k: v for k, v in sim.farmer.store.summary().items()},   # D-044 step 1: eaten / spoiled / hungry / fed
        season=sim.farmer.season_ledger(),                                      # D-051: what was missed
        extinct=[k for k, v in sim.pops.items() if len(v) == 0 and k in ("boar", "carp", "shrimp")],
    )


def run(args):
    level, policy = args
    L = LEVELS[level]
    kw = dict(L["map"])
    kw.update(L["policies"][policy] if policy else {})
    if policy is None:                       # the baseline: nobody farms
        kw.update(expand_every=10 ** 9, industries=0, harvest_every=0)
    cfg = Config(**kw)
    sim = Simulation(cfg)
    # A balanced start for everyone: the settle period runs with the farmer asleep.
    exp, ind, harv, fo, hu = cfg.expand_every, cfg.industries, cfg.harvest_every, cfg.forage_every, cfg.hunt_every
    cfg.expand_every, cfg.industries, cfg.harvest_every, cfg.forage_every, cfg.hunt_every = 10 ** 9, 0, 0, 0, 0
    for _ in range(L["settle"]):
        sim.step()
    start = measure(sim)
    cfg.expand_every, cfg.industries, cfg.harvest_every, cfg.forage_every, cfg.hunt_every = exp, ind, harv, fo, hu
    sim.world.food_total = 0.0
    sim.farmer.store.__init__(cfg.ticks_per_day)     # the household's year starts now (D-044 step 1)
    series = []
    for i in range(L["ticks"]):
        sim.step()
        if (i + 1) % 500 == 0:
            m = measure(sim); m["tick"] = i + 1; series.append(m)
    return policy or "baseline", dict(start=start, end=measure(sim), series=series)


def goals(level: str, results: dict) -> dict:
    """Pass/fail per policy. Round 4 (D-034): the bar for food and medaka is 80% of the *best* player,
    not the naive one — naive must be able to fail. Shrimp stays against the baseline (nature is the
    canary's reference); the rest are absolute."""
    B = results["baseline"]["end"]
    players = {p: r["end"] for p, r in results.items() if p != "baseline"}
    best_food = max(e["food"] for e in players.values())
    best_medaka = max(e["medaka"] for e in players.values())
    out = {}
    for pol, e in players.items():
        g = {
            "食物≥80%最佳": e["food"] >= 0.8 * best_food,
            "蝦子≥80%基線": e["shrimp"] >= 0.8 * B["shrimp"],
            "水質≥0.7": e["quality"] >= 0.7,
            "鱂魚≥80%最佳": e["medaka"] >= 0.8 * best_medaka,
            "稻沒被螺吃光(≥0.6)": e["rice_intact"] >= 0.6,
            "田流失中≤30%": e["eroding_farms"] <= 0.3 * max(1, e["farms"]),
        }
        out[pol] = g
    out["_thresholds"] = dict(food=round(0.8 * best_food), shrimp=round(0.8 * B["shrimp"]), medaka=round(0.8 * best_medaka))
    return out


def terrain_report(levels_: list, out_path: str = "out/terrain.json") -> dict:
    """Run ECONOMIES on each terrain (no baseline, no six goals); print food_all and food per cell
    walked, ranked. The question is only: does the ranking flip between maps?"""
    jobs = [(lv, p) for lv in levels_ for p in LEVELS[lv]["policies"]]
    with Pool(4) as pool:
        out = pool.map(run, jobs)
    res = {}
    for (lv, p), (_, r) in zip(jobs, out):
        res.setdefault(lv, {})[p] = r
    json.dump(res, open(out_path, "w"), ensure_ascii=False)
    for lv in levels_:
        print(f"== {LEVELS[lv]['name']}")
        rows = sorted(res[lv].items(), key=lambda kv: -kv[1]["end"]["food_all"])
        for p, r in rows:
            e = r["end"]
            print(f"  {p:15s} food_all {e['food_all']:6d}  (田 {e['food']:5d} 採 {e['foraged']:6.1f} 獵 {e['hunted']:6.1f})  hours {e['hours']:5d}  food/h {e['fph']:.2f}  farms {e['farms']:3d} shrimp {e['shrimp']:3d} carp {e['carp']:3d} boar {e['boar']:3d} 工時={e['hours_by']}"
                  f"  | 吃 {e['house_eaten']:5d} 爛 {e['house_spoiled']:5d} 飽 {e['house_fed']:.2f} 剩 {e['house_left']:4d}  爛掉的={e['house_spoiled_by']}  季節帳={e['season']} 絕={e['extinct']}")
    return res


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "terrain":
        terrain_report(sys.argv[2:] or ["forest", "canyon", "swamp"])
        sys.exit(0)
    level = sys.argv[1] if len(sys.argv) > 1 else "paddy"
    L = LEVELS[level]
    jobs = [(level, None)] + [(level, p) for p in L["policies"]]
    t = time.time()
    with Pool(4) as pool:
        results = dict(pool.map(run, jobs))
    G = goals(level, results)
    json.dump(dict(results=results, goals=G), open(f"out/level_{level}.json", "w"), ensure_ascii=False)
    thr = G.pop("_thresholds")
    names = list(next(iter(G.values())).keys())
    print(f"{L['name']}  baseline: {results['baseline']['end']}")
    print(f"thresholds: {thr}")
    print(f"{'policy':14s} " + " ".join(f"{n[:10]:>10s}" for n in names) + "   food  farms  rice shrimp medaka quality erod")
    for pol, g in G.items():
        e = results[pol]["end"]
        print(f"{pol:14s} " + " ".join(f"{'✓' if v else '✗':>10s}" for v in g.values())
              + f"  {e['food']:5d} {e['farms']:5d} {e['rice']:5d} {e['shrimp']:6d} {e['medaka']:6d} {e['quality']:7.2f} {e['eroding_farms']:4d}   ({sum(g.values())}/{len(g)})")
    print("done", round(time.time() - t), "s")

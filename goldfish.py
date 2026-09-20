"""D-043: 鯽 -> 金魚. Three arms on the forest river, one year (12000 ticks, ~12 carp generations):

    wild      no pen. Birds eat the bright ones; the population should stay olive.
    pen       a roofed pen on the pond, harvested for food (biggest first), no sieve. Drift.
    select    the same pen, but the breeding stock kept at every harvest is the most colourful.

    python3 goldfish.py            # prints mean / max colour, wild vs pen, every 1500 ticks
"""
from __future__ import annotations

import json
import sys
from multiprocessing import Pool

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from puddle_sim.config import Config  # noqa: E402
from puddle_sim.sim import Simulation  # noqa: E402

MAP = dict(sea=0, tributary=0, seg_highland=0.1, seg_canyon=0.0, seg_hills=0.2, seg_floodplain=0.7,
           n_settlements=1, t_north=17, t_south=23, season_amp=4.0, seed=2,
           expand_every=10 ** 9, industries=0, harvest_every=0)
ARMS = {
    "wild":   dict(),
    "pen":    dict(pen="pond", pen_stock=12, harvest_every=8, hunt_every=15, hunt_n=1, hunt_species="carp", hunt_min_size=0.8),
    "select": dict(pen="pond", pen_stock=12, harvest_every=8, hunt_every=15, hunt_n=1, hunt_species="carp", hunt_min_size=0.8, select_trait="color", select_keep=6),
}
TICKS = 12000


def run(arm: str):
    cfg = Config(**{**MAP, **ARMS[arm]})
    sim = Simulation(cfg)
    series = []
    for i in range(TICKS):
        sim.step()
        if (i + 1) % 1500 == 0:
            g = sim.gene_stats()
            g["tick"] = i + 1
            g["birds"] = sim.world.eaten_by_birds
            g["pen_meat"] = round(sim.pen_harvest["meat"], 1)
            series.append(g)
    return arm, series


if __name__ == "__main__":
    with Pool(3) as pool:
        out = dict(pool.map(run, list(ARMS)))
    json.dump(out, open("out/goldfish.json", "w"), ensure_ascii=False)
    for arm, series in out.items():
        print(f"== {arm}")
        for g in series:
            w, p = g["wild"], g["pen"]
            print(f"  t={g['tick']:5d}  wild n={w['n']:3d} mean={w['mean']} max={w['max']} red={w['red']}   "
                  f"pen n={p['n']:3d} mean={p['mean']} max={p['max']} red={p['red']}   birds={g['birds']} pen_meat={g['pen_meat']}")

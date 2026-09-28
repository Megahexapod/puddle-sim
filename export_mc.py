"""Export one settled map as blocks for the MC mod (D-052: the player face goes into MC).

The sim keeps the parameters; MC gets the conclusions (1852k). Per cell we hand over what a
block can carry: bed height, water depth, substrate, plant kind + age, farm, soil phase, tread,
and where every creature stands. The mod (`/puddle world <name>`) turns each cell into a 4x4
column (D-023: one cell = 4 blocks, one height unit = 4 blocks) and puts the shrimp in the water.

    python3 export_mc.py forest            # -> ../mc_survival_mod/run/puddle/forest.csv (+ _creatures.csv, .json)
    python3 export_mc.py forest --ticks 2000 --policy farmer   # a lived-in map instead of the balanced start
"""
from __future__ import annotations

import argparse
import json
import os

from levels import LEVELS
from puddle_sim.config import Config
from puddle_sim.flora import (ALGAE, BARE, CHESTNUT, DUNE, FERN, GRASS, MANGROVE, MARSH, MUSSEL,
                              PERSIMMON, SEAGRASS, SWAMP, TREE, VETIVER, XERO)
from puddle_sim.gen import MUD, ROCK, SALT, SAND, SOIL
from puddle_sim.sim import Simulation

KIND_NAME = {BARE: "bare", TREE: "tree", SWAMP: "swamp", MANGROVE: "mangrove", MARSH: "marsh", DUNE: "dune",
             GRASS: "grass", SEAGRASS: "seagrass", ALGAE: "algae", VETIVER: "vetiver", XERO: "xero",
             FERN: "fern", CHESTNUT: "chestnut", PERSIMMON: "persimmon", MUSSEL: "mussel"}
SUB_NAME = {SOIL: "soil", ROCK: "rock", SAND: "sand", MUD: "mud", SALT: "salt"}

DEFAULT_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "mc_survival_mod", "run", "puddle")


def build(level: str, policy: str | None, ticks: int) -> Simulation:
    L = LEVELS[level]
    kw = dict(L["map"])
    if policy:
        kw.update(L["policies"][policy])
    else:
        kw.update(expand_every=10 ** 9, industries=0, harvest_every=0)   # nobody farms: the balanced start
    cfg = Config(**kw)
    sim = Simulation(cfg)
    for _ in range(L["settle"] + ticks):
        sim.step()
    return sim


def export(sim: Simulation, name: str, out_dir: str) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    w = sim.world
    cfg = sim.cfg
    H, W = cfg.height, cfg.width
    b = w.hydro.b
    d = w.hydro.d
    water = d >= cfg.d_min
    rows = []
    for y in range(H):
        for x in range(W):
            rows.append("%d,%d,%.3f,%.3f,%s,%s,%d,%d,%.2f,%.2f,%d,%.2f" % (
                y, x, float(b[y, x]), float(d[y, x]) if water[y, x] else 0.0,
                SUB_NAME.get(int(w.substrate[y, x]), "soil"), KIND_NAME.get(int(w.flora.kind[y, x]), "bare"),
                int(w.flora.age[y, x]), int(w.farm[y, x]), float(w.soil.h[y, x]), float(w.ground.pack[y, x]),
                int(w.light[y, x]), float(w.quality[y, x])))
    with open(os.path.join(out_dir, name + ".csv"), "w") as f:
        f.write("y,x,b,d,sub,kind,age,farm,h,pack,light,quality\n")
        f.write("\n".join(rows) + "\n")
    with open(os.path.join(out_dir, name + "_creatures.csv"), "w") as f:
        f.write("species,y,x\n")
        for c in sim.creatures:
            f.write("%s,%d,%d\n" % (c.species.key, c.pos[0], c.pos[1]))
    home = sim.farmer.home
    meta = dict(
        name=name, width=W, height=H, cell_blocks=4, height_blocks=4, d_min=cfg.d_min,
        home=[int(home[0]), int(home[1])], spring=[int(w.hydro.spring[0]), int(w.hydro.spring[1])],
        tick=sim.tick, b_min=float(b.min()), b_max=float(b.max()),
        counts={k: len(v) for k, v in sim.pops.items()}, flora=w.flora.counts(),
        n_water=int(water.sum()), n_farm=int(w.farm.sum()),
    )
    with open(os.path.join(out_dir, name + ".json"), "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    return meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("level", nargs="?", default="forest", choices=list(LEVELS))
    ap.add_argument("--name", default=None)
    ap.add_argument("--policy", default=None)
    ap.add_argument("--ticks", type=int, default=0, help="extra ticks after settle")
    ap.add_argument("--out", default=DEFAULT_OUT)
    a = ap.parse_args()
    sim = build(a.level, a.policy, a.ticks)
    meta = export(sim, a.name or a.level, a.out)
    print(json.dumps(meta, ensure_ascii=False))

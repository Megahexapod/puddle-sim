#!/usr/bin/env python3
"""puddle_sim CLI.

  python3 run.py --ticks 4000 --buffer 0 --out out/b0
  python3 run.py sweep --buffers 0 1 2 3 --seeds 1 2 3 --out out/sweep

Any Config field can be overridden with --set name=value (repeatable).
"""
from __future__ import annotations

import argparse
import dataclasses
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from puddle_sim.config import Config  # noqa: E402
from puddle_sim.render import flow_map, snapshot, summary, sweep_plot  # noqa: E402
from puddle_sim.sim import Simulation  # noqa: E402


def apply_overrides(cfg: Config, sets):
    fields = {f.name: f.type for f in dataclasses.fields(cfg)}
    for kv in sets or []:
        k, v = kv.split("=", 1)
        if k not in fields:
            sys.exit(f"unknown config field: {k}")
        cur = getattr(cfg, k)
        setattr(cfg, k, type(cur)(v) if not isinstance(cur, bool) else v.lower() in ("1", "true"))
    return cfg


def run_one(cfg: Config, out: str, snaps: bool = True) -> Simulation:
    sim = Simulation(cfg)
    os.makedirs(out, exist_ok=True)
    sim.run(on_snapshot=(lambda s: (snapshot(s, out), flow_map(s, out))) if snaps else None)
    sim.save(out)
    summary(sim, out)
    return sim


def main(argv=None):
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd")
    p.add_argument("--ticks", type=int, default=None)
    p.add_argument("--buffer", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--out", default="out/run")
    p.add_argument("--set", action="append", help="name=value config override")

    sp = sub.add_parser("sweep")
    sp.add_argument("--buffers", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    sp.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    sp.add_argument("--ticks", type=int, default=None)
    sp.add_argument("--out", default="out/sweep")
    sp.add_argument("--set", action="append")

    a = p.parse_args(argv)

    if a.cmd == "sweep":
        results = []
        for b in a.buffers:
            for s in a.seeds:
                cfg = apply_overrides(Config(buffer=b, seed=s), a.set)
                if a.ticks:
                    cfg.ticks = a.ticks
                sim = run_one(cfg, os.path.join(a.out, f"b{b}_s{s}"), snaps=False)
                row = {"buffer": b, "seed": s, "n_farm": int(sim.world.farm.sum())}
                for sp in sim.species:
                    row[f"yield_{sp.key}"] = sim.yield_total[sp.key]
                    row[f"n_{sp.key}"] = len(sim.pops[sp.key])
                results.append(row)
                print(f"buffer={b} seed={s}  farm={row['n_farm']:3d}  " +
                      "  ".join(f"{sp.key}: yield={sim.yield_total[sp.key]:7.1f} n={len(sim.pops[sp.key]):3d}"
                                for sp in sim.species))
        print(sweep_plot(results, a.out, [sp.key for sp in sim.species]))
        return

    cfg = apply_overrides(Config(), a.set)
    if a.ticks is not None:
        cfg.ticks = a.ticks
    if a.buffer is not None:
        cfg.buffer = a.buffer
    if a.seed is not None:
        cfg.seed = a.seed
    sim = run_one(cfg, a.out)
    last = sim.log[-1]
    pops = "  ".join(f"{sp.key}={last[f'n_{sp.key}']} y={sim.yield_total[sp.key]:.0f}" for sp in sim.species)
    print(f"done. tick={sim.tick} farm={last['n_farm']} trees={last['n_trees']} bank_trees={last['n_trees_bank']} "
          f"food={sim.world.food_total:.0f} rich={last['n_rich']} eroding={last['n_eroding']} collapsed={last['collapsed']} "
          f"emerged={last['emerged']}  {pops}  quality={last['mean_quality']:.2f}")
    print(f"-> {a.out}/summary.png, snap_*.png, log.csv")


if __name__ == "__main__":
    main()

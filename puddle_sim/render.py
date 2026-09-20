"""Pictures. Snapshots of the map and time-series summaries."""
from __future__ import annotations

import os
from typing import List

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = ["PingFang TC", "Hiragino Sans", "Arial Unicode MS", "DejaVu Sans"]
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .crops import TABLE as CROPS  # noqa: E402
from .sim import Simulation  # noqa: E402

# Palette (RGB 0..1). Deliberate: damp, mossy.
LAND = np.array([0.62, 0.66, 0.42])
TREE = np.array([0.16, 0.36, 0.20])
FARM = np.array([0.45, 0.30, 0.17])
WATER_CLEAN = np.array([0.30, 0.55, 0.75])
WATER_DIRTY = np.array([0.45, 0.42, 0.28])
RICH = np.array([0.30, 0.26, 0.16])       # 分解者土壤: dark
MANGROVE = np.array([0.20, 0.55, 0.45])   # 紅樹林: teal-green on water
ROCK_C = np.array([0.56, 0.56, 0.53])
SAND_C = np.array([0.84, 0.78, 0.56])
MUD_C = np.array([0.46, 0.41, 0.34])
ERODING = np.array([0.72, 0.52, 0.40])    # 流失中: raw, reddish


def composite(sim: Simulation) -> np.ndarray:
    w = sim.world
    H, W = w.cfg.height, w.cfg.width
    img = np.empty((H, W, 3), dtype=np.float32)
    # Land shaded by height: high = pale, low = darker. Reads as relief without a hillshade.
    b = w.hydro.b
    rel = (b - b.min()) / max(1e-6, (b.max() - b.min()))
    img[:] = LAND[None, None, :] * (0.75 + 0.35 * rel[..., None])
    sub = w.substrate
    img[sub == 1] = ROCK_C * (0.8 + 0.3 * rel[sub == 1, None])
    img[sub == 2] = SAND_C
    img[sub == 3] = MUD_C
    soil = w.soil
    img[soil.rich] = RICH
    img[soil.eroding & (sub != 1)] = ERODING
    sal = w.saline()
    img[sal] = img[sal] * 0.6 + np.array([0.85, 0.85, 0.9]) * 0.4       # salt crust
    for k, pl in w.flora.table.items():
        m = w.flora.kind == k
        if pl.canopy:
            img[m] = np.array(pl.color)
        else:
            img[m] = img[m] * 0.5 + np.array(pl.color) * 0.5
    for k, c in CROPS.items():
        img[w.farm & (w.crop == k) & ~w.water] = np.array(c.color)
    img[w.farm & soil.rich] = FARM * 0.7
    img[w.farm & soil.eroding] = FARM * 0.7 + ERODING * 0.3
    # Water colour blends by quality; depth deepens it; shade darkens it a bit.
    q = w.quality[..., None]
    cov = w.cover()[..., None]
    depth = np.clip(w.hydro.d / 1.0, 0.0, 1.0)[..., None]
    water_rgb = (WATER_CLEAN * q + WATER_DIRTY * (1 - q)) * (1 - 0.25 * cov) * (1.05 - 0.45 * depth)
    algae = np.clip(w.algae / 1.0, 0.0, 1.0)[..., None]
    water_rgb = water_rgb * (1 - 0.5 * algae) + np.array([0.35, 0.60, 0.20]) * 0.5 * algae
    img[w.water] = water_rgb[w.water]
    img[w.substrate == 4] = np.array([0.92, 0.92, 0.90])                 # salt crust
    cal = w.farm & (w.crop == 3) & w.water
    img[cal] = np.array(CROPS[3].color)
    for k, pl in w.flora.table.items():
        if pl.aquatic:
            m = (w.flora.kind == k)
            img[m] = np.array(pl.color) if pl.canopy else img[m] * 0.4 + np.array(pl.color) * 0.6
    return np.clip(img, 0.0, 1.0)


def flow_map(sim: Simulation, out_dir: str) -> str:
    """Speed heatmap with arrows — the water physics on its own."""
    os.makedirs(out_dir, exist_ok=True)
    hy = sim.world.hydro
    H, W = sim.cfg.height, sim.cfg.width
    fig, ax = plt.subplots(figsize=(8 * W / max(W, H), 8 * H / max(W, H)))
    speed = np.where(sim.world.water, hy.speed, np.nan)
    im = ax.imshow(speed, cmap="viridis", vmin=0, vmax=max(0.5, float(np.nanpercentile(speed, 95))),
                   interpolation="nearest")
    fig.colorbar(im, ax=ax, fraction=0.046, label="speed (cells / time)")
    ys, xs = np.mgrid[0:speed.shape[0]:2, 0:speed.shape[1]:2]
    vx, vy = hy.vx[::2, ::2], hy.vy[::2, ::2]
    m = sim.world.water[::2, ::2]
    ax.quiver(xs[m], ys[m], vx[m], -vy[m], color="white", scale=40, width=0.004, alpha=0.8)
    ax.set_title(f"tick {sim.tick}  water {int(sim.world.water.sum())} cells  volume {hy.volume():.0f}  "
                 f"out {hy.outflow_tick:.3f}/tick", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    path = os.path.join(out_dir, f"flow_{sim.tick:05d}.png")
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return path


def snapshot(sim: Simulation, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    H, W = sim.cfg.height, sim.cfg.width
    fig, ax = plt.subplots(figsize=(8 * W / max(W, H), 8 * H / max(W, H)))
    ax.imshow(composite(sim), interpolation="nearest")
    for sp in sim.species:
        pop = sim.pops[sp.key]
        if not pop:
            continue
        ys = [c.pos[0] for c in pop]
        xs = [c.pos[1] for c in pop]
        sat = [c.satisfaction for c in pop]
        ax.scatter(xs, ys, c=sat, cmap="RdYlGn", vmin=0, vmax=1, s=22, marker=sp.marker,
                   edgecolors="k", linewidths=0.3, label=f"{sp.name} {len(pop)}")
    wood = np.argwhere(sim.world.soil.wood > 0.05)
    if len(wood):
        ax.scatter(wood[:, 1], wood[:, 0], marker="x", c="#3a2510", s=18, linewidths=1.2, label=f"倒木 {len(wood)}")
    from .industry import TABLE as IND
    for (y, x, k, owner) in sim.industries.sites():
        ax.text(x, y, IND[k].glyph, fontsize=7, ha="center", va="center", color="white",
                bbox=dict(facecolor="#333", alpha=0.85, pad=0.6, lw=0))
    for i, f in enumerate(sim.farmers):
        hy, hx = f.home
        ax.plot(hx, hy, marker="s", color="white" if i == 0 else "#ffd166", markersize=7, markeredgecolor="k")
        led = f.ledger
        ax.text(hx + 1, hy - 1, f"{f.name} 食{f.food_total:.0f} 木{led['wood']:.0f} 鹽{led['salt']:.0f} 炭{led['charcoal']:.0f} 力{led['power']:.0f}", fontsize=6, color="white",
                bbox=dict(facecolor="black", alpha=0.5, pad=1, lw=0))
    ax.legend(loc="lower left", fontsize=8, framealpha=0.8)
    totals = "  ".join(f"{sp.name} {sim.yield_total[sp.key]:.0f}" for sp in sim.species)
    so = sim.world.soil
    totals += f"  |  紅樹林 {int(sim.world.mangrove.sum())} 造陸 {sim.world.mg_built}"
    ax.set_title(f"tick {sim.tick}  farm {int(sim.world.farm.sum())}  trees {int(sim.world.tree.sum())}  "
                 f"rich {int(so.rich.sum())}  eroding {int((so.eroding & ~sim.world.water).sum())}  "
                 f"collapsed {so.collapsed_total}\nfood {sim.world.food_total:.0f}  |  {totals}", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    path = os.path.join(out_dir, f"snap_{sim.tick:05d}.png")
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return path


def summary(sim: Simulation, out_dir: str) -> str:
    log = sim.log
    t = [r["tick"] for r in log]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    ax = axes[0, 0]
    for sp in sim.species:
        ax.plot(t, [r[f"n_{sp.key}"] for r in log], label=sp.name)
    ax.plot(t, [r["n_farm"] for r in log], label="farm tiles", color="tab:brown")
    ax.plot(t, [r["n_trees"] for r in log], label="trees", color="tab:green")
    ax.plot(t, [r["n_trees_bank"] for r in log], label="bank trees", color="tab:green", ls="--")
    ax.set_title("population vs farmland"); ax.legend(fontsize=8)
    ax = axes[0, 2]
    ax.plot(t, [r["n_rich"] for r in log], label="rich soil cells", color="#3a2a10")
    ax.plot(t, [r["n_eroding"] for r in log], label="eroding cells", color="tab:red")
    ax.plot(t, [r["wood"] * 10 for r in log], label="deadwood mass x10", color="tab:olive", ls=":")
    ax.set_title("soil states"); ax.legend(fontsize=8)
    ax = axes[1, 2]
    ax.plot(t, [r["food_tick"] for r in log], label="farm food / tick", color="tab:brown")
    ax2 = ax.twinx()
    ax2.plot(t, [r["collapsed"] for r in log], label="bank cells collapsed", color="tab:blue")
    ax2.plot(t, [r["emerged"] for r in log], label="cells built by river", color="tab:cyan")
    ax.set_title("farm food  |  land lost / built"); ax.legend(fontsize=8, loc="upper left"); ax2.legend(fontsize=8, loc="upper right")
    ax = axes[0, 1]
    for sp in sim.species:
        ax.plot(t, [r[f"sat_{sp.key}"] for r in log], label=f"sat {sp.name}")
    ax.plot(t, [r["mean_quality"] for r in log], label="water quality", ls=":")
    ax.plot(t, [r["mean_cover"] for r in log], label="cover", ls=":")
    ax.plot(t, [r["mean_food"] for r in log], label="food", ls=":")
    ax.set_ylim(0, 1.05); ax.set_title("habitat (means over water)"); ax.legend(fontsize=8)
    ax = axes[1, 0]
    for sp in sim.species:
        ax.plot(t, [r[f"yield_{sp.key}"] for r in log], label=sp.name)
    ax.set_title("yield per tick"); ax.legend(fontsize=8)
    ax = axes[1, 1]
    for sp in sim.species:
        ax.plot(t, [r[f"total_{sp.key}"] for r in log], label=sp.name)
    ax.set_title("cumulative yield"); ax.legend(fontsize=8)
    for a in axes.flat:
        a.set_xlabel("tick")
    fig.suptitle(f"buffer={sim.cfg.buffer}  farmer={sim.cfg.farmer_mode}  seed={sim.cfg.seed}")
    fig.tight_layout()
    path = os.path.join(out_dir, "summary.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def sweep_plot(results: List[dict], out_dir: str, species_keys: List[str]) -> str:
    """results: [{buffer, seed, n_farm, yield_<key>...}, ...]"""
    os.makedirs(out_dir, exist_ok=True)
    buffers = sorted({r["buffer"] for r in results})

    def agg(key):
        return [np.mean([r[key] for r in results if r["buffer"] == b]) for b in buffers]

    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    colors = {"shrimp": "tab:red", "medaka": "tab:orange"}
    for k in species_keys:
        ax1.plot(buffers, agg(f"yield_{k}"), marker="o", color=colors.get(k, None), label=f"{k} yield")
    ax1.set_xlabel("riparian buffer (tiles left untouched next to water)")
    ax1.set_ylabel("creature yield (total)")
    ax1.legend(loc="upper left", fontsize=8)
    ax2 = ax1.twinx()
    ax2.plot(buffers, agg("n_farm"), marker="s", color="tab:brown", label="farm tiles at end")
    ax2.set_ylabel("farm tiles", color="tab:brown")
    ax1.set_title("what the buffer costs and what it buys")
    fig.tight_layout()
    path = os.path.join(out_dir, "sweep.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path

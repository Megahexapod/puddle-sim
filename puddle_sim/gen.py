"""Skeleton generator: the platter strung along the river (D-018, generation.md).

The map is a river. Its length is cut into reaches by the platter's proportions, in the only order
a river allows:  highland -> canyon -> hills -> floodplain -> delta -> sea.  Each reach shapes its
own valley (width, depth, gradient, roughness). Attachments hang off specific reaches: a pond on
the hills/floodplain, a swamp beside the floodplain, a tributary coming in from the map's side.
Turn a piece off (proportion 0 / flag 0) and the river simply skips it; turn the sea off and the
river leaves the map through a plain east edge with no tide.

Everything here is the skeleton. Water, salt, forest and mangroves are the runtime rules run to
quiet (World.__init__); history is the game.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from .config import Config


def smooth_noise(rng: np.random.Generator, H: int, W: int, cells: float, amp: float) -> np.ndarray:
    """Bilinear-upsampled gaussian grid: cheap smooth terrain noise with feature size ~cells."""
    gh, gw = int(H / cells) + 2, int(W / cells) + 2
    g = rng.normal(0.0, 1.0, (gh, gw))
    ys = np.linspace(0, gh - 1.001, H); xs = np.linspace(0, gw - 1.001, W)
    y0 = ys.astype(int); x0 = xs.astype(int)
    fy = (ys - y0)[:, None]; fx = (xs - x0)[None, :]
    v = (g[y0][:, x0] * (1 - fy) * (1 - fx) + g[y0 + 1][:, x0] * fy * (1 - fx)
         + g[y0][:, x0 + 1] * (1 - fy) * fx + g[y0 + 1][:, x0 + 1] * fy * fx)
    return (amp * v).astype(np.float32)


# Reach types in river order. (gradient x, valley width, valley depth, hinterland roughness x)
REACH = {
    "highland":   dict(grad=2.2, width=1.3, depth=1.1, rough=1.7),
    "canyon":     dict(grad=1.4, width=1.0, depth=2.6, rough=1.2),
    "hills":      dict(grad=1.0, width=2.2, depth=1.5, rough=1.0),
    "floodplain": dict(grad=0.30, width=4.5, depth=0.7, rough=0.25),
    "delta":      dict(grad=0.25, width=3.0, depth=0.6, rough=0.15),
}
ORDER = ["highland", "canyon", "hills", "floodplain", "delta"]


SOIL, ROCK, SAND, MUD, SALT = 0, 1, 2, 3, 4


@dataclass
class Skeleton:
    b: np.ndarray                      # terrain heights
    substrate: np.ndarray              # int8: SOIL / ROCK / SAND / MUD
    springs: List[Tuple[int, int, float]]   # (y, x, flow)
    center: np.ndarray                 # main river centre line y(x)
    segments: List[Tuple[str, int, int]]    # (reach, x0, x1)
    pond_center: Tuple[int, int]
    tributary: np.ndarray              # (n, 2) cells of the tributary centre line, or empty
    has_sea: bool


def _segments(cfg: Config, W: int) -> List[Tuple[str, int, int]]:
    props = {"highland": cfg.seg_highland, "canyon": cfg.seg_canyon, "hills": cfg.seg_hills,
             "floodplain": cfg.seg_floodplain, "delta": cfg.seg_delta if cfg.sea else 0.0}
    active = [(k, props[k]) for k in ORDER if props[k] > 0]
    total = sum(p for _, p in active)
    out, x = [], 0
    for i, (k, p) in enumerate(active):
        x1 = W if i == len(active) - 1 else min(W, x + int(round(W * p / total)))
        out.append((k, x, x1))
        x = x1
    return out


def _per_column(segments, W: int, key: str, blend: int = 4) -> np.ndarray:
    """A reach parameter as a smooth function of x (linear blend across boundaries)."""
    v = np.zeros(W, dtype=np.float32)
    for k, x0, x1 in segments:
        v[x0:x1] = REACH[k][key]
    if blend > 1:
        kern = np.ones(blend) / blend
        v = np.convolve(np.pad(v, blend, mode="edge"), kern, mode="same")[blend:-blend]
    return v.astype(np.float32)


def build(cfg: Config, rng: np.random.Generator) -> Skeleton:
    H, W = cfg.height, cfg.width
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
    segs = _segments(cfg, W)
    grad = _per_column(segs, W, "grad")
    width = _per_column(segs, W, "width")
    depth = _per_column(segs, W, "depth")
    rough = _per_column(segs, W, "rough")

    # Base profile: integrate the gradient from the sea end westward, so the profile is continuous
    # and steeper reaches really are steeper.
    base = np.cumsum((cfg.slope * grad)[::-1])[::-1]
    b = np.repeat(base[None, :], H, axis=0).astype(np.float32)

    # Hinterland: two octaves of smooth noise, scaled by each reach's roughness.
    b += (smooth_noise(rng, H, W, 14, cfg.noise_amp) + smooth_noise(rng, H, W, 5, cfg.noise_amp * 0.4)) * rough[None, :]

    # Main river centre line: a sine meander plus a smoothed random walk.
    phase = rng.uniform(0, 2 * np.pi)
    walk = np.cumsum(rng.normal(0, 0.6, W))
    walk = np.convolve(walk, np.ones(7) / 7, mode="same")
    center = H / 2 + cfg.meander_amp * (H / 48) * np.sin(2 * np.pi * xs[0] / (W * 0.85) + phase) + walk
    center = np.clip(center, 6, H - 7).astype(np.float32)
    b -= depth[None, :] * np.exp(-((ys - center[None, :]) / width[None, :]) ** 2)
    b_hillside = b.copy()          # substrate reads the valley sides, not the channel cut below
    # A narrow channel inside the valley, so a wide shallow floodplain still carries one river.
    b -= cfg.channel_depth * np.exp(-((ys - center[None, :]) / 1.0) ** 2)

    springs = [(int(round(center[1])), 1, float(cfg.spring_q))]
    seg_of = {k: (x0, x1) for k, x0, x1 in segs}

    # Pond: a basin off the hills or the floodplain.
    host = "floodplain" if "floodplain" in seg_of else ("hills" if "hills" in seg_of else segs[0][0])
    hx0, hx1 = seg_of[host]
    px = int(rng.integers(hx0 + 3, max(hx0 + 4, hx1 - 3)))
    side = 1 if rng.random() < 0.5 else -1
    py = int(np.clip(center[px] + side * 9, 5, H - 6))
    pond_center = (py, px)
    if cfg.pond:
        b -= cfg.pond_depth * np.exp(-((ys - py) ** 2 + (xs - px) ** 2) / (2 * cfg.pond_sigma ** 2))

    # Swamp: a wide shallow hollow beside the floodplain (freshwater wetland, shaded, slow).
    if cfg.swamp and "floodplain" in seg_of:
        fx0, fx1 = seg_of["floodplain"]
        sx = int(rng.integers(fx0 + 4, max(fx0 + 5, fx1 - 4)))
        sside = -side
        sy = int(np.clip(center[sx] + sside * 7, 4, H - 5))
        b -= 0.55 * np.exp(-((ys - sy) ** 2 / (2 * 3.5 ** 2) + (xs - sx) ** 2 / (2 * 6.0 ** 2)))

    # Tributary: a smaller stream from the map's side that joins the main river in the hills or
    # floodplain. Its own spring, its own narrower valley.
    headland_mask = np.zeros((H, W), dtype=bool)
    trib = np.zeros((0, 2), dtype=int)
    if cfg.tributary:
        host2 = "hills" if "hills" in seg_of else host
        tx0, tx1 = seg_of[host2]
        jx = int(rng.integers(tx0 + 4, max(tx0 + 5, tx1 - 2)))       # junction column
        tside = 1 if rng.random() < 0.5 else -1                       # from south (+) or north (-)
        y_edge = H - 2 if tside > 0 else 1
        jy = int(round(center[jx]))
        n = abs(y_edge - jy)
        path = []
        x = float(jx + rng.integers(-8, 9))                           # starts offset, drifts to the junction
        for i in range(n + 1):
            t = i / max(1, n)                                         # 0 at the edge, 1 at the junction
            yy = int(y_edge - tside * i)
            x += (jx - x) * 0.12 + rng.normal(0, 0.35)
            xx = int(np.clip(round(x), 1, W - 2))
            path.append((yy, xx))
            # valley depth grows toward the junction; blends into the main valley there
            dep = 0.35 + 0.55 * t
            b -= dep * np.exp(-((ys - yy) ** 2 + (xs - xx) ** 2) / (2 * 1.1 ** 2))
        trib = np.array(path, dtype=int)
        sy0, sx0 = path[0]
        springs.append((sy0, sx0, float(cfg.spring_q * cfg.tributary_flow)))

    # Rivers run downhill. Noise and reach changes can leave uphill steps along the centre line;
    # carve them out so the bed never rises going east (a river that pools and spills thin is a
    # broken map, not a feature).
    cy_idx = np.clip(np.round(center).astype(int), 0, H - 1)
    bed = b[cy_idx, np.arange(W)]
    mono = bed.copy()
    for x in range(1, W):
        mono[x] = min(bed[x], mono[x - 1] - cfg.min_drop)
    fix = (bed - mono).astype(np.float32)                     # >= 0 where the bed had to be lowered
    b -= fix[None, :] * np.exp(-((ys - center[None, :]) / 1.3) ** 2)

    # Endorheic basin: no sea; the last reach is a closed bowl below the river's bed. The river
    # pours in, the dry air takes the water back, the salt stays (D-027).
    if cfg.basin and not cfg.sea:
        bx0, bx1 = segs[-1][1], segs[-1][2]
        ys_, xs_ = np.mgrid[0:H, 0:W].astype(np.float32)
        cxb, cyb = (bx0 + bx1) / 2.0, float(center[min(W - 1, (bx0 + bx1) // 2)])
        bowl = np.exp(-(((xs_ - cxb) / ((bx1 - bx0) * 0.45)) ** 2 + ((ys_ - cyb) / (H * 0.32)) ** 2))
        floor = b[cy_idx[bx0], bx0] - cfg.basin_depth
        b = np.where(bowl > 0.35, np.minimum(b, floor + (1.0 - bowl) * 1.5 + smooth_noise(rng, H, W, 6, 0.15)), b)
        b[:, W - 1] = np.maximum(b[:, W - 1], b[:, W - 1].max() + 2.0)   # a wall: nothing leaves

    # Coast: last reach becomes a plain around sea level with its own relief (lagoon / intertidal /
    # dry decided by elevation), the channel fades and fans out, a sand ridge with one gap.
    if cfg.sea and "delta" in seg_of:
        dx0, dx1 = seg_of["delta"]
        ncols = dx1 - dx0
        relief = smooth_noise(rng, H, W, 9, cfg.coast_undulation)
        plain = cfg.sea_level + cfg.delta_height + relief
        # D-038: the delta is a ramp from the floodplain's own level at its head down to the plain,
        # not a step (the old cap cut a cliff along the delta's first column -- the other straight
        # line). The channel is kept near the head and fans out toward the sea.
        head = b[:, dx0 - 1].copy() if dx0 > 0 else b[:, 0].copy()
        head = np.maximum(head, plain[:, dx0] + 0.3)
        for x in range(dx0, dx1):
            f = (x - dx0 + 1) / ncols
            g = f * f * (3 - 2 * f)                                     # smoothstep: flat-ish at both ends
            target = head * (1 - g) + plain[:, x] * g
            col = np.minimum(b[:, x], target)
            b[:, x] = target - (target - col) * (1 - f) ** 2
        # D-038: no painted ridge, no column cut. The plain keeps its relief all the way east and
        # bends down a quadratic shelf (gentle inshore, steep offshore) to open water at the edge.
        # The coastline is wherever that surface crosses sea level -- a noisy contour, not a column.
        # The beach ridge is grown at runtime (world.py: swash deposition + dune grass), and the
        # river keeps its own mouth open by scouring it.
        xs0 = W - cfg.shelf_cols
        for x in range(xs0, W):
            f = (x - xs0 + 1) / cfg.shelf_cols
            b[:, x] -= cfg.shelf_drop * f * f
        b[:, W - 1] = np.minimum(b[:, W - 1], cfg.sea_level - 1.0)           # the edge is always open sea
        cy = int(round(center[W - 1]))
        # A rocky headland: a hard bump on the coast away from the river mouth (岩灘 lives on it).
        if cfg.headland:
            hy_ = int(np.clip(cy + (16 if rng.random() < 0.5 else -16), 4, H - 5))
            hx_ = W - cfg.shelf_cols // 2
            bump = 1.1 * np.exp(-((ys - hy_) ** 2 + (xs - hx_) ** 2) / (2 * 3.0 ** 2))
            b += bump
            headland_mask = bump > 0.25

    sub = substrate(cfg, b_hillside, segs, rng)
    if cfg.sea and "delta" in seg_of:          # the coast was reshaped after the copy: redo its sand/mud
        dx0 = seg_of["delta"][0]
        hi = cfg.sea_level + cfg.tide_amp
        coast = np.zeros((H, W), dtype=bool); coast[:, dx0:] = True
        sub[coast & (b >= hi)] = SAND
        sub[coast & (b < hi)] = MUD
        sub[headland_mask] = ROCK
    return Skeleton(b=b.astype(np.float32), substrate=sub, springs=springs, center=center, segments=segs,
                    pond_center=pond_center, tributary=trib, has_sea=bool(cfg.sea))


def substrate(cfg: Config, b: np.ndarray, segments, rng: np.random.Generator) -> np.ndarray:
    """底質 from the skeleton: steep ground is rock (threshold per reach, noisy so outcrops are
    patches), the coastal plain is sand above high tide and mud below it, the rest is soil."""
    H, W = b.shape
    gy, gx = np.gradient(b)
    slope = np.sqrt(gy ** 2 + gx ** 2)
    thr = np.full(W, 1e9, dtype=np.float32)
    table = {"highland": cfg.rock_slope_highland, "canyon": cfg.rock_slope_canyon,
             "hills": cfg.rock_slope_hills, "floodplain": cfg.rock_slope_floodplain, "delta": 1e9}
    for k, x0, x1 in segments:
        thr[x0:x1] = table[k]
    thr = thr[None, :] + smooth_noise(rng, H, W, 6, cfg.rock_patch)
    sub = np.full((H, W), SOIL, dtype=np.int8)
    sub[slope > thr] = ROCK
    if cfg.sea:
        dx0 = [x0 for k, x0, x1 in segments if k == "delta"]
        if dx0:
            hi = cfg.sea_level + cfg.tide_amp
            coast = np.zeros((H, W), dtype=bool); coast[:, dx0[0]:] = True
            sub[coast & (b >= hi)] = SAND
            sub[coast & (b < hi)] = MUD
    return sub


# ---------------------------------------------------------------------------- invariants

def check(world) -> Dict[str, object]:
    """Does this map have what the platter promised? Returns a report; 'ok' is the verdict."""
    cfg = world.cfg
    # Judge on the tide-averaged depth: at low tide the mouth thins below d_min and the river
    # "disconnects" from the sea for a few hundred ticks — that is the tide, not a broken map.
    w = world.hydro.d_ema >= cfg.d_min
    H, W = w.shape
    sk = world.skeleton
    rep: Dict[str, object] = {}

    # River connectivity: from the main spring to the east edge, through water (8-neighbour).
    from collections import deque
    sy, sx, _ = sk.springs[0]
    seen = np.zeros_like(w)
    start = None
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            yy, xx = sy + dy, sx + dx
            if 0 <= yy < H and 0 <= xx < W and w[yy, xx]:
                start = (yy, xx); break
        if start: break
    reached_east = False
    goal_x = (sk.segments[-1][1] + 2) if (cfg.basin and not cfg.sea) else W - 1   # into the bowl, or out to sea
    if start:
        q = deque([start]); seen[start] = True
        while q:
            y, x = q.popleft()
            if x >= goal_x:
                reached_east = True
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < H and 0 <= nx < W and w[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True; q.append((ny, nx))
    rep["river_connected"] = reached_east
    rep["main_water_cells"] = int(seen.sum())

    if cfg.pond:
        py, px = sk.pond_center
        rep["pond_wet"] = int(w[max(0, py - 4): py + 5, max(0, px - 4): px + 5].sum())
    if cfg.tributary and len(sk.tributary):
        tw = sum(1 for y, x in sk.tributary if w[y, x])
        rep["tributary_wet_frac"] = round(tw / len(sk.tributary), 2)
        rep["tributary_joined"] = bool(seen[sk.tributary[-1][0], sk.tributary[-1][1]])
    if sk.has_sea:
        b = world.hydro.b
        lo, hi = cfg.sea_level - cfg.tide_amp, cfg.sea_level + cfg.tide_amp
        dx0 = [x0 for k, x0, x1 in sk.segments if k == "delta"][0]
        coast = np.zeros_like(w); coast[:, dx0:] = True
        rep["coast_lagoon"] = int((coast & (b < lo)).sum())
        rep["coast_intertidal"] = int((coast & (b >= lo) & (b < hi)).sum())
        rep["coast_dry"] = int((coast & (b >= hi)).sum())
    # Home: land next to water with some forest within 6 and >= 20 farmable cells within reach.
    hy_, hx_ = world.farmer_home if hasattr(world, "farmer_home") else (None, None)
    rep["forest_cells"] = int(world.tree.sum())
    rep["farmable_cells"] = int(((~w) & (world.water_dist <= cfg.water_reach) & (world.water_dist > 0)).sum())

    ok = rep["river_connected"] and rep["farmable_cells"] >= 60
    if cfg.pond:
        ok = ok and rep["pond_wet"] >= 6
    if cfg.tributary and len(sk.tributary):
        ok = ok and rep["tributary_wet_frac"] >= 0.5 and rep["tributary_joined"]
    if sk.has_sea:
        ok = ok and rep["coast_intertidal"] >= 40 and rep["coast_dry"] >= 20
    rep["ok"] = bool(ok)
    return rep

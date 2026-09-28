#!/usr/bin/env python3
"""puddle — watch it run, poke it.

  python3 play.py            -> http://localhost:7748

The sim runs in a background thread at an adjustable ticks/second. The page draws the grid on a
canvas, shows the numbers, and lets you click tools onto the map (tree / log / field / dam / dig).
Idle-friendly: leave it running and glance at it.
"""
from __future__ import annotations

import argparse
import base64
import os
import sys
import threading
import time

import numpy as np
from flask import Flask, jsonify, request, Response

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from puddle_sim.config import Config  # noqa: E402
from puddle_sim.flora import BARE  # noqa: E402
from puddle_sim.sim import Simulation  # noqa: E402

app = Flask(__name__)
LOCK = threading.Lock()
STATE = {"sim": None, "running": False, "tps": 30.0, "cfg": {}}


def make_sim(overrides: dict) -> Simulation:
    cfg = Config()
    for k, v in overrides.items():
        if hasattr(cfg, k):
            cur = getattr(cfg, k)
            try:
                setattr(cfg, k, type(cur)(v) if not isinstance(cur, bool) else bool(v))
            except (TypeError, ValueError):
                pass
    return Simulation(cfg)


def runner():
    while True:
        if STATE["running"] and STATE["sim"] is not None:
            with LOCK:
                STATE["sim"].step()
            time.sleep(1.0 / max(1.0, STATE["tps"]))
        else:
            time.sleep(0.05)


def u8(a, scale, cap=255):
    """uint8 array as base64 — a 96x64 map is ~6k cells per layer; lists of ints were too fat."""
    return base64.b64encode(np.clip(np.asarray(a, dtype=np.float32) * scale, 0, cap).astype(np.uint8).ravel().tobytes()).decode("ascii")


def bits(a):
    return base64.b64encode(np.asarray(a).astype(np.uint8).ravel().tobytes()).decode("ascii")


@app.route("/state")
def state():
    sim = STATE["sim"]
    if sim is None:
        return jsonify({"empty": True})
    with LOCK:
        w = sim.world
        hy = w.hydro
        so = w.soil
        b = hy.b
        rel = (b - b.min()) / max(1e-6, float(b.max() - b.min()))
        lo, hi = w.cfg.sea_level - w.cfg.tide_amp, w.cfg.sea_level + w.cfg.tide_amp
        tidal = np.where(b < lo, 1, np.where(b < hi, 2, 0)).astype(np.uint8)   # 1 lagoon, 2 intertidal
        soil_state = np.where(so.rich, 2, np.where(so.eroding, 1, 0)).astype(np.uint8)
        creatures = []
        for sp in sim.species:
            for c in sim.pops[sp.key]:
                creatures.append([int(c.pos[0]), int(c.pos[1]), round(float(c.satisfaction), 2), sp.key, round(c.genes.get("color", 0.0), 2) if c.genes else 0])
        stats = {
            "tick": sim.tick,
            "farm": int(w.farm.sum()),
            "trees": int(w.tree.sum()),
            "bank_trees": int((w.tree & (w.water_dist == 1)).sum()),
            "water": int(w.water.sum()),
            "rich": int((so.rich & ~w.water).sum()),
            "eroding": int((so.eroding & ~w.water).sum()),
            "collapsed": so.collapsed_total,
            "emerged": so.emerged_total,
            "wood": round(float(so.wood.sum()), 1),
            "food_total": round(w.food_total),
            "food_tick": round(w.food_tick, 3),
            "quality": round(float(w.quality[w.water].mean()) if w.water.any() else 1.0, 2),
            "volume": round(hy.volume(), 1),
            "pops": {sp.key: len(sim.pops[sp.key]) for sp in sim.species},
            "yield": {sp.key: round(sim.yield_total[sp.key]) for sp in sim.species},
            "replanted": sim.farmer.replanted,
            "moved": round(sim.farmer.moved, 1),
            "mangrove": int(w.mangrove.sum()),
            "mg_built": w.mg_built,
            "tide": round(hy.sea_head, 2),
            "sea_level": w.cfg.sea_level,
            "gen": w.gen_report,
            "flora": w.flora.counts(),
            "wild": w.wild.counts(),
            "wild_total": {k: round(v, 1) for k, v in zip(["蕨", "香菇", "草菇", "松茸", "栗", "柿", "河蚌"], w.wild.offered_total.values())},
            "shells": w.shells,
            "genes": sim.gene_stats(),
            "birds": w.eaten_by_birds,
            "season": round(w.climate.season, 2),
            "temp_mean": round(float(w.temp.mean()), 1),
            "snowpack": round(w.climate.snowpack, 1),
            "spring_mult": round(w.climate.spring_mult, 2),
            "saline": int(w.saline().sum()),
            "storms": w.climate.storms_total,
            "storm_on": w.climate.storm is not None,
            "gw_total": round(float(w.ground.G.sum()), 1),
            "springs": int((w.ground.exfil[~w.water] > 1e-5).sum()),
            "paths": int((w.ground.pack > w.cfg.pack_grass).sum()),
            "rock": int((w.substrate == 1).sum()),
            "mode": w.cfg.mode,
            "mc_events": getattr(w.ground, "events", 0),
        }
        payload = {
            "H": w.cfg.height, "W": w.cfg.width,
            "rel": u8(rel, 255),
            "d": u8(hy.d, 100),
            "q": u8(w.quality, 255),
            "speed": u8(hy.speed, 60),
            "tree": bits(w.tree),
            "farm": bits(w.farm),
            "water": bits(w.water),
            "soil": bits(soil_state),
            "h": u8(so.h, 170),
            "wood": u8(so.wood, 100),
            "light": bits(w.light),
            "mangrove": bits(w.mangrove),
            "salt": u8(w.salt_ema, 255),
            "tidal": bits(tidal),
            "sub": bits(w.substrate),
            "kind": bits(w.flora.kind),
            "temp": u8(w.temp + 10.0, 5),        # -10..41 -> 0..255
            "algae": u8(w.algae, 200),
            "o2": u8(w.oxygen, 255),
            "crop": bits(w.crop),
            "wild": bits(w.wild.kind),            # D-037: 0 none 1 蕨 2 香菇 3 草菇 4 松茸
            "snow": bits(w.climate.snow),
            "ssal": u8(w.soil_salinity(), 255),
            "elev": u8((b - lo + 1.0), 60),      # elevation relative to low tide, 60/unit (sea level = 60+27)
            "rain": u8(w.climate.pattern, 80),   # spatial rain factor, 1.0 = 80 (D-029)
            "gw": u8(w.ground.saturation(w), 255),
            "wt": u8(np.clip(b - w.ground.water_table(w), 0.0, 2.0), 100),   # water table depth below the surface
            "pack": u8(w.ground.pack, 255),
            "infil": u8(w.ground.infil_frac, 255),
            "exfil": bits(w.ground.exfil > 1e-5),
            "gwl": bits(getattr(w.ground, "level", np.zeros_like(w.substrate))),      # MC mode: 0..3
            "rainc": bits(getattr(w.ground, "rain_class", np.ones_like(w.substrate))),
            "home": [int(sim.farmer.home[0]), int(sim.farmer.home[1])],
            "homes": [[int(f.home[0]), int(f.home[1]), f.name, round(f.food_total),
                       {k: round(v, 1) for k, v in f.ledger.items()}] for f in sim.farmers],
            "industries": [[y, x, k, o] for (y, x, k, o) in sim.industries.sites()],
            "site": u8(sim.site_score, 255),
            "creatures": creatures,
            "stats": stats,
            "running": STATE["running"],
            "tps": STATE["tps"],
            "cfg": STATE["cfg"],
        }
    return jsonify(payload)


@app.route("/control", methods=["POST"])
def control():
    j = request.get_json(force=True) or {}
    if "running" in j:
        STATE["running"] = bool(j["running"])
    if "tps" in j:
        STATE["tps"] = float(j["tps"])
    if j.get("reset"):
        with LOCK:
            STATE["cfg"] = j.get("cfg", {})
            STATE["sim"] = make_sim(STATE["cfg"])
    if j.get("step"):
        with LOCK:
            for _ in range(int(j.get("n", 1))):
                STATE["sim"].step()
    return jsonify({"ok": True})


@app.route("/steps", methods=["POST"])
def steps():
    """Fold a Minecraft footstep log (puddle_steps.csv) into the tread field (D-030)."""
    from puddle_sim.steps import fold
    j = request.get_json(force=True) or {}
    path = os.path.expanduser(j.get("path", ""))
    sim = STATE["sim"]
    if sim is None or not os.path.exists(path):
        return jsonify({"ok": False, "error": "file not found"})
    with LOCK:
        rep = fold(sim.world, path, amount=float(j["amount"]) if j.get("amount") else None)
    return jsonify({"ok": True, **rep})


@app.route("/act", methods=["POST"])
def act():
    j = request.get_json(force=True) or {}
    tool, y, x = j.get("tool"), int(j.get("y", -1)), int(j.get("x", -1))
    sim = STATE["sim"]
    if sim is None or not (0 <= y < sim.cfg.height and 0 <= x < sim.cfg.width):
        return jsonify({"ok": False})
    with LOCK:
        w = sim.world
        if tool == "tree" and not w.water[y, x]:
            w.plant_tree(y, x)
        elif tool == "wood":
            w.soil.wood[y, x] += sim.cfg.tree_wood
        elif tool == "field" and not w.water[y, x]:
            w.till(y, x)
            if (y, x) not in sim.farmer.fields:
                sim.farmer.fields.append((y, x))
        elif tool == "dam":                       # 填: one block (D-023: 0.25 height unit), loosely packed
            w.ground.fill(y, x, w, 0.25)
        elif tool == "dig":                       # 鏟: one block down; through the soil it takes the rock too
            w.ground.dig(y, x, w, 0.25)
        elif tool == "walk":                      # 腳: a crossing
            w.ground.tread(y, x, y, x, w, amount=0.1)
        elif tool == "tamp":                      # 夯: a road bed / dike core — no rain gets in
            w.ground.pack[y, x] = 1.0
        elif tool == "clear":
            w.flora.kind[y, x] = BARE; w.farm[y, x] = False; w.soil.wood[y, x] = 0.0; w.recompute_light()
        elif tool == "cover":
            w.plant_cover(y, x)
        elif tool == "richsoil" and not w.water[y, x]:
            w.soil.h[y, x] = min(sim.cfg.h_cap, w.soil.h[y, x] + 0.5)
    return jsonify({"ok": True})


PAGE = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>puddle</title>
<style>
 body{margin:0;background:#1b1d1a;color:#ddd;font:13px/1.4 -apple-system,"PingFang TC",sans-serif;display:flex;gap:14px;padding:12px}
 canvas{image-rendering:pixelated;border:1px solid #444;cursor:crosshair}
 .panel{width:320px}
 button{background:#2c2f2a;color:#ddd;border:1px solid #555;padding:4px 8px;margin:2px;cursor:pointer;border-radius:3px}
 button.on{background:#5a6b3a;border-color:#8fa35a}
 .row{margin:6px 0}
 .stat{display:grid;grid-template-columns:1fr 1fr;gap:2px 10px;font-variant-numeric:tabular-nums}
 .stat b{color:#fff;font-weight:500}
 label{margin-right:6px}
 input[type=number]{width:60px;background:#2c2f2a;color:#ddd;border:1px solid #555}
 select{background:#2c2f2a;color:#ddd;border:1px solid #555}
 .legend span{display:inline-block;width:12px;height:12px;vertical-align:middle;margin-right:4px;border:1px solid #333}
 h3{margin:10px 0 4px;font-size:13px;color:#aaa;text-transform:uppercase;letter-spacing:.05em}
</style></head><body>
<div><canvas id="c" width="768" height="512"></canvas>
<div class="row legend">
 <span style="background:#4c8cbf"></span>水（深＝暗，濁＝褐） <span style="background:#295c33"></span>樹 <span style="background:#73472b"></span>田
 <span style="background:#4d4229"></span>分解者土壤 <span style="background:#b88566"></span>流失中 &nbsp; <span style="background:#338c73"></span>紅樹林 ○蝦子 △鱂魚 ▢螃蟹 ▽蘋果螺 ◇龍蝨 ·鹵蟲（紅→綠＝達成度） ×倒木 □家<br>
 植被：<span style="background:#738038"></span>胡楊（旱地河樹） <span style="background:#6e5537"></span>踏痕（越深越硬） <span style="background:#1f4d3d"></span>沼澤林 <span style="background:#8c9e52"></span>鹽沼草 <span style="background:#9ead66"></span>濱草 <span style="background:#8c9e59"></span>草 <span style="background:#338066"></span>海草 <span style="background:#59734d"></span>岩灘藻 &nbsp; 底質：<span style="background:#8f8f87"></span>岩 <span style="background:#d6c78f"></span>沙 <span style="background:#756957"></span>泥 &nbsp; 地形/潮：<span style="background:#1e3c8c"></span>低潮線以下（潟湖） <span style="background:#7aa0c8"></span>潮間帶（低→高） <span style="background:#a0a860"></span>高潮線以上（越亮越高）
</div></div>
<div class="panel">
 <div class="row"><button id="play">▶ 跑</button><button id="step">＋1</button><button id="step100">＋100</button>
  <label>tick/s <input id="tps" type="number" value="30" min="1" max="500"></label></div>
 <h3>看什麼</h3>
 <div class="row" id="views"><button data-v="normal" class="on">地圖</button><button data-v="q">水質</button><button data-v="speed">流速</button><button data-v="h">土 h</button><button data-v="light">光</button><button data-v="salt">鹽度</button><button data-v="terrain">地形/潮</button><button data-v="sub">底質</button><button data-v="flora">植被</button><button data-v="ssal">土鹽</button><button data-v="site">選址</button><button data-v="temp">溫度</button><button data-v="algae">藻</button><button data-v="o2">氧</button><button data-v="rain">雨</button><button data-v="gw">地下水</button><button data-v="wt">水位深</button><button data-v="pack">踏痕</button><button data-v="infil">入滲</button><button data-v="bits">MC bits</button><button data-v="wild">野味</button></div>
 <h3>點地圖放</h3>
 <div class="row" id="tools"><button data-t="none" class="on">看</button><button data-t="tree">樹</button><button data-t="wood">木頭</button><button data-t="field">田</button><button data-t="cover">護坡</button><button data-t="richsoil">肥土</button><button data-t="dam">填一方塊</button><button data-t="dig">鏟一方塊</button><button data-t="walk">腳</button><button data-t="tamp">夯</button><button data-t="clear">清</button></div>
 <div class="row"><label>腳步 CSV <input id="steps_path" value="./saves/" style="width:190px;background:#2c2f2a;color:#ddd;border:1px solid #555"></label><button id="steps_btn">匯入腳步</button><span id="steps_out" style="font-size:11px;color:#aaa"></span></div>
 <h3>重開</h3>
 <div class="row">
  <label>seed <input id="seed" type="number" value="1"></label>
  <label>buffer <input id="buffer" type="number" value="2" min="0" max="4"></label>
  <label>tide_amp <input id="tide_amp" type="number" value="0.45" step="0.05"></label><br>
  <label>farmer <select id="farmer_mode"><option>blind</option><option>adaptive</option></select></label>
  <label>expand_every <input id="expand_every" type="number" value="20"></label><br>
  <label>feed_wood_every <input id="feed_wood_every" type="number" value="0"></label>
  <label><select id="feed_mode"><option>shade</option><option>field</option></select></label><br>
  <label>move_soil_every <input id="move_soil_every" type="number" value="0"></label><br>
  <label>模式 <select id="mode"><option>continuous</option><option>mc</option></select></label>
  <label>聚落數 <input id="n_settlements" type="number" value="3" min="1" max="8"></label>
  <label><input id="industries" type="checkbox" checked> 產業</label>
  <label>間距 <input id="site_spacing" type="number" value="18"></label><br>
  <h3>拼盤（河段比例，0＝關）</h3>
  <label>高地 <input id="seg_highland" type="number" value="0.10" step="0.02" min="0"></label>
  <label>峽谷 <input id="seg_canyon" type="number" value="0.12" step="0.02" min="0"></label><br>
  <label>丘陵 <input id="seg_hills" type="number" value="0.28" step="0.02" min="0"></label>
  <label>氾濫平原 <input id="seg_floodplain" type="number" value="0.30" step="0.02" min="0"></label><br>
  <label>三角洲 <input id="seg_delta" type="number" value="0.20" step="0.02" min="0"></label><br>
  <label><input id="sea" type="checkbox" checked> 海</label>
  <label><input id="pond" type="checkbox" checked> 池塘</label>
  <label><input id="swamp" type="checkbox" checked> 沼澤</label>
  <label><input id="tributary" type="checkbox" checked> 支流</label>
  <label><input id="basin" type="checkbox"> 內流盆地</label>
  <label>evap× <input id="evap_mult" type="number" value="1" step="0.5" min="0.5"></label><br>
  <h3>天空（D-029）</h3>
  <label>風從 <select id="wind"><option>E</option><option>W</option><option>N</option><option>S</option></select></label>
  <label>南北 <input id="rain_ns" type="number" value="0" step="0.25" min="-1" max="1"></label>
  <label>地形雨 <input id="orog_k" type="number" value="0.2" step="0.1" min="0"></label><br>
  <label>暴雨機率 <input id="storm_prob" type="number" value="0" step="0.0005" min="0"></label>
  <label>×強度 <input id="storm_mult" type="number" value="15" step="5"></label>
  <label>雨量× <input id="rain_x" type="number" value="1" step="0.25" min="0"></label>
  <label>species <input id="species" value="shrimp,medaka,crab,snail,beetle,carp,boar" style="width:110px;background:#2c2f2a;color:#ddd;border:1px solid #555"></label><br>
  <button id="reset">重開</button>
 </div>
 <h3>數字</h3>
 <div class="stat" id="stats"></div>
</div>
<script>
const C=document.getElementById('c'),ctx=C.getContext('2d');let S=null,view='normal',tool='none',cell=12;
function dec(b64){const s=atob(b64);const a=new Uint8Array(s.length);for(let i=0;i<s.length;i++)a[i]=s.charCodeAt(i);return a}
const LAYERS=['rel','d','q','speed','tree','farm','water','soil','h','wood','light','mangrove','salt','tidal','elev','sub','kind','ssal','site','temp','snow','algae','o2','crop','rain','gw','wt','pack','infil','exfil','gwl','rainc','wild'];
const WC=[null,[120,220,90],[200,150,80],[235,225,170],[230,120,60],[170,110,50],[255,140,40],[190,170,210]];const WN=['','蕨','香菇','草菇','松茸','栗','柿','河蚌'];
const CROPC=[null,[115,77,43],[140,158,51],[38,102,56]];
const KC=[null,[41,92,51],[31,77,61],[51,140,115],[140,158,82],[158,173,102],[140,158,89],[51,128,102],[89,115,77],[178,184,77],[115,128,56],[77,140,71],[77,102,41],[107,92,31],[115,102,128]];
const KN=['','樹','沼澤林','紅樹林','鹽沼草','濱草','草','海草','岩灘藻','護坡草','胡楊','蕨','栗','柿','河蚌'];
const SUBC=[null,[143,143,135],[214,199,143],[117,105,87]];
const col={land:[158,168,107],tree:[41,92,51],farm:[115,71,43],rich:[77,66,41],erod:[184,133,102],wclean:[76,140,191],wdirty:[115,107,71]};
function mix(a,b,t){return [a[0]+(b[0]-a[0])*t,a[1]+(b[1]-a[1])*t,a[2]+(b[2]-a[2])*t]}
function rgb(c){return `rgb(${c[0]|0},${c[1]|0},${c[2]|0})`}
function viridis(t){t=Math.max(0,Math.min(1,t));const r=[68,59,33,94,253],g=[1,82,145,201,231],b=[84,139,140,98,37];const i=Math.min(3,Math.floor(t*4)),f=t*4-i;return [r[i]+(r[i+1]-r[i])*f,g[i]+(g[i+1]-g[i])*f,b[i]+(b[i+1]-b[i])*f]}
function draw(){if(!S)return;const H=S.H,W=S.W;cell=Math.floor(Math.min(800/W,600/H));if(C.width!==cell*W){C.width=cell*W;C.height=cell*H}
 for(let y=0;y<H;y++)for(let x=0;x<W;x++){const i=y*W+x;let c;
  if(view==='q'){c=S.water[i]?mix([120,40,20],[40,160,220],S.q[i]/255):[40,40,40]}
  else if(view==='speed'){c=S.water[i]?viridis(S.speed[i]/120):[40,40,40]}
  else if(view==='h'){c=S.water[i]?[30,40,60]:viridis(S.h[i]/255)}
  else if(view==='light'){c=viridis(S.light[i]/15)}
  else if(view==='salt'){c=S.water[i]?mix([40,90,180],[230,230,240],S.salt[i]/255):[40,40,40]}
  else if(view==='terrain'){const e=S.elev[i]/60-1;/* relative to low tide */
   if(S.tidal[i]===1)c=[30,60,140];else if(S.tidal[i]===2)c=mix([90,140,190],[200,200,160],Math.max(0,Math.min(1,e/0.9)));else c=mix([120,130,80],[245,240,225],Math.max(0,Math.min(1,(e-0.9)/4)));
   if(S.water[i]&&S.tidal[i]!==1)c=c.map(v=>v*0.75)}
  else if(view==='sub'){c=S.sub[i]?SUBC[S.sub[i]]:[120,130,80];if(S.water[i])c=c.map(v=>v*0.6)}
  else if(view==='flora'){c=S.kind[i]?KC[S.kind[i]]:(S.water[i]?[40,60,90]:[70,70,64])}
  else if(view==='wild'){/* dim map; wood cells brown; ripe wild food in its own colour */
   c=S.water[i]?[30,45,70]:(S.tree[i]?[35,60,40]:[60,62,55]);if(S.wood[i]&&!S.water[i])c=[90,70,45];if(S.kind[i]===11)c=[55,95,55];if(S.wild[i])c=WC[S.wild[i]]}
  else if(view==='ssal'){c=S.water[i]?[40,60,90]:mix([80,110,60],[235,235,245],S.ssal[i]/255)}
  else if(view==='site'){c=S.water[i]?[40,60,90]:viridis(S.site[i]/255)}
  else if(view==='algae'){c=S.water[i]?mix([30,50,90],[120,200,60],S.algae[i]/255):[40,40,40]}
  else if(view==='o2'){c=S.water[i]?mix([150,30,30],[60,170,220],S.o2[i]/255):[40,40,40]}
  else if(view==='rain'){c=viridis(S.rain[i]/200);if(S.water[i])c=c.map(v=>v*0.7)}
  else if(view==='gw'){c=S.water[i]?[30,50,90]:mix([120,110,70],[30,90,200],S.gw[i]/255);if(S.exfil[i]&&!S.water[i])c=[120,220,255]}
  else if(view==='wt'){c=S.water[i]?[30,50,90]:mix([30,90,200],[200,190,150],Math.min(1,S.wt[i]/100))}
  else if(view==='pack'){c=S.water[i]?[30,50,90]:mix([150,160,110],[60,35,15],S.pack[i]/255)}
  else if(view==='infil'){c=S.water[i]?[30,50,90]:mix([200,60,40],[40,160,220],S.infil[i]/255)}
  else if(view==='bits'){/* MC mode: ground-water level 0..3 as four flat colours, rain class as brightness */
   const L=[[150,130,80],[120,150,120],[60,110,200],[120,220,255]];c=S.water[i]?[30,50,90]:L[S.gwl[i]].map(v=>v*(0.7+0.1*S.rainc[i]));if(S.pack[i]>85&&!S.water[i])c=mix(c,[60,35,15],0.6)}
  else if(view==='temp'){const t=S.temp[i]/5-10;c=t<0?mix([230,240,255],[60,90,200],Math.min(1,-t/10)):mix([60,90,200],[220,60,40],Math.min(1,t/30));if(S.water[i])c=c.map(v=>v*0.8)}
  else{const rel=S.rel[i]/255;c=mix(col.land.map(v=>v*0.75),col.land.map(v=>v*1.1),rel);
   if(S.sub[i]===1)c=[143,143,135].map(v=>v*(0.8+0.3*rel));if(S.sub[i]===2)c=[214,199,143];if(S.sub[i]===3)c=[117,105,87];
   if(S.soil[i]===2)c=col.rich;if(S.soil[i]===1&&S.sub[i]!==1)c=col.erod;
   if(S.ssal[i]>128&&!S.water[i])c=c.map(v=>v*0.6+90);
   if(S.pack[i]>100&&!S.water[i])c=mix(c,[110,85,55],Math.min(1,(S.pack[i]-100)/155));
   if(S.snow[i])c=[235,238,245];if(S.sub[i]===4)c=[235,235,230];
   if(S.kind[i]&&S.kind[i]!==3&&S.kind[i]!==7&&S.kind[i]!==8&&S.kind[i]!==4){const k=S.kind[i];c=(k===1||k===2||k===10)?KC[k]:(k===9?KC[9]:c.map((v,j)=>v*0.5+KC[k][j]*0.5))}if(S.tree[i]&&S.kind[i]!==10)c=(S.kind[i]===12||S.kind[i]===13)?KC[S.kind[i]]:col.tree;if(S.farm[i])c=S.soil[i]===2?col.farm.map(v=>v*0.7):col.farm;
   if(S.farm[i]&&S.crop[i])c=CROPC[S.crop[i]];
   if(S.water[i]){const q=S.q[i]/255,d=Math.min(1,S.d[i]/100),a=S.algae[i]/255;c=mix(col.wdirty,col.wclean,q).map(v=>v*(1.05-0.45*d));c=mix(c,[90,150,50],0.5*a);if(S.farm[i]&&S.crop[i]===3)c=CROPC[3]}
   if(S.kind[i]===3)c=KC[3];else if(S.kind[i]===4||S.kind[i]===7||S.kind[i]===8){c=c.map((v,j)=>v*0.4+KC[S.kind[i]][j]*0.6)}}
  ctx.fillStyle=rgb(c);ctx.fillRect(x*cell,y*cell,cell,cell);
  if(S.wood[i]>5&&view==='normal'){ctx.strokeStyle='#2a1a0a';ctx.lineWidth=1.5;ctx.beginPath();ctx.moveTo(x*cell+3,y*cell+3);ctx.lineTo(x*cell+cell-3,y*cell+cell-3);ctx.moveTo(x*cell+cell-3,y*cell+3);ctx.lineTo(x*cell+3,y*cell+cell-3);ctx.stroke()}}
 for(const cr of S.creatures){const [y,x,s,k,g]=cr;const cx=x*cell+cell/2,cy=y*cell+cell/2;ctx.fillStyle=k==='carp'?rgb(mix([90,100,60],[255,120,30],g||0)):rgb(mix([220,50,40],[60,200,70],s));ctx.strokeStyle='#111';ctx.lineWidth=0.8;ctx.beginPath();
  if(k==='shrimp'){ctx.arc(cx,cy,cell*0.28,0,6.283)}else if(k==='crab'){ctx.rect(cx-cell*0.28,cy-cell*0.28,cell*0.56,cell*0.56)}
  else if(k==='snail'){ctx.moveTo(cx-cell*0.3,cy-cell*0.25);ctx.lineTo(cx+cell*0.3,cy-cell*0.25);ctx.lineTo(cx,cy+cell*0.3);ctx.closePath()}
  else if(k==='beetle'){ctx.moveTo(cx,cy-cell*0.33);ctx.lineTo(cx+cell*0.3,cy);ctx.lineTo(cx,cy+cell*0.33);ctx.lineTo(cx-cell*0.3,cy);ctx.closePath()}
  else if(k==='brine'){ctx.arc(cx,cy,cell*0.16,0,6.283)}
  else if(k==='carp'){ctx.ellipse(cx,cy,cell*0.42,cell*0.22,0,0,6.283)}
  else if(k==='boar'){ctx.rect(cx-cell*0.4,cy-cell*0.3,cell*0.8,cell*0.6)}
  else{ctx.moveTo(cx,cy-cell*0.33);ctx.lineTo(cx+cell*0.3,cy+cell*0.25);ctx.lineTo(cx-cell*0.3,cy+cell*0.25);ctx.closePath()}
  ctx.fill();ctx.stroke()}
 const IG=['','鹽','炭','車'];for(const [iy,ix,ik,io] of S.industries){ctx.fillStyle='rgba(30,30,30,0.85)';ctx.fillRect(ix*cell,iy*cell,cell,cell);ctx.fillStyle='#fff';ctx.font=(cell-1)+'px sans-serif';ctx.textBaseline='top';ctx.fillText(IG[ik],ix*cell+0.5,iy*cell)}
 for(const [hy,hx,nm,fd,led] of S.homes){ctx.fillStyle=nm==='player'?'#fff':'#ffd166';ctx.strokeStyle='#000';ctx.fillRect(hx*cell+1,hy*cell+1,cell-2,cell-2);ctx.strokeRect(hx*cell+1,hy*cell+1,cell-2,cell-2);ctx.fillStyle='#fff';ctx.font='10px sans-serif';ctx.fillText(nm+' 食'+fd+' 木'+Math.round(led.wood)+' 鹽'+Math.round(led.salt)+' 炭'+Math.round(led.charcoal)+' 力'+Math.round(led.power),hx*cell+cell+2,hy*cell+cell-1)}
 const st=S.stats;const pops=Object.entries(st.pops).map(([k,v])=>`${k} <b>${v}</b> (產出 ${st.yield[k]})`).join('<br>');
 document.getElementById('stats').innerHTML=`<div>tick</div><b>${st.tick}</b><div>田 / 樹 / 岸樹</div><b>${st.farm} / ${st.trees} / ${st.bank_trees}</b><div>食物 總/tick</div><b>${st.food_total} / ${st.food_tick}</b><div>分解者土壤 / 流失中</div><b>${st.rich} / ${st.eroding}</b><div>岸崩 / 造陸</div><b>${st.collapsed} / ${st.emerged}</b><div>倒木質量</div><b>${st.wood}</b><div>水 格 / 體積 / 水質</div><b>${st.water} / ${st.volume} / ${st.quality}</b><div>搬肥土 / 補樹</div><b>${st.moved} / ${st.replanted}</b><div>紅樹林 / 造陸</div><b>${st.mangrove} / ${st.mg_built}</b><div>植被</div><div style="font-size:11px">${Object.entries(st.flora).map(([k,v])=>k+' '+v).join(' · ')}</div><div>鯽 體色 野/池（n·mean·max·紅）</div><div style="font-size:11px">${['wild','pen'].map(t=>{const g=st.genes[t];return t+' '+g.n+'·'+(g.mean??'-')+'·'+(g.max??'-')+'·'+g.red}).join(' | ')} · 鳥吃 ${st.birds}</div><div>野味 現在 / 累計潛力</div><div style="font-size:11px">${Object.entries(st.wild).map(([k,v])=>k+' '+v).join(' · ')}<br>${Object.entries(st.wild_total).map(([k,v])=>k+' '+v).join(' · ')}</div><div>鹽化土 / 岩</div><b>${st.saline} / ${st.rock}</b><div>地下水 總 / 湧泉格 / 路</div><b>${st.gw_total} / ${st.springs} / ${st.paths}</b><div>暴雨 次 / 現在</div><b>${st.storms} / ${st.storm_on?'⛈':'—'}</b><div>模式</div><b>${st.mode}${st.mode==='mc'?' · 事件 '+st.mc_events:''}</b><div>季節 / 均溫 / 雪量 / 春汛</div><b>${st.season} / ${st.temp_mean}° / ${st.snowpack} / ×${st.spring_mult}</b><div>潮位（海平面 ${st.sea_level}）</div><b>${st.tide}</b><div>生成</div><div style="font-size:11px">attempt ${st.gen.attempt} ${st.gen.ok?'✓':'✗'} 河${st.gen.river_connected?'通':'斷'} 池${st.gen.pond_wet??'-'} 支流${st.gen.tributary_wet_frac??'-'} 潮間帶${st.gen.coast_intertidal??'-'}</div><div>生物</div><div>${pops}</div>`;
 document.getElementById('play').textContent=S.running?'❚❚ 停':'▶ 跑';document.getElementById('play').className=S.running?'on':''}
async function poll(){try{const r=await fetch('/state');const j=await r.json();if(!j.empty){for(const k of LAYERS)j[k]=dec(j[k]);S=j;draw()}}catch(e){console.error(e)}setTimeout(poll,S&&S.running?100:400)}
async function ctl(o){await fetch('/control',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(o)})}
document.getElementById('play').onclick=()=>ctl({running:!(S&&S.running)});
document.getElementById('step').onclick=()=>ctl({step:1,n:1});
document.getElementById('step100').onclick=()=>ctl({step:1,n:100});
document.getElementById('tps').onchange=e=>ctl({tps:+e.target.value});
document.getElementById('steps_btn').onclick=async()=>{const r=await fetch('/steps',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({path:document.getElementById('steps_path').value})});const j=await r.json();document.getElementById('steps_out').textContent=j.ok?`${j.players} 人 ${j.samples} 筆 → ${j.cells} 格`:('✗ '+j.error)};
document.getElementById('reset').onclick=()=>{const cfg={};for(const id of ['seed','buffer','tide_amp','farmer_mode','expand_every','feed_wood_every','feed_mode','move_soil_every','n_settlements','site_spacing','industries','basin','evap_mult','species','seg_highland','seg_canyon','seg_hills','seg_floodplain','seg_delta','sea','pond','swamp','tributary','wind','rain_ns','orog_k','storm_prob','storm_mult','mode']){const el=document.getElementById(id);cfg[id]=el.type==='checkbox'?(el.checked?1:0):(el.type==='number'?+el.value:el.value)}const rx=+document.getElementById('rain_x').value;if(rx!==1)cfg.rain=0.0006*rx;ctl({reset:1,cfg})};
for(const b of document.querySelectorAll('#views button'))b.onclick=()=>{view=b.dataset.v;for(const o of document.querySelectorAll('#views button'))o.className='';b.className='on';draw()};
for(const b of document.querySelectorAll('#tools button'))b.onclick=()=>{tool=b.dataset.t;for(const o of document.querySelectorAll('#tools button'))o.className='';b.className='on'};
C.onmousedown=async e=>{if(tool==='none')return;const r=C.getBoundingClientRect();const x=Math.floor((e.clientX-r.left)/cell),y=Math.floor((e.clientY-r.top)/cell);await fetch('/act',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({tool,y,x})})};
document.addEventListener('keydown',e=>{if(e.code==='Space'){e.preventDefault();ctl({running:!(S&&S.running)})}});
poll();
</script></body></html>"""


@app.route("/")
def index():
    return Response(PAGE, mimetype="text/html")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=7748)
    p.add_argument("--set", action="append", default=[])
    a = p.parse_args()
    cfg = {}
    for kv in a.set:
        k, v = kv.split("=", 1)
        cfg[k] = v
    STATE["cfg"] = cfg
    STATE["sim"] = make_sim(cfg)
    STATE["running"] = True
    threading.Thread(target=runner, daemon=True).start()
    app.run(host="127.0.0.1", port=a.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()

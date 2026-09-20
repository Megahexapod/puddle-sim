# Puddle Sim

> Watch water, soil, and life emerge on a tiny river landscape from simple local rules.
<!-- tagline 草稿——換成你的話。目標：陌生人一行定位 + 一點鉤子。 -->

<!-- 放一張截圖：從 out/ 挑最能代表 puddle 的（live_7748.png / steps_on_map.png…），複製到 docs/screenshot.png。哪張最能代表＝你的品味，我不替你選。 -->
![Puddle Sim screenshot](docs/screenshot.png)

## What is this

Puddle Sim grows a tiny landscape — a river cutting down a slope into a tidal
estuary — and then lets water, soil, and life play out on it under simple local
rules. Every cell only talks to its four neighbours, yet out of that you get
rivers that pool and overflow, salt gradients at the river mouth, mangroves
colonising the mudflats, and crabs that eat the mangrove seedlings and keep them
in check.

The interesting part is watching a small rule-change ripple through the whole
system. A scripted "farmer" clears trees to plant fields; do that too close to
the water and the shrimp — which quietly need shade, clean water, and leaf
litter all at once — collapse. The shrimp are the canary: one number that tells
you whether the ecosystem is still alive.

It's a sandbox for that kind of question. Change a number, run ten thousand
ticks, look at the picture.

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
# Interactive web UI — opens http://localhost:7748
python3 play.py

# Headless simulation, writes snapshots and a summary to a folder
python3 run.py --ticks 4000 --buffer 0 --out out/b0

# Sweep a parameter across seeds
python3 run.py sweep --buffers 0 1 2 3 --seeds 1 2 3 --out out/sweep
```

Any config field can be overridden: `python3 run.py --set n_shrimp=40 --set runoff_per_farm=0.02`

## How it works

<!-- How 草稿——這是給工程師的房間，術語可以進來了。微調成你的話，或換成你覺得更漂亮的機制。 -->

- **Water as virtual pipes** — each cell exchanges flow only with its four neighbours, so basins fill and overflow on their own, O(cells) per tick.
- **A tidal estuary** — salt advects downstream; mangroves colonise the mudflats and crabs graze their seedlings: a built-in predator feedback loop.
- **Soil as one number per cell** — eroding / healthy / decomposer-rich, with bank collapse, sedimentation, and rotting deadwood feeding back into it.
- **Habitat by Liebig's law of the minimum** — a creature scores shade, water quality, and food *together*; any one hitting zero zeroes the whole score.

## License

MIT — see [LICENSE](LICENSE).

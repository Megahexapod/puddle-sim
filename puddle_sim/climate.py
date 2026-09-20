"""Climate: temperature as a field, rain and evaporation as a season, snow as a store (D-026).

Temperature = latitude gradient (north cold, south warm)
            + sea moderation (near the sea the swing is small; inland it is continental)
            + surface (canopy cools, bare sand/rock warms, snow cools)
            − altitude lapse
            + season (a yearly sine, amplified inland).
Rain and evaporation follow the season and the temperature; where it is below freezing the rain
is snow and waits in a pack until the thaw sends it down the river as the spring flood.
Everything that lives reads `world.temp`; the plant and creature tables carry a temperature band.
"""
from __future__ import annotations

import numpy as np

from .config import Config


class Climate:
    def __init__(self, cfg: Config, world):
        self.cfg = cfg
        H, W = cfg.height, cfg.width
        from .world import chebyshev_distance_to
        self.lat = np.linspace(cfg.t_north, cfg.t_south, H, dtype=np.float32)[:, None] * np.ones((1, W), np.float32)
        if cfg.sea:
            sea = world.hydro.b < cfg.sea_level                      # D-038: the sea is wherever the ground is under it
            sea[:, :world.coast_x0] = False
            if not sea.any():
                sea[:, W - cfg.berm_col:] = True
            dist = chebyshev_distance_to(sea, cfg.maritime_reach).astype(np.float32)
            self.maritime = np.exp(-dist / (cfg.maritime_reach / 2.5)).astype(np.float32)
        else:
            self.maritime = np.zeros((H, W), dtype=np.float32)
        self.temp = self.lat.copy()
        self.season = 0.0             # 0..1 through the year; 0.25 = midsummer
        self.rain_now = cfg.rain
        self.evap_now = cfg.evap
        self.snowpack = 0.0
        self.spring_mult = 1.0
        self.snow = np.zeros((H, W), dtype=bool)
        self.pattern = np.ones((H, W), dtype=np.float32)   # spatial rain factor, mean 1 over land (D-029)
        self.moisture = np.ones((H, W), dtype=np.float32)  # air moisture left after the march (for the view)
        self.rain_field = np.full((H, W), cfg.rain, dtype=np.float32)
        self.evap_field = np.full((H, W), cfg.evap, dtype=np.float32)
        self.storm = None             # (y, x, ticks_left) while a storm is on the map
        self.storms_total = 0
        self.update(world, 0)

    # ------------------------------------------------------------------ rain pattern (D-029)

    def _pattern(self, world) -> None:
        """Where the rain falls, given the same sky: latitude, distance to the sea, and an air mass
        marched across the relief from the windward edge — it drops moisture climbing slopes
        (orographic), keeps what is left for the lee (rain shadow), and picks some back up over
        water (lake effect) and forest (transpiration). Normalised to mean 1 over land."""
        cfg = self.cfg
        H, W = self.lat.shape
        b = world.hydro.b
        water = world.water
        canopy = world.canopy()
        # Latitude and sea.
        lat = np.linspace(0.0, 1.0, H, dtype=np.float32)[:, None] * np.ones((1, W), np.float32)
        f = 1.0 + cfg.rain_ns * (lat - 0.5) * 2.0
        if cfg.sea:
            f = f * (1.0 - cfg.rain_continental * (1.0 - self.maritime))
        f = np.clip(f, 0.05, None)
        # March with the wind. Rotate so the air enters at column 0 and walks +x.
        k = {"E": 2, "W": 0, "N": 1, "S": 3}.get(cfg.wind.upper(), 2)
        # np.rot90(a, k) puts the windward edge at column 0 (checked: k=2 -> east column first, walking west).
        bb = np.rot90(b, k); ww = np.rot90(water, k); cc = np.rot90(canopy, k)
        Hh, Ww = bb.shape
        M = np.ones(Hh, dtype=np.float32)
        r = np.zeros((Hh, Ww), dtype=np.float32)
        mm = np.zeros((Hh, Ww), dtype=np.float32)
        prev = bb[:, 0]
        for x in range(Ww):
            lift = np.clip(bb[:, x] - prev, 0.0, None)
            drop = np.minimum(M, M * (cfg.rain_bg_frac + cfg.orog_k * lift))
            r[:, x] = drop
            M = M - drop
            M = M + cfg.moist_recharge * (ww[:, x] + 0.5 * cc[:, x]) * (1.0 - M)
            mm[:, x] = M
            prev = bb[:, x]
        r = np.rot90(r, -k); mm = np.rot90(mm, -k)
        # A little lateral smoothing: clouds are wider than a cell.
        from .world import count_window
        r = count_window(r, 1) / 9.0
        land = ~water
        r = r / max(1e-6, float(r[land].mean()))
        f = f * r
        f = f / max(1e-6, float(f[land].mean()))
        self.pattern = f.astype(np.float32)
        self.moisture = mm.astype(np.float32)
        if hasattr(world.ground, "set_rain_class"):        # MC mode reads the pattern as 2-bit classes
            world.ground.set_rain_class(self.pattern)

    def _storm_field(self, world, rain_base: float, tick: int) -> np.ndarray:
        """An episodic cloudburst: a moving blob of heavy rain. Its intensity is the point —
        light rain soaks in, a burst runs off (ground.py decides)."""
        cfg = self.cfg
        H, W = self.lat.shape
        if self.storm is None:
            if cfg.storm_prob > 0 and world.rng.random() < cfg.storm_prob * 10:   # update runs every 10 ticks
                y, x = int(world.rng.integers(0, H)), int(world.rng.integers(0, W))
                self.storm = [float(y), float(x), cfg.storm_len]
                self.storms_total += 1
            else:
                return np.zeros((H, W), dtype=np.float32)
        y, x, left = self.storm
        ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
        blob = np.exp(-((ys - y) ** 2 + (xs - x) ** 2) / (2.0 * cfg.storm_r ** 2))
        vy, vx = {"E": (0, -1), "W": (0, 1), "N": (1, 0), "S": (-1, 0)}.get(cfg.wind.upper(), (0, -1))
        self.storm = [y + 2.0 * vy, x + 2.0 * vx, left - 10]
        if self.storm[2] <= 0:
            self.storm = None
        return (rain_base * cfg.storm_mult * blob).astype(np.float32)

    def update(self, world, tick: int) -> None:
        cfg = self.cfg
        year = cfg.year_ticks
        self.season = (tick % year) / year
        phase = np.sin(2 * np.pi * (self.season - 0.25 + 0.25))   # +1 at season 0.25 (summer), -1 at 0.75
        swing = cfg.season_amp * (1.0 + cfg.continental * (1.0 - self.maritime))
        b = world.hydro.b
        lapse = cfg.lapse * np.clip(b - cfg.sea_level, 0.0, None)
        surf = np.zeros_like(b)
        canopy = world.canopy()
        surf -= cfg.surf_canopy * canopy
        bare = (~world.water) & (world.flora.kind == 0)
        surf += cfg.surf_bare * bare
        surf += cfg.surf_bare * 0.5 * (world.substrate == 2)          # sand
        T = self.lat + swing * phase - lapse + surf
        T = T * (1.0 - 0.35 * self.maritime) + cfg.t_sea * 0.35 * self.maritime   # the sea pulls toward its own temperature
        self.snow = (T < 0.0) & (~world.water)
        T = T - cfg.surf_snow * self.snow
        self.temp = T.astype(np.float32)

        # Rain: one sky (the season) spread by the pattern (D-029), plus storms.
        rain_phase = np.sin(2 * np.pi * (self.season - cfg.rain_peak))
        self.rain_now = cfg.rain * (1.0 + cfg.rain_season * rain_phase)
        if tick % cfg.rain_pattern_every == 0:
            self._pattern(world)
        field = self.rain_now * self.pattern
        if cfg.storm_prob > 0 or self.storm is not None:
            field = field + self._storm_field(world, cfg.rain, tick)
        # Rain that falls on frozen ground is snow: it stays put (no surface water, no infiltration).
        self.rain_field = np.where(self.snow, 0.0, field).astype(np.float32)
        # Evaporation: by the local temperature (sand in the sun dries, the forest floor keeps), or the mean.
        tmean = float(T.mean())
        self.evap_now = cfg.evap * float(np.clip(0.4 + tmean / 20.0, 0.2, 2.5))
        if cfg.evap_local:
            self.evap_field = (cfg.evap * np.clip(0.4 + T / 20.0, 0.2, 2.5)).astype(np.float32)
        else:
            self.evap_field = np.full(T.shape, self.evap_now, dtype=np.float32)
        # Snow: where the headwaters are frozen the rain waits; the thaw comes down the river.
        head = float(T[:, : max(2, cfg.width // 8)].mean())
        if head < 0.0:
            self.snowpack += self.rain_now * cfg.snow_catch
            melt = 0.0
        else:
            melt = min(self.snowpack, cfg.melt_rate * head)
            self.snowpack -= melt
        self.spring_mult = 1.0 + melt * cfg.melt_to_flow
        hy = world.hydro
        hy.evap_now = self.evap_field
        hy.spring_mult = float(self.spring_mult)
        # hy.rain_now is set every tick by world.step after the ground has taken its share.

    def temp_factor(self, lo: float, hi: float, soft: float = 4.0) -> np.ndarray:
        """0..1 — how far inside the band (lo..hi) the current temperature is, with soft edges."""
        T = self.temp
        return np.clip(np.minimum((T - lo) / soft, (hi - T) / soft), 0.0, 1.0).astype(np.float32)

"""Life history from body mass (life_economy.md, D-032).

Two numbers per species — body mass M (shrimp = 1) and a strategy nudge s in [-1, 1] (r .. K) —
and the allometry the field guides agree on does the rest (Kleiber 1932; Peters 1983; Calder 1984;
Damuth 1981):

    food need      ~ M^0.75
    lifespan       ~ M^0.25 (1 + s/2)
    generation     ~ M^0.25 (1 + s/2)
    litter         ~ M^-0.25 (1 - s/2)
    growth rate k  ~ M^-0.25            (von Bertalanffy: size = M (1 - e^{-k t})^3)
    density cap    ~ M^-0.75            (per 5x5)

The one door to the environment is the habitat score (Liebig): it scales growth, litter and
lifespan. Nothing here is tuned per species; change M or s, or the base constants in config.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import Config


@dataclass
class LifeTable:
    mass: float          # M, adult body mass (relative)
    strategy: float      # s, -1 (r) .. +1 (K)
    egg: bool            # egg-layers: tenth the birth mass, ten times the count
    need: float          # food per tick at full size
    lifespan: int        # ticks, at full satisfaction
    generation: int      # ticks between broods
    litter: float        # expected offspring per brood (stochastic rounding)
    k: float             # von Bertalanffy rate per tick
    density_cap: int     # max of this kind in a 5x5
    birth_frac: float    # size at birth as a fraction of M
    mature_frac: float = 0.6

    @staticmethod
    def derive(cfg: Config, mass: float, strategy: float, egg: bool) -> "LifeTable":
        kk = 1.0 + 0.5 * strategy
        m25 = mass ** 0.25
        litter = cfg.life_litter0 * (mass ** -0.25) * (1.0 - 0.5 * strategy)
        birth = 0.1
        if egg:
            litter *= 10.0
            birth = 0.01
        return LifeTable(
            mass=mass, strategy=strategy, egg=egg,
            need=mass ** 0.75,
            lifespan=int(cfg.life_L0 * m25 * kk),
            generation=int(cfg.life_G0 * m25 * kk),
            litter=litter,
            k=cfg.life_k0 / m25,
            density_cap=max(1, int(round(cfg.life_D0 * mass ** -0.75))),
            birth_frac=birth,
        )

    def size_at(self, t_eff: float) -> float:
        """von Bertalanffy on effective (satisfaction-weighted) age."""
        return float(self.mass * (1.0 - np.exp(-self.k * t_eff)) ** 3)

    def is_mature(self, size: float) -> bool:
        return size >= self.mature_frac * self.mass

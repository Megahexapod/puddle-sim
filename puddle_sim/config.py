"""All tunable numbers in one place. Change a number, rerun, look at the plot."""
from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass
class Config:
    @property
    def ticks_per_day(self) -> float:
        return self.year_ticks / 365.0

    def __post_init__(self):
        # The calendar clock, converted once (D-049). Anything the household feels is written in days.
        tpd = self.ticks_per_day
        if self.eat_per_person <= 0:
            self.eat_per_person = self.eat_per_day / tpd
        if self.step_cost <= 0:
            self.step_cost = self.job_hours_per_cell / (self.day_hours / tpd)   # ticks per cell = hours per cell / hours per tick
    # --- run ---
    seed: int = 1
    width: int = 96
    height: int = 64
    ticks: int = 8000
    log_every: int = 10
    snap_every: int = 1000

    # --- the platter (D-018): reaches along the river, proportions of river length; 0 = off ---
    seg_highland: float = 0.10
    seg_canyon: float = 0.12
    seg_hills: float = 0.28
    seg_floodplain: float = 0.30
    seg_delta: float = 0.20
    sea: int = 1                        # 0 = inland map: no tide, no salt, no mangroves
    pond: int = 1
    swamp: int = 1
    tributary: int = 1
    tributary_flow: float = 0.4         # tributary spring as a fraction of the main spring
    gen_attempts: int = 6               # re-roll the skeleton this many times if invariants fail

    # --- substrate (底質): read off the skeleton, not painted. 0 soil / 1 rock / 2 sand / 3 mud ---
    rock_slope_highland: float = 0.16   # steeper than this (height per cell) => rock, per reach
    rock_slope_canyon: float = 0.16
    rock_slope_hills: float = 0.45
    rock_slope_floodplain: float = 0.80
    rock_patch: float = 0.06            # noise on the threshold so outcrops are patchy, not contour lines
    sand_h_cap: float = 0.35            # sand is poor: h cannot exceed this
    sand_root: float = 0.6              # roots hold less in sand
    sand_scour: float = 1.5             # sand washes out faster
    mud_scour: float = 0.7              # mud is cohesive
    mud_moist: float = 0.2              # mud is wetter at baseline

    # --- terrain (heights in "blocks") ---
    slope: float = 0.035                # height drop per cell, west (high) -> east (low)
    valley_depth: float = 1.5           # how deep the river valley is carved
    valley_width: float = 2.2           # gaussian half-width of the valley (per-reach in gen.py)
    channel_depth: float = 0.5          # a narrow channel inside every valley
    min_drop: float = 0.008             # the bed must fall at least this per cell going east
    meander_amp: float = 6.0            # how far the valley wanders north/south
    noise_amp: float = 0.5              # hills on the plain (two octaves of smooth noise)
    pond_depth: float = 1.6             # a basin off the river
    pond_sigma: float = 3.0             # basin radius
    pond_q: float = 0.2                 # a small spring in the basin (rain alone barely fills it)
    tree_density: float = 0.28          # base chance a land tile starts with a tree
    riparian_bonus: float = 0.40        # extra tree chance within 2 tiles of water (bank forest)

    # --- water physics (hydro.py) ---
    hydro_substeps: int = 4             # physics substeps per sim tick
    hydro_dt: float = 0.04              # substep length
    gravity: float = 9.8
    hydro_friction: float = 0.02        # fraction of pipe flux lost per substep (bed friction, crude)
    spring_q: float = 1.5               # volume per time unit entering at the valley head
    rain: float = 0.0006                # depth per time unit, everywhere
    evap: float = 0.0012                # depth per time unit, everywhere (keeps plateaus dry)
    # East edge = estuary. The sea sits at sea_level and breathes with the tide; a flat delta plain
    # in the last `estuary_cols` columns floods at high tide and drains at low tide (intertidal).
    sea_level: float = -1.30            # outlet valley bed is at -1.5 (slope*0 - valley_depth)
    tide_amp: float = 0.45
    tide_period: int = 600              # ticks per tidal cycle
    delta_height: float = 0.05          # plain sits about here relative to mean sea level ...
    coast_undulation: float = 0.35      # ... plus low hills and hollows: some dry, some marsh, some lagoon
    berm_col: int = 4                   # (retired D-038: the ridge is grown now; kept for the climate's sea mask fallback)
    shelf_cols: int = 12                # D-038: the coastal plain bends down to open sea over this many columns (quadratic)
    shelf_drop: float = 1.4             # ... by this much at the edge; the coastline is where the surface crosses sea level
    beach_rate: float = 0.0006          # swash deposition per tick on a gentle sandy shore touching salt water (x tide_amp); the ridge's first force
    beach_top: float = 0.35             # swash cannot pile sand higher than high tide + this; dune grass takes it from there
    storm_wave: float = 3.0             # wave scour x this during a storm (and the swash stops building): the winter profile
    storm_reach: float = 1.0            # a storm wave reaches this far above high tide (fair weather: beach_top)
    closure_depth: float = 0.35         # below low tide minus this, waves no longer move sand: the beach stops growing when the inshore reaches it (equilibrium profile)
    headland: int = 1                   # a rocky promontory on the coast (岩灘 lives there)
    ema_ticks: int = 300                # smoothing window for depth/speed the plants read (rides over tides)
    salt_diffusion: float = 0.08        # salinity mixing between adjacent water cells per tick
    d_min: float = 0.04                 # depth that counts as water (for ecology, velocity, drawing)
    tree_drown: float = 0.6             # trees die under this much water
    farm_flood: float = 0.3             # fields are lost under this much water
    still_speed: float = 0.5            # speed at which "stillness" reaches 0 (river runs ~1.4)
    speed_cap: float = 5.0              # clip for display/ecology (outlet cells spike)
    spinup_ticks: int = 2500            # hydro-only ticks before anything lives (water must cross the map)
    dam_tick: int = -1                  # >=0: at this tick raise a wall across the valley (course-change test)
    dam_x: int = 24                     # column of the wall
    dam_height: float = 1.5

    # --- light (same idea as the MC experiment: trees shade their 3x3) ---
    max_light: int = 15
    tree_shade: int = 3                 # light lost per tree in the 3x3 around a tile
    crop_min_light: int = 10            # crops need this; the farmer clears trees to get it

    # --- water quality (0..1, water tiles only) ---
    quality_recover: float = 0.004      # per tick, toward 1.0
    riparian_recover: float = 0.002     # extra recovery per tree in the 3x3 (roots, shade)
    runoff_per_farm: float = 0.008      # loss per tick per farmland tile in the 3x3
    quality_diffusion: float = 0.15     # mixing between adjacent water tiles

    # --- detritus = shrimp food (leaf litter falling from bank trees) ---
    litter_per_tree: float = 0.015      # per tick per tree in the 3x3, on water tiles
    detritus_decay: float = 0.008       # per tick
    detritus_eat: float = 0.02          # per shrimp per tick
    food_sat: float = 1.0               # detritus level that counts as "full"

    # --- settlements (D-022): player + NPC farmers at the best-scored sites ---
    n_settlements: int = 1              # 1 = player only; more = NPC farmers on the same river
    site_spacing: int = 18              # minimum distance between settlements (cells)

    # --- industries (D-024/D-025) ---
    industries: int = 1                 # farmers build salt pans / kilns / water wheels
    industry_min_fit: float = 0.35      # will not build on a worse site
    industry_radius: int = 14           # how far from home a settlement builds
    build_every: int = 40               # ticks between build attempts per settlement (~5 days)
    harvest_every: int = 8              # ticks between felling one tree into the ledger (~1 day)
    saltpan_salinise: float = 0.004     # soil salt added per tick to the pan's neighbours at fitness 1

    # --- farmer (scripted "normal agriculture") ---
    expand_every: int = 5               # ticks between new farmland tiles (~0.6 day: a plot a day, ~1 ha a year by hand)
    max_farms: int = 0                  # farmer stops opening fields at this many (0 = no cap; a player strategy)
    rice_cap: int = 0                   # mixed farm: this many paddies at the river, then dry fields behind (0 = off)
    dry_buffer: int = -1                # riparian buffer for the dry fields of a mixed farm (-1 = same as `buffer`)
    rotate: int = 0                     # 稲麦二毛作: paddies are rice inside `rice_season`, drained and sown dry outside it
    rice_season: tuple = (0.08, 0.55)   # season fraction (0.25 = midsummer) during which a rotated paddy holds rice
    water_reach: int = 4               # farmland must be within this (Chebyshev) of water — MC rule
    buffer: int = 0                     # tiles next to water the farmer leaves alone (riparian buffer)
    clear_radius: int = 1               # trees cleared around a new field so crops get light >= 10
    tend_radius: int = 3                # a tending trip weeds every field within this of the one visited (D-046)
    farmer_mode: str = "blind"          # "blind": expand no matter what. "adaptive": watches shrimp yield
    cover_plant: str = "vetiver"        # what the adaptive farmer plants on a bank: "vetiver" or "tree"
    water_fields: int = 1               # farmers may make caltrop ponds on still fresh water
    crop_pref: str = "any"              # "rice": open paddy-eligible cells first (a player strategy, levels.py)
    watch_species: str = "shrimp"       # adaptive farmer watches this species' yield
    drop_tol: float = 0.25              # adaptive: yield below (1-drop_tol)*peak => stop, replant a bank tree
    ema: float = 0.02                   # adaptive: smoothing of the watched yield

    # --- soil (soil.py): one number h, three states ---
    h_min: float = 0.20                 # below: 流失中
    h_mid: float = 0.50                 # litter repairs toward this, never above
    h_rich: float = 0.70                # above: 分解者土壤
    h_silt: float = 0.15                # fresh silt / collapsed cells start here
    m_base: float = 0.08                # baseline moisture on land (dry upland)
    m_reach: int = 10                   # cells from water over which ground water fades
    m_reach_gain: float = 0.30          # moisture added right next to water by ground water
    m_crit: float = 0.60                # wetter than m_crit*(1+roots) => eroding
    root_tree: float = 0.60
    root_crop: float = 0.10
    root_grass: float = 0.25
    root_bare: float = 0.00
    litter_soil: float = 0.0006         # per tree in the 3x3 per tick, toward h_mid
    crop_drain: float = 0.00015         # per farm cell per tick
    h_diffusion: float = 0.002          # h mixing between land cells per tick
    wood_rot: float = 0.002             # wood mass lost per tick in shade+damp (tree = 1.0 => ~500 ticks)
    wood_sun_rot: float = 2.0           # sun multiplies rot by (1 + this*sun)  — 茨 D-013
    wood_sun_waste: float = 0.7         # sun cuts the soil yield to (1 - this*sun)
    wood_soaked: float = 0.95           # moisture above: no rot (submerged piles last forever)
    wood_dry: float = 0.05              # moisture below: no rot (desert)
    wood_to_soil: float = 0.6           # h delivered per unit of wood rotted
    wood_under: float = 0.6             # share that lands on the log's own cell (the rest spreads to the ring)
    h_cap: float = 1.5                  # rich soil can stockpile above 1.0 (surplus the farmer can carry)
    fern_eat: float = 0.01              # fraction of surplus (h - h_rich) eaten per tick when no wood near
    tree_wood: float = 1.0              # deadwood mass a dead tree leaves
    tree_die_p: float = 0.00004         # natural death per tree per tick
    tree_regrow_p: float = 0.00004      # per land cell per tick with >=2 tree neighbours and h >= h_min (~balances death)
    scour_rate: float = 0.02            # scour += rate * neighbour speed * (1 - roots) on eroding cells
    scour_v0: float = 0.15              # water slower than this does not scour (Hjulström: a still shore heals, D-029)
    scour_heal: float = 0.01            # scour relaxes when the cell is healthy again
    scour_crit: float = 1.0             # collapse threshold
    # --- bedrock (D-029): under every cell a floor that collapse cannot pass; reaching it exposes rock ---
    depth_soil: float = 1.0             # soil column thickness in height units (x4 = blocks) by substrate
    depth_mud: float = 0.6
    depth_sand: float = 0.5
    depth_rock: float = 0.0
    soil_to_sediment: float = 1.0       # terrain volume lost -> sediment mass
    v_dep: float = 0.4                  # water slower than this drops sediment
    sediment_bg: float = 0.12           # sediment concentration the spring brings in (a muddy river; erosion beyond the map)
    dep_rate: float = 0.05              # fraction of suspended sediment deposited per tick at speed 0
    farm_food: float = 0.01             # food per farm cell per tick at h_mid (x2 on rich soil, 0 when eroding)
    feed_wood_every: int = 0            # farmer: every N ticks cut one tree and lay the wood (0 = never)
    feed_mode: str = "shade"            # "shade": next to a tree near the fields; "field": on the poorest field
    move_soil_every: int = 0            # farmer: every N ticks carry surplus h from a rich cell to the poorest field (0 = never)
    move_soil_amount: float = 0.3       # h moved per trip (the rich_soil block of the MC experiment)

    # --- climate (D-026): temperature field, seasons, snow ---
    t_north: float = 10.0               # mean temperature at the top edge
    t_south: float = 24.0               # ... at the bottom edge
    t_sea: float = 17.0                 # the sea's own temperature (pulls the coast toward it)
    maritime_reach: int = 24            # cells over which the sea's moderation fades
    lapse: float = 2.4                  # degrees lost per height unit above sea level (highland freezes in winter)
    surf_canopy: float = 1.5            # canopy cools
    surf_bare: float = 1.5              # bare ground warms (sand half again)
    surf_snow: float = 3.0              # snow cools (albedo)
    season_amp: float = 6.0             # +/- yearly swing at the coast
    continental: float = 0.8            # inland swing = amp * (1 + this)
    # --- the calendar clock (D-049). One year, and everything the household feels, is written in DAYS below;
    # __post_init__ turns days into ticks once. The biological clock (trees, animals, soil, flora) keeps its
    # own tick rates -- compressed on purpose (a tree in ~7 weeks, a boar generation in ~a season). ---
    year_ticks: int = 3000              # was 12000: at 3000 the existing biology lands near real per-year ratios
    day_hours: float = 24.0
    rain_season: float = 0.6            # rain = base * (1 +/- this) through the year
    rain_peak: float = 0.25             # season (0..1) when rain peaks; 0.25 = summer monsoon
    snow_catch: float = 400.0           # snowpack units per rain unit while the headwaters freeze
    melt_rate: float = 0.02             # snowpack melted per tick per degree above 0 at the headwaters
    melt_to_flow: float = 3.0           # spring flow multiplier per unit melt
    crop_t_lo: float = 8.0              # crops yield nothing below this, fully above lo + 6

    # --- rain as a field (D-029): the same total, spread by latitude, sea, wind and relief ---
    rain_ns: float = 0.0                # -1..1: +1 = the south edge gets 2x the north edge (monsoon side)
    rain_continental: float = 0.5       # sea maps: the far interior gets (1 - this) of the coast's rain
    wind: str = "E"                     # where the moist air comes FROM: E (off the sea), W, N, S
    orog_k: float = 0.2                 # moisture dropped per height unit of uplift (a 4-unit range takes ~60%; a berm, little)
    rain_bg_frac: float = 0.008         # moisture dropped per cell on flat ground (~half the air mass across the map)
    moist_recharge: float = 0.03        # moisture picked back up per water cell crossed (lake effect); canopy half
    rain_pattern_every: int = 200       # ticks between recomputing the pattern (terrain moves slowly)
    storm_prob: float = 0.0012          # per tick chance a storm starts (0 = off); ~3-4 a year at 3000 ticks/yr. Was 0 until D-039
    storm_surge: float = 0.9            # D-039: the sea head rises this much above the tide while a storm is on (overwash / back-barrier lagoons)
    crust_mass: float = 0.12            # salinity x depth a drying cell must have held to leave a salt crust (a thin surge film does not)
    ocean_mix: float = 0.01             # per tick, deep shelf water relaxes toward sea salinity (the ocean beyond the map edge)
    storm_len: int = 6                  # ticks (~a day)
    storm_mult: float = 15.0            # rain at the storm's centre, x base
    storm_r: float = 12.0               # cells
    evap_local: int = 1                 # evaporation reads the local temperature (bare sand dries, forest shade keeps)

    # --- infiltration + ground water (D-028 ①, D-029): rain enters the ground, the ground leaks ---
    infil_soil: float = 0.8             # share of rain an unsaturated, untrodden cell can take, by substrate
    infil_sand: float = 1.0
    infil_mud: float = 0.4
    infil_rock: float = 0.05
    por_soil: float = 0.35              # porosity: water a saturated column holds per unit of thickness
    por_sand: float = 0.40
    por_mud: float = 0.30
    por_rock: float = 0.05
    gw_flow: float = 0.04               # Darcy-ish: share of the water-table head difference moved per tick
    gw_et: float = 0.00002              # evapotranspiration from the store per tick at 20°C (canopy x1.5); rain is ~1e-4/tick
    gw_recharge: float = 0.002          # per tick a wet cell's column takes from the surface (x room left)
    m_gw: float = 0.25                  # soil moisture added by a full store

    # --- MC mode (D-030 / 1852k): land layers as bits + events + random ticks, side by side with the continuous mode ---
    mode: str = "continuous"            # "continuous" | "mc"
    mc_random_ticks: int = 3            # random-ticked cells per 256-cell section per tick (MC: 3 per 16^3 section)
    mc_gw_every: int = 500              # ticks between ground-water lookups when nothing moved (the season)
    mc_table_drop: float = 0.05         # local water table falls this much per cell away from open water (pit rule)
    pack_per_level: int = 4             # crossings per tread level (0..3)
    mc_leach_p: float = 0.2             # chance a random-ticked saline cell in a wet region is leached clean

    # --- tread (D-028): compaction is a field; paths are where it stays high ---
    pack_step: float = 0.04             # per crossing
    pack_relax: float = 0.0004          # per tick, x (1 + roots)
    pack_grass: float = 0.4             # ground cover cannot establish above this
    pack_h_loss: float = 0.0002         # h lost per tick on hard bare ground

    # --- water chemistry (D-027): algae, oxygen, decay ---
    algae_grow: float = 0.03            # per tick at full light/warmth/nutrients
    algae_die: float = 0.012            # per tick (dead algae -> detritus)
    algae_bg: float = 0.01              # floating background so blooms can start anywhere
    algae_uptake: float = 0.02          # pollution removed per unit algae growth
    algae_eat: float = 0.03             # per grazing animal per tick
    o2_reaer: float = 0.03              # reaeration toward 1 per tick, times (0.3 + speed)
    o2_photo: float = 0.03              # oxygen per unit algae per tick in light
    o2_bod: float = 0.02                # oxygen consumed per unit (detritus + dead algae) per tick, times warmth
    o2_resp: float = 0.004              # per animal per tick
    beetle_pressure: float = 0.2        # (retired D-041: pressure now lives on the Species row; kept for old configs)

    # --- crops (D-027): a table like the plants'; the farmer picks per field ---
    rice_runoff: float = 1.5            # paddies leak more nutrients than dry fields
    snail_damage: float = 0.25          # rice yield x (1 - this * snails in 3x3)
    irrigate_depth: float = 0.05        # water moved per tick into a thirsty paddy from the nearest water
    irrigate_min_depth: float = 0.10    # a source must be at least this deep (a shallow inland river still counts)
    paddy_seep: float = 0.0015          # paddy water lost per tick besides evaporation
    caltrop_shade: float = 0.5          # water caltrop shades its cell (cover for algae/shrimp)
    # --- paddy fertility (D-035/D-036): derived, not tuned ---
    paddy_flooded: float = 0.15         # standing water above this = anoxic soil (same threshold rice fitness uses)
    anox_drain: float = 0.3             # crop drain under standing water x this (anaerobic decomposition ~1/3-1/5 of aerobic)
    flush_rate: float = 0.0006          # a drained paddy pays back its deferred decomposition at this rate per tick (Birch flush)
    trap_to_h: float = 0.03             # river load carried in by irrigation -> soil h. Sized so murky water (q~0.85) supplies
                                        # ~15% of a paddy's drain (irrigation water = 3-30 kg N/ha/crop vs ~100 removed; 待讀 Roger & Ladha 1992).
                                        # At 0.8 the paddy re-drank its own runoff and h ran to the cap (conservation hole).
    humify: float = 1.0 / 6000          # deferred (labile) decomposition turns stable at this rate: two-pool SOM, not an unbounded debt
    winter_sun: float = 0.35            # crop yield in midwinter relative to midsummer (day length x sun angle, ~35N)

    # --- endorheic basin (D-027): no sea, a closed bowl, dry air, salt that stays ---
    basin: int = 0                      # 1 = the last reach is a closed basin (sea must be 0)
    basin_depth: float = 1.2
    arid_preset: int = 1                # basin maps get a dry sky automatically (rain 1e-4, evap x4, salty river)
    evap_mult: float = 1.0              # arid maps: 2-4
    salt_bg: float = 0.02               # salinity the springs bring in (rivers are not pure)

    # --- ground water (D-021): fresh and salt in the soil, ratio decides salinisation ---
    salt_flip: float = 0.5              # soil salinity above this = 鹽化土 (crops die, halophytes only)

    # --- coast dynamics (D-020) ---
    sea_slow_amp: float = 0.12          # slow sea-level swing on top of the tide (small by default)
    sea_slow_period: int = 24000        # ticks
    wave_rate: float = 0.0025           # scour per tick on unvegetated shore touching salty water, x tide_amp
    dune_max: float = 1.2               # dune grass stops building this far above high tide
    flora_spinup: int = 1200            # generation: run the world (plants, soil water) this long before play

    # --- mangrove (紅樹林): a tree that lives in shallow, slow, brackish, sunlit water ---
    mg_d_lo: float = 0.03               # depth band (on the tide-smoothed depth)
    mg_d_hi: float = 0.45
    mg_v_est: float = 0.6               # seedlings need calmer water than this (window of opportunity)
    mg_s_lo: float = 0.15               # salinity band: brackish only (freshwater kills it slowly)
    mg_s_hi: float = 1.0
    mg_light: int = 8                   # seedlings need this much light (D-017: none under a closed canopy, edges ok)
    mg_p_est: float = 0.02              # establishment chance per tick per eligible cell with a neighbour stand
    mg_p_far: float = 0.0005            # ... without one (propagules drift)
    mg_adult_age: int = 400             # ticks to become an adult (shade, trap, litter)
    mg_trap: float = 8.0                # sediment deposition multiplier in and around adults
    mg_litter: float = 0.02             # detritus added to own cell per tick (shrimp nursery)
    mg_die_p: float = 0.00005           # old age
    mg_dry_ticks: int = 600             # dried out this long -> becomes a land tree on rich peat
    mg_fresh_ticks: int = 900           # salinity below band this long -> dies
    mg_seed: int = 8                    # initial stands placed on eligible estuary cells
    crab_pressure: float = 0.25         # establishment chance x (1 - this * crabs in 3x3)

    # --- creatures (shared movement/breeding rules; habitat functions live in creatures.py) ---
    species: str = "shrimp,medaka,crab,snail,beetle,carp,boar"   # which species to spawn (brine for salt lakes)
    boar_damage: float = 0.35           # D-042: dry-field yield x (1 - this * boars in 3x3)
    boar_root: float = 0.03             # per tick chance a boar turns the ground cover under it to bare earth
    boar_mast: float = 0.5              # per tick chance a boar on a ripe nut/fruit cell eats it
    n_shrimp: int = 36                  # start population, applies to every species
    eval_every: int = 20                # ticks between habitat evaluations (MC sensors: 20)
    search_radius: int = 12             # how far a shrimp looks for a better tile (MC bee->hive: 20)
    move_threshold: float = 0.45        # below this satisfaction the shrimp wants to move
    hysteresis: float = 0.15            # target must beat current score by this much
    wander_prob: float = 0.125          # per tick chance to take a random step when idle
    w_quality: float = 0.50
    w_cover: float = 0.25
    w_food: float = 0.25
    score_mode: str = "geometric"       # "geometric" (Liebig-ish: any factor at 0 kills the score),
                                        # "min" (strict Liebig), "weighted" (sum — a bad tile can hide behind a good one)
    crowd_penalty: float = 0.08         # score lost per shrimp already in the 3x3 (local mob cap)
    yield_rate: float = 0.01            # resource per tick at satisfaction 1.0 (linear — 茨 D-003.3)
    starve_threshold: float = 0.10
    starve_ticks: int = 150             # MC animals never starve; ours do, slowly (~18 days)
    strand_ticks: int = 25              # dry ground this long (no wet neighbour) => stranded (~3 days)
    breed_threshold: float = 0.80
    breed_every: int = 600              # (legacy; life.py derives the interval from body mass now)
    density_cap: int = 4                # (legacy; life.py derives it from body mass now)
    immigrate_every: int = 75           # eggs on birds' feet: a near-extinct species gets a few newcomers where it could live (~9 days)
    immigrate_min: int = 6              # ... when fewer than this are alive
    immigrate_n: int = 3
    immigrate_score: float = 0.3        # ... and only if some cell scores at least this

    # --- life history (D-032, life.py): base constants for a shrimp (M = 1); everything else is allometry ---
    life_L0: int = 2500                 # shrimp lifespan in ticks at full satisfaction (D-033: short, so 8000 ticks show three generations)
    life_G0: int = 600                  # shrimp brood interval
    life_litter0: float = 2.0           # shrimp offspring per brood
    life_k0: float = 0.0037             # shrimp growth rate: mature (0.6 M) after ~500 effective ticks (a fifth of its life)
    life_D0: float = 4.0                # shrimp per 5x5 before breeding stops
    passive_min_sat: float = 0.6        # passive products (eggs, moults, wool) only from content animals

    # --- harvest (D-032 tier 0, the forager): the farmer kills n of a species within R of home every N ticks ---
    # --- labour (D-043 step 6): a trip costs time; food/ cell walked is the ruler every economy is measured with ---
    trip_time: int = 1                  # 1: a farmer is busy for 2 * distance * step_cost ticks per trip (0: trips are free, the old way)
    job_hours_per_cell: float = 0.5     # (retired D-050; kept for step_cost derivation)
    step_cost: float = 0.0              # ticks per cell walked; 0 = derived (__post_init__). Unused when work_hours > 0
    # --- labour budget (D-050): a household has people x work_hours a day; jobs cost hours; walking is 4 km/h over 4 m cells ---
    work_hours: float = 8.0             # per person per day
    hunger_floor: float = 0.4           # D-051: on an empty larder the household still manages this fraction of a day's work (tired, not dead)
    ration_days: float = 10.0           # the ration that sets today's strength is the last ~10 days' average
    cell_m: float = 4.0                 # a sim cell is 4 x 4 blocks of a metre
    walk_kmh: float = 4.0
    hours_till: float = 3.0             # open one plot by hand
    hours_tend_field: float = 0.15      # weeding, per field per visit (a block visit sums its fields)
    tend_days: float = 7.0              # a field wants weeding this often
    hours_wood: float = 2.0             # fell one tree and carry it
    hours_forage: float = 1.5           # one gathering trip
    hours_hunt: float = 4.0             # one hunting / fishing trip
    hours_industry: float = 3.0         # a visit to a working site, or a build
    hours_rotate: float = 0.5           # per paddy switched
    wood_target: float = 3.0            # cut wood when the ledger falls below this (if the farm uses wood at all)
    forage_every: int = 0               # ticks between foraging trips (0 = does not forage)
    # --- the household (D-044 step 1): food sits in a store, rots by its kind and the weather, and gets eaten ---
    people: int = 1                     # mouths at home
    eat_per_day: float = 21.0           # food units per day per person (2500 kcal; 1 unit = 120 kcal)
    eat_per_person: float = 0.0         # per tick; 0 = derived (__post_init__)
    decay_q10: float = 2.0              # spoilage doubles every 10 degrees (microbial Q10); winter is a larder
    decay_ref_t: float = 20.0
    forage_radius: int = 30             # how far from home a forager looks for something ripe
    hunt_every: int = 0                 # 0 = off
    hunt_n: int = 1
    hunt_radius: int = 12
    hunt_species: str = "shrimp"
    hunt_min_size: float = 0.5          # fraction of M: leave the small ones (a mesh size)

    # --- pen (D-032 tier 2): a fenced patch of water; inside cannot leave, outside cannot enter, fed from the ledger ---
    pen: str = ""                       # "y,x,h,w", or "pond" (a 5x5 on the pond, D-043), or empty
    # --- heredity (D-043): a few genes per animal; wild selection by birds, human selection by a sieve ---
    mutation: float = 0.04              # sd of the noise added to each gene at birth (after averaging the two parents)
    mutation_major_p: float = 0.02      # per birth: a large-effect mutation (the goldfish's xanthic allele is one gene, not a thousand small ones)
    mutation_major: float = 0.45        # ... of about this size
    bird_base: float = 0.0006           # per tick death chance of a fully conspicuous fish in open shallow water (herons)
    color_frailty: float = 0.35         # lifespan x (1 - this * color): the fancy ones are frail (goldfish history)
    pen_roof: int = 1                   # 1: no birds over the pen
    select_trait: str = ""              # "color": at pen harvest, the breeding stock kept is the most colourful, not the biggest
    select_keep: int = 4                # breeding stock kept in the pen at each harvest
    pen_stock: int = 6                  # animals the keeper puts in the pen at the start
    pen_species: str = "carp"
    pen_feed: float = 0.02              # detritus added per pen cell per tick (paid in wood from the ledger, 0.001 each)
    pen_waste: float = 0.004            # quality lost per animal per tick inside the pen (dung)
    pen_density_mult: float = 20.0      # fed animals tolerate this many times the wild density cap (aquaculture is ~100x Damuth; 3x left a 5x5 pond with ten fish, too few for a sieve to act on)

    def to_dict(self) -> dict:
        return asdict(self)

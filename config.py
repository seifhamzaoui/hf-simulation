# ----- CORE -----
col_width  = 70     # column width (x)
col_depth  = 60     # stacking depth (z)

bobine_spacing = 10 # distance between inner faces of columns
# ----- MANDREL -----
mandrel_thickness = 20

# ----- SECONDARY CONDUCTOR -----
p_cond_width      = 12.9     # radial build per layer
p_cond_thickness  = 6.9    # axial conductor thickness
p_cond_insul_thickness = 0.25
p_conduc_circular = False

radius = 8


s_cond_width      = 4.2     # radial build per layer
s_cond_thickness  = 7.2    # axial conductor thickness
s_cond_insul_thickness = 0.25
s_conduc_circular = False

primary_n_turns_total   = 17
primary_n_layers        = 1
primary_turns_per_layer = 17
p_flyback  =False

secondary_n_turns_total   = 43
secondary_n_layers        = 1
secondary_turns_per_layer = 43
s_flyback  =False

p_layer_insulation = 1
s_layer_insulation = 1
prim_secondary_gap = 11

# ----- DISTANCES -----
gap_winding_yoke    = 10
gap_isulation_yoke = 5

# entrefer par colonne
entrefer = 0.15 # mm
nombre_entrefer = 1

# simulation_ngsolve.run_dc_resistance()'s litz_aware default -- False
# (solid-copper, full geometric cross-section) is what was VALIDATED
# against Q3D to <0.2% (see that function's docstring); set True to divide
# by each winding's MATERIALS[...]["litz"]["fill_factor"] instead, matching
# simulation_ngsolve_litz.py / simulation_ngsolve_litz_ratio.py's own
# fill-factor-aware Rdc, at the cost of no longer matching that Q3D check.
DC_RESISTANCE_LITZ_AWARE = True

# simulation_ngsolve_litz_ratio.run_ratio_sweep()'s representative-sample
# size PER WINDING (how many turns near the middle of each winding stack
# are actually meshed/solved to compute the AC/DC ratio -- see
# pick_middle() in that file). Larger = better averaging across turns,
# closer to a true whole-winding result, at the cost of a bigger mesh and
# slower solve per frequency (scales with count, not just count^2, since
# each extra ring adds its own current-source solve + mesh regions, but
# the per-frequency factorization cost also grows with total DOF count --
# LITZ_RATIO_SAMPLE_COUNT_PRIMARY + LITZ_RATIO_SAMPLE_COUNT_SECONDARY rings
# ALL share one mesh/Glue(), so cost is driven by their SUM, not either
# alone). Primary (17 total turns) and secondary (43 total turns) are
# split into separate settings since they have very different total turn
# counts and very different meshing cost per added turn -- a count that's
# fine for one winding may not be for the other. Was a single shared
# LITZ_RATIO_SAMPLE_COUNT=16 (both windings, same value); split here so
# each can be tuned independently. Pass primary_count=/secondary_count=
# to run_ratio_sweep() to override either for one call without editing
# this file -- see that function's own docstring.
LITZ_RATIO_SAMPLE_COUNT_PRIMARY = 16
LITZ_RATIO_SAMPLE_COUNT_SECONDARY = 40

# Simulation parameters

sim_frequencies = [f * 1e3 for f in [
    1, 2, 5, 10, 20, 50, 100, 200, 500,
    1000, 2000, 2500, 4000,
    5000,  6000, 7000,
    8000, 9000, 10000
]]

sweep_active = False
min_freq_kHz = 1
max_freq_kHz = 500
number_of_point = 2

# ----- MATERIALS -----
# Single source of truth for every material used across the geometry and
# every simulation backend. Keyed by the solid name/prefix exactly as it
# appears in the geometry array (transformer_geometry.py /
# transformer_geometry_rectangular.py) -- edit values here once per run.
#   "pattern" : name pattern matched against mesh region names in
#               simulation_ngsolve.py / simulation_ngsolve_cuda.py (".*"
#               suffix for solids that carry a numeric index, e.g.
#               ringp1..ringp17; exact name otherwise). None = background
#               region, not an explicit solid.
#   "aedt"    : material_name assigned on solids in Q3D / Maxwell 3D
#               (simulation.py, simulation_Maxwell.py) -- must exist in
#               AEDT's material library under that exact name.
#   "ngsolve" : numeric properties (sigma, eps_r, mu_r) used directly as
#               coefficient-function values in simulation_ngsolve.py /
#               simulation_ngsolve_cuda.py (no external library involved).
MATERIALS = {
    "ringp": {
        "pattern": "ringp.*",
        "aedt": "copper",
        "ngsolve": {"sigma": 5.8e7, "eps_r": 1.0, "mu_r": 1.0},
        # Litz-wire bundle parameters for simulation_ngsolve_litz.py's
        # frequency sweep -- PLACEHOLDERS, edit to match the real wire:
        #   strand_diameter_m : single-strand copper diameter (m)
        #   n_strands         : number of strands in the bundle
        #   fill_factor       : (total strand copper area) / (bundle cross-
        #                       section area actually modeled by the ringp
        #                       solid) -- accounts for insulation + packing
        #                       gaps between round strands (~0.6-0.8 typical
        #                       for round strands, unless already baked into
        #                       p_cond_width/p_cond_thickness above).
        "litz": {"strand_diameter_m": 0.2e-3, "n_strands": 5350, "fill_factor": 0.46},
    },
    "rings": {
        "pattern": "rings.*",
        "aedt": "copper",
        "ngsolve": {"sigma": 5.8e7, "eps_r": 1.0, "mu_r": 1.0},
        "litz": {"strand_diameter_m": 0.2e-3, "n_strands": 445, "fill_factor": 0.46},
    },
    "Core": {
        # Real physical material. In the capacitance stage the Core is
        # instead assigned "ringp"/"rings" copper (see simulation.py /
        # simulation_Maxwell.py) so it can be solved as a plain signal
        # conductor -- that swap is a solve-stage trick, not a material
        # property, so it's left as an explicit override at the call site.
        "pattern": "Core",
        "aedt": "ferrite",  # matches the Q3D/Maxwell material library value
        "ngsolve": {"sigma": 1e-2, "eps_r": 1.0, "mu_r": 30000.0},
        # AC/eddy-current parameters for simulation_ngsolve_litz.py's
        # frequency sweep. The core gets its OWN complex-permeability
        # homogenization -- the classical lamination formula
        #   mu_complex/mu_r = tanh(k*d/2) / (k*d/2),  k=(1-1j)/skin_depth
        # (same k as the windings' Bessel/strand formula, different closed
        # form since a lamination is a slab, not a round strand) -- rather
        # than the windings' explicit j*w*sigma term, because that term
        # only captures loss, not the accompanying (small but real)
        # reduction in effective permeability from partial flux exclusion,
        # which the lamination formula gives for free.
        #   sigma_ac              : core bulk conductivity for the skin-
        #                           depth calc (S/m) -- ferrites are
        #                           deliberately poor conductors (why
        #                           ferrite beats iron at high frequency),
        #                           so expect this loss channel to be small.
        #   lamination_thickness_m : PLACEHOLDER -- edit to match the real
        #                           core. Most soft ferrites used here are
        #                           actually solid (unlaminated) blocks, in
        #                           which case set this to the core's own
        #                           physical thickness in the relevant
        #                           cross-section (col_width or col_depth,
        #                           in meters) so the formula reduces to
        #                           modeling the WHOLE core cross-section
        #                           as one "lamination" -- do not leave the
        #                           thin-lamination placeholder below
        #                           unedited if the real core isn't
        #                           actually laminated.
        #   mu_r                  : reuses the same real, linear low-
        #                           frequency permeability as the DC stage
        #                           -- core saturation/nonlinearity is not
        #                           modeled (see simulation_ngsolve.py's
        #                           B-field diagnostic from this session:
        #                           core flux density stayed in the sub-mT
        #                           range, nowhere near ferrite saturation).
        #   fill_factor           : lamination STACKING factor (fraction of
        #                           the core's geometric cross-section that
        #                           is actually magnetic material, the rest
        #                           being inter-lamination insulation) --
        #                           PLACEHOLDER, 1.0 if the real core is a
        #                           solid (unlaminated) block. Applied as
        #                           mu_eff = 1 + fill_factor*(mu_complex-1),
        #                           the standard stacking-factor
        #                           homogenization (insulation layers
        #                           themselves contribute mu_r=1).
        "ac": {"sigma_ac": 8e5, "lamination_thickness_m": 2e-5, "mu_r": 30000.0, "fill_factor": 0.75},
    },
    "pinsulator": {
        "pattern": "pinsulator.*",
        "aedt": "teflon_based",
        "ngsolve": {"sigma": 0.0, "eps_r": 1.25, "mu_r": 1.0},
    },
    "sinsulator": {
        "pattern": "sinsulator.*",
        "aedt": "teflon_based",
        "ngsolve": {"sigma": 0.0, "eps_r": 1.25, "mu_r": 1.0},
    },
    "p_layer insulator": {
        "pattern": "p_layer insulator",
        "aedt": "teflon_based",
        "ngsolve": {"sigma": 0.0, "eps_r": 2.1, "mu_r": 1.0},
    },
    "s_layer insulator": {
        "pattern": "s_layer insulator",
        "aedt": "teflon_based",
        "ngsolve": {"sigma": 0.0, "eps_r": 2.1, "mu_r": 1.0},
    },
    "primary_secondary_insulation": {
        "pattern": "primary_secondary_insulation",
        "aedt": "teflon_based",
        "ngsolve": {"sigma": 0.0, "eps_r": 1.2, "mu_r": 1.0},
    },
    "entrefer": {
        # Air-gap solid cut out of the Core (see transformer_geometry_rectangular.py) --
        # now exported as its own named body ("entrefer") in transformer_model_closed.step
        # instead of silently merging into the background air region, so its material
        # can be set/tuned independently of "air" if needed.
        "pattern": "entrefer.*",
        "aedt": "vacuum",
        "ngsolve": {"sigma": 0.0, "eps_r": 1.0, "mu_r": 1.0},
    },
    "air": {
        "pattern": None,  # background region, not an explicit solid
        "aedt": "vacuum",
        "ngsolve": {"sigma": 0.0, "eps_r": 1.0, "mu_r": 1.0},
    },
}
"""
NGSolve frequency sweep that computes AC/DC ratios for the primary and
secondary windings from a small, dynamically-chosen representative sample
(config.py's LITZ_RATIO_SAMPLE_COUNT_PRIMARY / _SECONDARY turns, picked
from the middle of each winding stack, never hardcoded indices, see
pick_middle() below -- set independently per winding since they have very
different total turn counts, 17 vs 43, and very different meshing cost per
added turn; run_ratio_sweep()'s primary_count/secondary_count arguments
override either for one call) -- then uses those ratios to scale the
FULL, already-computed DC reference matrices into AC-corrected
matrices -- far cheaper than
re-simulating all 60 turns at every frequency.

DC baseline matrices come from "./ngsolve matrices/DCR.mat" and
"./ngsolve matrices/induc.mat" (simulation_ngsolve.run_dc_resistance() /
run_inductance()'s own output) -- NOT from Q3D's "traited Values/" -- so
the DC baseline being scaled is internally consistent with this script's
own Litz/lamination material model. config.py's DC_RESISTANCE_LITZ_AWARE
should be True when DCR.mat was (re)generated, so that baseline itself
already accounts for the winding's fill_factor -- see run_dc_resistance()'s
own docstring in simulation_ngsolve.py.

Output: R_ratio_ngsolve.mat matches femm/R_Ratio.mat's own convention
(confirmed by inspecting that file this session) -- a DIAGONAL (N,N)
matrix per frequency (N = total ring count, same row/col order as
traitement_inductance.py's convention -- primary rows first, then
secondary), keyed "f_<freq>kHz", plus "frequencies_hz". Diagonal-only is
fine for R: DC resistance has no real mutual term between separate
conductors, so DCR.mat's own off-diagonal is already ~0.

L_ratio_ngsolve.mat is NOT diagonal, unlike femm/L_Ratio.mat -- a
diagonal-only ratio matrix would zero out induc.mat's real off-diagonal
mutual-inductance terms when multiplied element-wise into the AC-corrected
matrix, which is real physics that shouldn't be discarded. Instead every
entry (i,j) is populated: ratio_L_primary(f) where both i,j are primary
turns, ratio_L_secondary(f) where both are secondary, and the geometric
mean sqrt(ratio_L_primary*ratio_L_secondary) for primary-secondary cross
(mutual) entries -- a reasonable default since this representative-sample
approach never directly measures a primary-secondary mutual AC/DC ratio
itself.

Circuit setup -- COMBINED injection, REVERSED current: the sampled
primary turns are driven in series with I_primary=1A; the sampled
secondary turns are driven SIMULTANEOUSLY, in series, in the OPPOSITE
direction, with I_secondary = -I_primary * (len(mid_primary)/
len(mid_secondary)) -- ampere-turn balance over the ACTIVATED (sampled)
turn counts, NOT the winding's full total turn count -- the sampled
bundle IS the circuit actually being solved, so balancing against the
full-winding ratio would mismatch the real mmf the two sampled bundles
produce relative to each other. Both windings are solved in ONE combined
field (needed for L, which still uses the L_ij=Integral(A.J)dV identity
and is reported as an "apparent inductance under this specific loading
condition").

Rac, however, is NOT extracted via that same superposition identity
(-w*Im(L_complex)) anymore -- that identity's cross term is a genuine
"reflected impedance" from the OTHER winding (classical coupled-circuit
theory: Z1_apparent = jw(L11 + M*I2/I1)), and under our fixed, idealized
REAL current ratio it can legitimately subtract enough to push the total
below Rdc (observed: ratio < 1 in the 5-20kHz band) or even negative at
high frequency -- correct given what it's measuring, but not what "Rac"
should mean here. Instead, Rac comes from directly integrating the LOCAL
physical loss density
    p_loss = (w/2) * Im(nu) * |B|^2,  B = curl(A)
restricted to ONLY that winding's own conductor material (ringp.*/rings.*
regions), from the same combined-loading field solution -- a real
dissipated power is manifestly non-negative, so this Rac can never fall
below Rdc. See _solve_combined()'s docstring for the full derivation and
the not-yet-empirically-validated factor of 2 (assumes I=1A is a peak
amplitude).

Rdc for each bundle comes from the Ohmic identity R=Integrate(J^2/sigma_eff)
applied to the SAME normalized current field already built for the AC
solve (sigma_eff divides by the winding's Litz fill_factor), summed over
the 3 sampled turns -- independent of the AC solve/loading condition.
"""

import os

import numpy as np
from scipy.io import loadmat, savemat

from netgen.occ import OCCGeometry, Box, Pnt, Glue
from ngsolve import Mesh, HCurl, GridFunction, BilinearForm, LinearForm, curl, dx, Integrate, Conj, InnerProduct

import simulation_ngsolve as sim
import simulation_ngsolve_litz as lz
from config import sim_frequencies
import config as _config

MU0 = sim.MU0
MATRIX_DIR = sim.MATRIX_DIR
DC_MATRIX_DIR = sim.MATRIX_DIR  # "./ngsolve matrices" -- see module docstring for why not "traited Values"
os.makedirs(MATRIX_DIR, exist_ok=True)


def pick_middle(names, count=3):
    """count consecutive names centered on the middle of the list -- no
    hardcoded indices, works for any N and any (odd or even) count."""
    n = len(names)
    if n < count:
        raise ValueError(f"only {n} rings available, need at least {count}")
    start = (n - count) // 2
    return names[start:start + count]


def build_ratio_geometry(mid_primary, mid_secondary, order=1, pad=0.1,
                          core_maxh=0.03, ring_maxh=0.006, battery_maxh=0.006,
                          air_maxh=0.1, entrefer_maxh=None, entrefer_solid_maxh=0.01):
    """Builds/meshes the 6-ring geometry once (shared across DC and every
    swept frequency).

    entrefer_solid_maxh default is 2mm here (coarser than
    simulation_ngsolve_litz.py's own 1mm default) -- this is the single
    largest, turn-count-independent contributor to mesh size (confirmed
    this session: meshing the entrefer at 1mm for even a SINGLE sampled
    ring already produced 367k-1.6M elements and repeatedly caused
    MemoryError during complex sparse Cholesky factorization on this
    machine once free-DOF crossed ~1-2M). Raising LITZ_RATIO_SAMPLE_COUNT
    (config.py) adds more turns on TOP of that fixed entrefer cost, so
    2mm here buys back headroom for a larger turn-count sample without
    hitting that wall again.

    core_maxh/ring_maxh/battery_maxh/air_maxh were also coarsened
    (0.02->0.03, 0.004->0.006, 0.0015->0.003, 0.03->0.05) after
    LITZ_RATIO_SAMPLE_COUNT=15 (30 total rings) never finished meshing
    within 10 minutes even with the coarser entrefer above -- that failure
    happened before any "mesh: N elements" line printed, so it's not
    confirmed whether coarsening these actually fixes it (could be a
    boolean/Glue() topological cost from 30 individual ring+battery cut
    solids, not purely an element-count/resolution issue) -- needs a
    fresh test at the same count to confirm."""
    all_ring_names = list(mid_primary) + list(mid_secondary)
    sigma_copper = sim.MATERIALS["ringp"]["sigma"]

    print("Loading closed-ring geometry...")
    solids = sim.load_solids(sim.STEP_FILE_CLOSED)
    core = sim.core_solid_from(solids)
    core.mat("Core")
    core.maxh = core_maxh
    sim.refine_entrefer_faces(core, maxh=entrefer_maxh)

    entrefer_solid = sim.entrefer_solid_from(solids)
    if entrefer_solid is not None:
        entrefer_solid.mat("entrefer")
        entrefer_solid.maxh = entrefer_solid_maxh

    ring_parts = []
    for name in all_ring_names:
        ring = solids[name]
        battery = ring * sim._ring_battery_tool(ring)
        ring_passive = ring - battery
        ring_passive.mat(name)
        ring_passive.maxh = ring_maxh
        battery.mat(f"{name}_battery")
        battery.maxh = battery_maxh
        ring_parts.append(ring_passive)
        ring_parts.append(battery)

    bb = OCCGeometry(sim.STEP_FILE_CLOSED).shape.bounding_box
    bb = ([v * 1e-3 for v in bb[0]], [v * 1e-3 for v in bb[1]])
    lo = [bb[0][i] - pad for i in range(3)]
    hi = [bb[1][i] + pad for i in range(3)]
    box = Box(Pnt(*lo), Pnt(*hi))
    box.faces.name = "outer"
    air = box - core
    if entrefer_solid is not None:
        air = air - entrefer_solid
    for name in all_ring_names:
        air = air - solids[name]
    air.mat("air")
    air.maxh = air_maxh

    glued_parts = [air, core] + ring_parts
    if entrefer_solid is not None:
        glued_parts.append(entrefer_solid)

    print("Meshing (6-ring sample, shared across DC + every frequency)...")
    geo = OCCGeometry(Glue(glued_parts))
    mesh = Mesh(geo.GenerateMesh(maxh=air_maxh))
    print(f"mesh: {mesh.ne} elements")

    print("Building each selected ring's current source...")
    J = {}
    Rdc_turn = {}
    for k, name in enumerate(all_ring_names):
        # _solve_ring_current's normalized-J SHAPE doesn't depend on the
        # absolute conductivity value (uniform sigma within the ring
        # domain cancels out of the normalization) -- but Rdc extraction
        # (P = Integral(J^2/sigma) dV) very much does, so the EFFECTIVE
        # (fill-factor-reduced) conductivity must be used here even though
        # the solve itself can keep using the bulk copper sigma.
        J[name] = sim._solve_ring_current(mesh, name, sigma_copper)
        kind = "ringp" if name.startswith("ringp") else "rings"
        sigma_eff = sigma_copper * lz.LITZ_PARAMS[kind]["fill_factor"]
        Rdc_turn[name] = Integrate(J[name] * J[name] / sigma_eff, mesh)
        print(f"  {k + 1}/{len(all_ring_names)} {name}: current source ready, "
              f"Rdc_turn={Rdc_turn[name]:.6g} ohm (fill_factor={lz.LITZ_PARAMS[kind]['fill_factor']:.3g})")

    Ldom = max(hi[i] - lo[i] for i in range(3))
    return mesh, J, Rdc_turn, Ldom


# module-level so _solve_combined (called many times per sweep) doesn't
# need them threaded through every call -- set by run_ratio_sweep() before use.
J_primary_bundle_names = []
J_secondary_bundle_names = []


def _solve_combined(mesh, Ldom, J_primary_bundle, J_secondary_bundle, I_primary, I_secondary,
                     freq_hz, order=1, reg_factor=1e-4, core_fill_factor_aware=True):
    """Solves the primary-alone and secondary-alone unit-current fields
    (A_p, A_s) at a given frequency, then reconstructs the COMBINED
    (reversed/opposing, ampere-turn-balanced -- see module docstring)
    field as the linear combination A = I_primary*A_p + I_secondary*A_s --
    valid because the curl-curl operator is linear, so this is exactly
    the same physical field the old single combined-RHS solve produced,
    just decomposed by source. Solving two RHS instead of one costs only
    one extra back-substitution against the SAME factorization (like
    simulation_ngsolve.run_inductance()'s N-ring loop), not a second
    factorization.

    Returns (L_primary, L_secondary, Rac_loss_primary, Rac_loss_secondary):

    L_* are each bundle's own apparent self-linkage under the combined
    loading condition -- algebraically identical to the old
    L=Integral(A.J)dV/I^2 identity (verified via reciprocity: since A_p
    solves a(A_p,v)=Integral(J_primary.v), Integral(A.J_primary)dV =
    a(A, A_p) for the SAME bilinear form a(u,v)=Integral(nu*curl(u).curl(v))
    used to assemble the system) -- EXCEPT the energy form a(.,.) used
    here can be evaluated with a fill_factor-weighted nu*energy_weight
    instead of raw nu, which the old A.J form could not do (J is zero
    outside the conductors, so it can't "see" the core at all -- weighting
    requires the field-energy form, evaluated over the whole mesh).
    core_fill_factor_aware=True (default): config.py's
    MATERIALS["Core"]["ac"]["fill_factor"] multiplies the Core's AND
    entrefer's share of that energy integral (windings/air stay weight
    1.0) when computing L_primary/L_secondary -- mirrors
    simulation_ngsolve.run_inductance()'s same flag. mu_r itself is left
    exactly as core_mu_effective()/complex_mu_litz() already compute it
    (raw, undiluted) -- this function never edits permeability, only how
    L is extracted from the solved field afterward.

    Rac_loss_* are UNAFFECTED by core_fill_factor_aware -- computed a
    DIFFERENT way, per explicit request, to get a "real" (always
    non-negative) Rac instead of the -w*Im(L_complex) identity applied to
    a combined/loaded solve -- that identity's cross term (reflected
    impedance from the OTHER winding, see the "reflected impedance"
    explanation given this session) can legitimately go negative under a
    fixed, idealized real current ratio, which is why the ratio was
    dipping below 1.0 in the 5-20kHz band.

    Instead, Rac_loss_* integrates the LOCAL physical loss density
        p_loss = (w/2) * Im(nu) * |B|^2,  B = curl(A)
    restricted (via definedon) to ONLY that winding's own conductor
    material -- a real dissipated power can never be negative, so this
    is manifestly free of the reflected-impedance sign issue, at the cost
    of no longer being expressible as a simple correction to Rdc via
    superposition alone (it's the ACTUAL loss in that winding's own
    copper, under this specific combined-loading field solution). This
    integral only ever touches ringp.*/rings.* regions -- it was never
    core-weighted to begin with, so it already reflects only the
    WINDINGS' own Litz fill_factor (via complex_mu_litz -> nu, same as
    always), independent of the core's fill_factor entirely.
        R_ac_from_loss = 2*P_loss_winding / I_winding^2
    The factor of 2 assumes I=1A is a peak (not RMS) amplitude, consistent
    with every other unit-current convention in this codebase -- NOT yet
    empirically cross-checked against the already-validated isolated
    single-ring Rac from simulation_ngsolve_litz.py; do that check before
    trusting the absolute scale here (see this function's caller for the
    flagged caveat)."""
    mu_complex_p = lz.complex_mu_litz(freq_hz, lz.LITZ_PARAMS["ringp"]["strand_diameter_m"], sim.MATERIALS["ringp"]["sigma"])
    mu_complex_s = lz.complex_mu_litz(freq_hz, lz.LITZ_PARAMS["rings"]["strand_diameter_m"], sim.MATERIALS["ringp"]["sigma"])
    mu_complex_core = lz.core_mu_effective(freq_hz)

    is_complex = freq_hz != 0
    if not is_complex:
        # freq_hz=0 uses a REAL FES -- complex_mu_litz()/core_mu_effective()
        # still return Python `complex` (zero imaginary part) at f=0, which
        # NGSolve's MaterialCF can't evaluate against a real-valued space
        # ("no real evaluate for ConstantCF-Complex") -- cast down here.
        mu_complex_p, mu_complex_s, mu_complex_core = mu_complex_p.real, mu_complex_s.real, mu_complex_core.real
    mat_mu = {"Core": mu_complex_core, "entrefer": complex(1.0, 0.0) if is_complex else 1.0}
    for name in J_primary_bundle_names:
        mat_mu[name] = mu_complex_p
        mat_mu[f"{name}_battery"] = mu_complex_p
    for name in J_secondary_bundle_names:
        mat_mu[name] = mu_complex_s
        mat_mu[f"{name}_battery"] = mu_complex_s

    default_mu = complex(1.0, 0.0) if is_complex else 1.0
    mu_r = mesh.MaterialCF(mat_mu, default=default_mu)
    nu = 1.0 / (MU0 * mu_r)
    reg = reg_factor * (1.0 / MU0) / Ldom ** 2

    if core_fill_factor_aware:
        core_fill_factor = lz.CORE_AC["fill_factor"]
        fill_val = complex(core_fill_factor, 0.0) if is_complex else core_fill_factor
        default_weight = complex(1.0, 0.0) if is_complex else 1.0
        energy_weight = mesh.MaterialCF({"Core": fill_val, "entrefer": fill_val}, default=default_weight)
    else:
        energy_weight = complex(1.0, 0.0) if is_complex else 1.0

    fes_h = HCurl(mesh, order=order, nograds=True, dirichlet="outer", complex=is_complex)
    uu, vv = fes_h.TnT()
    a = BilinearForm(nu * curl(uu) * curl(vv) * dx + reg * uu * vv * dx)
    a.Assemble()
    inv = a.mat.Inverse(fes_h.FreeDofs(), inverse="sparsecholesky")

    f_p = LinearForm(J_primary_bundle * vv * dx)
    f_p.Assemble()
    f_s = LinearForm(J_secondary_bundle * vv * dx)
    f_s.Assemble()

    gfA_p = GridFunction(fes_h)
    gfA_p.vec.data = inv * f_p.vec
    gfA_s = GridFunction(fes_h)
    gfA_s.vec.data = inv * f_s.vec

    gfA = GridFunction(fes_h)
    gfA.vec.data = I_primary * gfA_p.vec + I_secondary * gfA_s.vec

    M_pp = Integrate(nu * InnerProduct(curl(gfA_p), curl(gfA_p)) * energy_weight, mesh)
    M_ps = Integrate(nu * InnerProduct(curl(gfA_p), curl(gfA_s)) * energy_weight, mesh)
    M_ss = Integrate(nu * InnerProduct(curl(gfA_s), curl(gfA_s)) * energy_weight, mesh)

    L_primary = (I_primary * M_pp + I_secondary * M_ps) / I_primary ** 2
    L_secondary = (I_primary * M_ps + I_secondary * M_ss) / I_secondary ** 2

    if not is_complex:
        # f=0: no loss by definition (omega=0), and nu has no .imag on a
        # real-valued FES/CF -- skip straight to zero rather than evaluate
        # Im() on something that isn't complex-typed here.
        return L_primary, L_secondary, 0.0, 0.0

    omega = 2 * np.pi * freq_hz
    B = curl(gfA)
    B_mag_sq = InnerProduct(B, Conj(B)).real  # |B|^2, real-valued despite B being complex
    loss_density = 0.5 * omega * nu.imag * B_mag_sq

    P_loss_primary = Integrate(loss_density, mesh, definedon=mesh.Materials("ringp.*"))
    P_loss_secondary = Integrate(loss_density, mesh, definedon=mesh.Materials("rings.*"))
    Rac_loss_primary = 2 * P_loss_primary / I_primary ** 2
    Rac_loss_secondary = 2 * P_loss_secondary / I_secondary ** 2

    return L_primary, L_secondary, Rac_loss_primary, Rac_loss_secondary


def run_ratio_sweep(frequencies_hz=None, count=None, primary_count=None, secondary_count=None, **geom_kwargs):
    """Sample size resolution, per winding, in priority order:
    1. primary_count/secondary_count, if given explicitly -- lets you pick
       a different sample size for each winding (e.g. primary_count=3,
       secondary_count=7), useful since the two windings have very
       different total turn counts (17 vs 43) and very different meshing
       cost per added turn.
    2. count, if given -- applies the SAME size to both windings (kept for
       backward compatibility with one-off calls).
    3. config.py's LITZ_RATIO_SAMPLE_COUNT_PRIMARY / _SECONDARY (default
       path -- separate settings per winding)."""
    global J_primary_bundle_names, J_secondary_bundle_names

    if primary_count is None:
        primary_count = count if count is not None else getattr(_config, "LITZ_RATIO_SAMPLE_COUNT_PRIMARY", 3)
    if secondary_count is None:
        secondary_count = count if count is not None else getattr(_config, "LITZ_RATIO_SAMPLE_COUNT_SECONDARY", 3)

    if frequencies_hz is None:
        frequencies_hz = sim_frequencies
    frequencies_hz = list(frequencies_hz)

    primary, secondary = sim.ring_names(sim.STEP_FILE_CLOSED)
    all_ring_names_full = primary + secondary
    N_total = len(all_ring_names_full)

    mid_primary = pick_middle(primary, primary_count)
    mid_secondary = pick_middle(secondary, secondary_count)
    J_primary_bundle_names = mid_primary
    J_secondary_bundle_names = mid_secondary
    print(f"Representative sample: primary={mid_primary}  secondary={mid_secondary}")

    mesh, J, Rdc_turn, Ldom = build_ratio_geometry(mid_primary, mid_secondary, **geom_kwargs)

    def _sum_cf(names):
        # plain sum() starts from scalar 0, which fails on vector-valued
        # NGSolve CoefficientFunctions with a dimension mismatch (unlike
        # OCC shapes, which happen to support 0+shape) -- accumulate
        # explicitly instead.
        total = J[names[0]]
        for name in names[1:]:
            total = total + J[name]
        return total

    J_primary_bundle = _sum_cf(mid_primary)
    J_secondary_bundle = _sum_cf(mid_secondary)

    I_primary = 1.0
    # Ampere-turn ratio uses the ACTIVATED (sampled) turn counts --
    # primary_count/secondary_count, i.e. len(mid_primary)/len(mid_secondary)
    # -- not the winding's full total turn count (N_primary_total/
    # N_secondary_total). The sampled bundle IS the circuit being solved
    # here; balancing against the full-winding turn ratio would mismatch
    # the actual mmf the two SAMPLED bundles produce relative to each other.
    I_secondary = -I_primary * (len(mid_primary) / len(mid_secondary))
    print(f"I_primary={I_primary:.4g} A, I_secondary={I_secondary:.4g} A "
          f"(reversed/opposing, ampere-turn balance over ACTIVATED turns: "
          f"{len(mid_primary)} primary / {len(mid_secondary)} secondary)")

    Rdc_primary = sum(Rdc_turn[name] for name in mid_primary)
    Rdc_secondary = sum(Rdc_turn[name] for name in mid_secondary)
    print(f"Rdc_primary_bundle={Rdc_primary:.6g} ohm, Rdc_secondary_bundle={Rdc_secondary:.6g} ohm")

    print("Solving DC (f=0) baseline for Ldc...")
    Ldc_primary, Ldc_secondary, _, _ = _solve_combined(mesh, Ldom, J_primary_bundle, J_secondary_bundle,
                                                        I_primary, I_secondary, freq_hz=0.0)
    Ldc_primary, Ldc_secondary = Ldc_primary.real, Ldc_secondary.real
    print(f"Ldc_primary_bundle={Ldc_primary * 1e9:.2f} nH, Ldc_secondary_bundle={Ldc_secondary * 1e9:.2f} nH")

    ratio_R_primary = np.zeros(len(frequencies_hz))
    ratio_R_secondary = np.zeros(len(frequencies_hz))
    ratio_L_primary = np.zeros(len(frequencies_hz))
    ratio_L_secondary = np.zeros(len(frequencies_hz))

    for fi, freq_hz in enumerate(frequencies_hz):
        Lc_p, Lc_s, Rac_loss_p, Rac_loss_s = _solve_combined(
            mesh, Ldom, J_primary_bundle, J_secondary_bundle, I_primary, I_secondary, freq_hz=freq_hz)

        # Rac from the ACTUAL physical loss density integrated over each
        # winding's own conductor material (see _solve_combined's
        # docstring) -- manifestly non-negative (a real dissipated power
        # can't be negative), unlike the earlier -w*Im(L_complex) "apparent
        # impedance under load" identity, whose reflected-impedance cross
        # term could push the ratio below 1.0 (observed in the 5-20kHz
        # band) or even negative at high frequency. abs() kept only as a
        # defensive floor against numerical noise, not to fix a structural
        # sign problem anymore.
        R_ac_total_p = Rdc_primary + Rac_loss_p
        R_ac_total_s = Rdc_secondary + Rac_loss_s

        ratio_R_primary[fi] = abs(R_ac_total_p / Rdc_primary)
        ratio_R_secondary[fi] = abs(R_ac_total_s / Rdc_secondary)
        ratio_L_primary[fi] = abs(Lc_p.real / Ldc_primary)
        ratio_L_secondary[fi] = abs(Lc_s.real / Ldc_secondary)

        print(f"[{fi + 1}/{len(frequencies_hz)}] f={freq_hz:g} Hz  "
              f"R_ratio(p,s)=({ratio_R_primary[fi]:.4f}, {ratio_R_secondary[fi]:.4f})  "
              f"L_ratio(p,s)=({ratio_L_primary[fi]:.4f}, {ratio_L_secondary[fi]:.4f})")

    is_primary_row = np.array([name.startswith("ringp") for name in all_ring_names_full])

    R_ratio_mats = {}
    L_ratio_mats = {}
    for fi, freq_hz in enumerate(frequencies_hz):
        # R: diagonal only, matching femm/R_Ratio.mat's own convention --
        # DC resistance genuinely has no mutual term between separate
        # conductors, so DCR.mat's off-diagonal is already ~0 and a
        # diagonal scaling matrix doesn't lose anything real there.
        diag_R = np.where(is_primary_row, ratio_R_primary[fi], ratio_R_secondary[fi])
        R_ratio_mats[f"f_{sim.freq_label(freq_hz / 1e3)}"] = np.diag(diag_R)

        # L: FULL (N,N) scaling matrix, not diagonal -- induc.mat's
        # off-diagonal (mutual inductance) terms are real physics, and a
        # diagonal-only ratio matrix would zero them out entirely under
        # the element-wise multiply below. Primary-primary entries scale
        # by ratio_L_primary, secondary-secondary by ratio_L_secondary,
        # and primary-secondary CROSS (mutual) entries by the geometric
        # mean of the two -- a reasonable default for scaling a mutual
        # term shared between two differently-scaled windings; there's no
        # single "correct" convention here since this representative-
        # sample approach never directly measures a primary-secondary
        # mutual AC/DC ratio itself.
        row_ratio = np.where(is_primary_row, ratio_L_primary[fi], ratio_L_secondary[fi])
        L_scale = np.sqrt(np.outer(row_ratio, row_ratio))
        L_ratio_mats[f"f_{sim.freq_label(freq_hz / 1e3)}"] = L_scale

    R_ratio_mats["frequencies_hz"] = np.array(frequencies_hz).reshape(1, -1)
    L_ratio_mats["frequencies_hz"] = np.array(frequencies_hz).reshape(1, -1)

    r_path = os.path.join(MATRIX_DIR, "R_ratio_ngsolve.mat")
    l_path = os.path.join(MATRIX_DIR, "L_ratio_ngsolve.mat")
    savemat(r_path, R_ratio_mats)
    savemat(l_path, L_ratio_mats)
    print(f"Saved {r_path} (diagonal, femm-format), {l_path} (full (N,N), NOT diagonal -- preserves mutual terms)")

    dcr_path = os.path.join(DC_MATRIX_DIR, "DCR.mat")
    induc_path = os.path.join(DC_MATRIX_DIR, "induc.mat")
    if os.path.exists(dcr_path):
        dcr = loadmat(dcr_path)
        dcr_keys = [k for k in dcr if k.startswith("f_")]
        if dcr_keys:
            R_dc_matrix = dcr[dcr_keys[0]]
            if R_dc_matrix.shape != (N_total, N_total):
                print(f"[warn] DCR.mat shape {R_dc_matrix.shape} != expected ({N_total},{N_total}) -- skipping AC-corrected R matrix")
            else:
                out = {}
                for fi, freq_hz in enumerate(frequencies_hz):
                    key = f"f_{sim.freq_label(freq_hz / 1e3)}"
                    out[key] = R_ratio_mats[key] * R_dc_matrix
                out["frequencies_hz"] = R_ratio_mats["frequencies_hz"]
                r_ac_path = os.path.join(MATRIX_DIR, "DCR_ac_corrected.mat")
                savemat(r_ac_path, out)
                print(f"Saved {r_ac_path}")
    else:
        print(f"[warn] {dcr_path} not found -- skipping AC-corrected R matrix")

    if os.path.exists(induc_path):
        ind = loadmat(induc_path)
        ind_keys = [k for k in ind if k.startswith("f_")]
        if ind_keys:
            L_dc_matrix = ind[ind_keys[0]]
            if L_dc_matrix.shape != (N_total, N_total):
                print(f"[warn] induc.mat shape {L_dc_matrix.shape} != expected ({N_total},{N_total}) -- skipping AC-corrected L matrix")
            else:
                out = {}
                for fi, freq_hz in enumerate(frequencies_hz):
                    key = f"f_{sim.freq_label(freq_hz / 1e3)}"
                    out[key] = L_ratio_mats[key] * L_dc_matrix
                out["frequencies_hz"] = L_ratio_mats["frequencies_hz"]
                l_ac_path = os.path.join(MATRIX_DIR, "induc_ac_corrected.mat")
                savemat(l_ac_path, out)
                print(f"Saved {l_ac_path}")
    else:
        print(f"[warn] {induc_path} not found -- skipping AC-corrected L matrix")

    return {
        "frequencies_hz": frequencies_hz,
        "ratio_R_primary": ratio_R_primary, "ratio_R_secondary": ratio_R_secondary,
        "ratio_L_primary": ratio_L_primary, "ratio_L_secondary": ratio_L_secondary,
        "Rdc_primary": Rdc_primary, "Rdc_secondary": Rdc_secondary,
        "Ldc_primary": Ldc_primary, "Ldc_secondary": Ldc_secondary,
    }


if __name__ == "__main__":
    run_ratio_sweep()

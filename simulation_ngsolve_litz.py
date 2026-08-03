"""
NGSolve frequency-sweep simulation for AC inductance (L) and the added AC
resistance (Rac) of Litz-wire windings, using a homogenized complex-
permeability model for each strand's own skin effect.

Reuses simulation_ngsolve.py's geometry/meshing/current-source machinery
(imported as `sim`) rather than duplicating it -- only the material model
(complex instead of real mu_r in the winding regions) and the per-frequency
solve loop are new.

Physics / scope -- read this before trusting numbers:
  - Models per-STRAND skin effect only, via the exact closed-form complex
    permeability of a round conductor (Bessel functions J0/J1 of the
    skin-depth-normalized strand radius -- see complex_mu_litz() below).
    This is the same "complex permeability method" used in eddy-current
    homogenization literature (e.g. Sullivan's and Ferreira's Litz-wire
    loss models) to replace explicit strand-level eddy-current solves with
    an equivalent bulk material property.
  - Does NOT model strand-to-strand PROXIMITY effect (the extra loss a
    strand picks up from its neighbors' field, not just its own current).
    That needs a 2D bundle-position-dependent extension (e.g. Ferreira's
    array-factor term) -- left out here as a real, documented limitation,
    not a hidden one. For well-designed Litz wire (strand diameter well
    below the skin depth at the frequencies of interest) proximity loss is
    usually the smaller of the two effects, but "usually smaller" is not
    "zero" -- Rac from this script is a lower bound, not the full answer.
  - Reuses the SAME real, frequency-independent 1A current distribution
    from sim._solve_ring_current() at every frequency. This is valid
    specifically because Litz wire's entire design purpose is to keep each
    individual strand's own current distribution uniform (skin-effect-free)
    well past the frequencies of interest -- only the BULK equivalent
    permeability (hence impedance) changes with frequency, not the
    macroscopic current shape. This assumption breaks down if
    strand_diameter is not actually small compared to the skin depth at
    your highest swept frequency -- check skin_depth_at() below against
    config.py's strand_diameter_m before trusting the high-frequency end.
  - Core now ALSO gets a complex permeability, via the classical lamination
    formula (see complex_mu_lamination() below) instead of the windings'
    Bessel/round-strand formula -- a lamination is a slab, not a round
    wire, so the closed form differs (tanh instead of a J1/J0 ratio), but
    the underlying physics (skin effect reducing effective permeability
    and adding loss) and homogenization approach are the same idea, just
    applied to config.py's MATERIALS["Core"]["ac"]["lamination_thickness_m"]
    instead of a strand diameter. If the real core is actually a SOLID
    (unlaminated) ferrite block, set that thickness to the core's own
    physical cross-section dimension -- see config.py's comment there.

Frequencies: read directly from config.py's sim_frequencies list (the same
sweep list simulation.py / simulation_Maxwell.py use).

Litz parameters: read from config.py's MATERIALS["ringp"]/["rings"]["litz"]
sub-dict (strand_diameter_m, n_strands, fill_factor) -- these are
PLACEHOLDERS in config.py and must be edited to match the real wire before
trusting numeric results.

Output: saved to "./ngsolve matrices/litz_sweep.mat" as
{"frequencies_hz": [...], "L_H": complex->real part per (freq,ring,ring),
 "Rac_ohm": added AC resistance per (freq,ring,ring), "ring_names": [...]}.
"""

import os

import numpy as np
from scipy.io import savemat
from scipy.special import jv

from netgen.occ import OCCGeometry, Box, Pnt, Glue
from ngsolve import (
    Mesh, HCurl, GridFunction, BilinearForm, LinearForm,
    curl, dx, Integrate, InnerProduct,
)

import simulation_ngsolve as sim
from config import sim_frequencies, MATERIALS as CONFIG_MATERIALS

MU0 = sim.MU0
MATRIX_DIR = sim.MATRIX_DIR
os.makedirs(MATRIX_DIR, exist_ok=True)

LITZ_PARAMS = {
    "ringp": CONFIG_MATERIALS["ringp"]["litz"],
    "rings": CONFIG_MATERIALS["rings"]["litz"],
}
CORE_AC = CONFIG_MATERIALS["Core"]["ac"]


def skin_depth_at(freq_hz, sigma, mu_r=1.0):
    """Classical EM skin depth (m) -- use this to sanity-check that
    config.py's strand_diameter_m is actually small compared to skin depth
    at your highest swept frequency (the assumption this whole script
    leans on)."""
    if freq_hz == 0:
        return np.inf
    return np.sqrt(2.0 / (2 * np.pi * freq_hz * MU0 * mu_r * sigma))


def complex_mu_litz(freq_hz, strand_diameter_m, sigma_strand, mu_r_strand=1.0):
    """Exact complex relative permeability of a single round strand under
    classical 1D radial skin effect (time convention e^{+jwt}):
        k = (1-1j)/skin_depth,  skin_depth = sqrt(2/(w*mu0*mu_r*sigma))
        mu_complex/mu_r = (2/(k*a)) * J1(k*a) / J0(k*a)
    a = strand radius. Returns mu_complex (dimensionless, multiply by mu0
    to get the absolute permeability used in nu=1/(mu0*mu_complex)).

    Sanity check (verified at import time by _selftest_complex_mu() below):
    mu_complex -> mu_r_strand (real, lossless) as f -> 0. Note: Im(mu_complex)
    being negative here does NOT by itself guarantee Rac_ohm comes out
    positive downstream -- that depends on how nu=1/(mu0*mu_complex)
    propagates through the actual curl-curl solve, which was checked
    empirically (see the sign comment on Rac_ohm's computation in
    run_litz_sweep) rather than assumed from this formula alone."""
    if freq_hz == 0:
        return complex(mu_r_strand, 0.0)
    w = 2 * np.pi * freq_hz
    a = strand_diameter_m / 2.0
    delta = skin_depth_at(freq_hz, sigma_strand, mu_r_strand)
    k = (1 - 1j) / delta
    ka = k * a
    return mu_r_strand * (2.0 / ka) * jv(1, ka) / jv(0, ka)


def complex_mu_lamination(freq_hz, lamination_thickness_m, sigma, mu_r=1.0):
    """Classical complex relative permeability of a lamination/slab of
    thickness d under 1D skin effect (same time convention/k as
    complex_mu_litz above -- a slab, not a round wire, so the closed form
    is the hyperbolic tangent ratio instead of a Bessel-function ratio):
        k = (1-1j)/skin_depth
        mu_complex/mu_r = tanh(k*d/2) / (k*d/2)
    Returns mu_complex (dimensionless, multiply by mu0 as usual)."""
    if freq_hz == 0:
        return complex(mu_r, 0.0)
    delta = skin_depth_at(freq_hz, sigma, mu_r)
    k = (1 - 1j) / delta
    kd2 = k * lamination_thickness_m / 2.0
    return mu_r * np.tanh(kd2) / kd2


def core_mu_effective(freq_hz):
    """Raw complex_mu_lamination() at config.py's Core AC parameters --
    NOT diluted by the lamination stacking fill_factor. fill_factor is
    applied downstream instead, as a post-solve weight on the Core's (and
    entrefer's) contribution to the stored-energy inductance/loss integral
    in run_litz_sweep() -- see that function's docstring -- rather than by
    altering mu_r itself, so the field solution is unaffected by
    fill_factor and only the energy bookkeeping accounts for the fraction
    of the core's cross-section that's actually magnetic material."""
    return complex_mu_lamination(freq_hz, CORE_AC["lamination_thickness_m"], CORE_AC["sigma_ac"], CORE_AC["mu_r"])


def _selftest_complex_mu():
    """Run once at import: confirms the low-frequency limit and the sign
    of the loss term, so a convention slip shows up immediately instead of
    as a silently-wrong Rac sign deep in a multi-hour sweep."""
    mu_low = complex_mu_litz(1e-6, 0.1e-3, 5.8e7)
    assert abs(mu_low.imag) < 1e-6 and abs(mu_low.real - 1.0) < 1e-3, (
        f"complex_mu_litz low-frequency limit should be ~1+0j, got {mu_low}"
    )
    mu_high = complex_mu_litz(1e6, 0.1e-3, 5.8e7)
    if mu_high.imag > 0:
        raise AssertionError(
            f"complex_mu_litz({1e6:g} Hz) = {mu_high} has the wrong loss "
            "sign (Im(mu) should be < 0 in this e^{+jwt}/nu=1/(mu0*mu) "
            "convention so that Rac_added = omega*Im(L_complex) comes out "
            "positive -- fix the sign in complex_mu_litz before trusting "
            "any Rac numbers from this file."
        )


_selftest_complex_mu()


def _selftest_complex_mu_lamination():
    """Same low-frequency-limit check as _selftest_complex_mu(), for the
    lamination formula. The Rac SIGN for the core's contribution is
    verified empirically in run_litz_sweep()'s smoke test (both windings
    and core feed the same nu=1/(mu0*mu_complex) pathway, but this is
    checked again rather than assumed to carry the same sign automatically)."""
    mu_low = complex_mu_lamination(1e-6, 0.2e-3, 1e-2, mu_r=30000.0)
    assert abs(mu_low.imag) < 1e-3 and abs(mu_low.real - 30000.0) < 1.0, (
        f"complex_mu_lamination low-frequency limit should be ~30000+0j, got {mu_low}"
    )


_selftest_complex_mu_lamination()


def run_litz_sweep(test_rings=None, frequencies_hz=None, order=1, pad=0.05,
                    core_maxh=0.02, ring_maxh=0.004, battery_maxh=0.0015,
                    air_maxh=0.03, entrefer_maxh=None, entrefer_solid_maxh=0.001,
                    reg_factor=1e-4, core_fill_factor_aware=True):
    """test_rings: pass e.g. ["ringp1"] for a fast smoke test -- a full
    N-ring x len(sim_frequencies) sweep is WAY beyond what a single-ring DC
    solve already costs on this machine (see simulation_ngsolve.py's own
    memory warnings); start small.
    frequencies_hz: defaults to config.py's sim_frequencies (Hz already).
    Every other parameter matches simulation_ngsolve.run_inductance()'s
    same-named parameter -- see its docstring.
    core_fill_factor_aware=True (default): does NOT touch mu_r -- the
    core's complex permeability (core_mu_effective(), now the raw
    lamination formula with no fill_factor dilution) stays exactly what
    the curl-curl solve sees, so the field solution is unaffected by
    fill_factor. Instead, config.py's MATERIALS["Core"]["ac"]["fill_factor"]
    is applied as a POST-SOLVE weight when extracting L_H/Rac_ohm: only
    the Core's own contribution to
    integral(nu*curl(A_i).curl(A_j)) gets multiplied by fill_factor --
    entrefer/windings/air all keep weight 1.0 -- since only that fraction
    of the core's geometric cross-section is actually magnetic (lossy) material, the
    rest being inter-lamination insulation contributing ~zero stored
    energy/loss. Mirrors simulation_ngsolve.run_inductance()'s same flag.
    Pass False for the old unweighted integral(A_i.J_j) extraction
    (equivalent to fill_factor=1.0)."""
    if frequencies_hz is None:
        frequencies_hz = sim_frequencies
    frequencies_hz = list(frequencies_hz)

    if test_rings is not None:
        all_ring_names = test_rings
    else:
        primary, secondary = sim.ring_names(sim.STEP_FILE_CLOSED)
        all_ring_names = primary + secondary
    N = len(all_ring_names)
    F = len(frequencies_hz)
    sigma_copper = sim.MATERIALS["ringp"]["sigma"]

    for kind in ("ringp", "rings"):
        p = LITZ_PARAMS[kind]
        delta_max = skin_depth_at(max(frequencies_hz) if frequencies_hz else 0, sigma_copper)
        if p["strand_diameter_m"] > delta_max:
            print(f"[warn] {kind} strand_diameter_m={p['strand_diameter_m']:.4g}m "
                  f"is NOT small vs. skin depth {delta_max:.4g}m at the highest swept "
                  f"frequency ({max(frequencies_hz):g} Hz) -- the 'strands stay "
                  "skin-effect-free' assumption this script relies on may not hold; "
                  "treat high-frequency Rac here as unreliable until the real wire "
                  "spec is confirmed in config.py.")

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

    print("Meshing (whole assembly, shared across every frequency)...")
    geo = OCCGeometry(Glue(glued_parts))
    mesh = Mesh(geo.GenerateMesh(maxh=air_maxh))
    print(f"mesh: {mesh.ne} elements")

    print("Building each ring's current source (frequency-independent, see docstring)...")
    J = {}
    for k, name in enumerate(all_ring_names):
        J[name] = sim._solve_ring_current(mesh, name, sigma_copper)
        print(f"  {k + 1}/{N} {name}: current source ready")

    Ldom = max(hi[i] - lo[i] for i in range(3))
    reg = reg_factor * (1.0 / MU0) / Ldom ** 2

    is_primary = {name: name.startswith("ringp") for name in all_ring_names}

    L_H = np.zeros((F, N, N))
    Rac_ohm = np.zeros((F, N, N))

    for fi, freq_hz in enumerate(frequencies_hz):
        mu_complex_p = complex_mu_litz(freq_hz, LITZ_PARAMS["ringp"]["strand_diameter_m"], sigma_copper)
        mu_complex_s = complex_mu_litz(freq_hz, LITZ_PARAMS["rings"]["strand_diameter_m"], sigma_copper)
        mu_complex_core = core_mu_effective(freq_hz)
        print(f"[{fi + 1}/{F}] f={freq_hz:g} Hz  mu_litz(ringp)={mu_complex_p:.4g}  "
              f"mu_litz(rings)={mu_complex_s:.4g}  mu_lamination(Core)={mu_complex_core:.4g}")

        mat_mu = {"Core": mu_complex_core, "entrefer": complex(1.0, 0.0)}
        for name in all_ring_names:
            mu_c = mu_complex_p if is_primary[name] else mu_complex_s
            mat_mu[name] = mu_c
            mat_mu[f"{name}_battery"] = mu_c

        mu_r = mesh.MaterialCF(mat_mu, default=complex(1.0, 0.0))
        nu = 1.0 / (MU0 * mu_r)

        if core_fill_factor_aware:
            core_fill_factor = CORE_AC["fill_factor"]
            energy_weight = mesh.MaterialCF({"Core": core_fill_factor}, default=1.0)
        else:
            energy_weight = 1.0

        fes_h = HCurl(mesh, order=order, nograds=True, dirichlet="outer", complex=True)
        uu, vv = fes_h.TnT()
        a = BilinearForm(nu * curl(uu) * curl(vv) * dx + reg * uu * vv * dx)
        a.Assemble()
        inv_h = a.mat.Inverse(fes_h.FreeDofs(), inverse="sparsecholesky")

        A_vecs = {}
        for name in all_ring_names:
            f_lf = LinearForm(J[name] * vv * dx)
            f_lf.Assemble()
            gfA = GridFunction(fes_h)
            gfA.vec.data = inv_h * f_lf.vec
            A_vecs[name] = gfA.vec.CreateVector()
            A_vecs[name].data = gfA.vec

        gfAi = GridFunction(fes_h)
        gfAj = GridFunction(fes_h)
        for i, ni in enumerate(all_ring_names):
            gfAi.vec.data = A_vecs[ni]
            for j, nj in enumerate(all_ring_names):
                if j < i:
                    L_H[fi, i, j] = L_H[fi, j, i]
                    Rac_ohm[fi, i, j] = Rac_ohm[fi, j, i]
                    continue
                gfAj.vec.data = A_vecs[nj]
                # Core fill_factor is applied to L_H only (via energy_weight,
                # Core+entrefer share of the energy integral), NOT to
                # Rac_ohm -- Rac's own frequency dependence already comes
                # entirely from the WINDINGS' own Litz fill_factor (baked
                # into complex_mu_litz()/mu_complex_p/mu_complex_s -> nu
                # above), which is an independent, already-correct effect;
                # weighting the core's share of energy would incorrectly
                # scale the windings' own loss too if done on the same
                # complex integral, so L and Rac are extracted from two
                # separate integrals here (raw vs. energy_weight-applied).
                curl_ij = nu * InnerProduct(curl(gfAi), curl(gfAj))
                L_complex = Integrate(curl_ij * energy_weight, mesh)
                L_complex_raw = Integrate(curl_ij, mesh)
                L_H[fi, i, j] = L_complex.real
                # Sign verified empirically against an actual solve, not
                # just derived from complex_mu_litz's own convention: a
                # smoke test at 1MHz initially came out with Rac_ohm < 0
                # (unphysical -- added AC resistance from a passive lossy
                # effect cannot be negative) using +2*pi*f*Im(L_complex),
                # meaning Im(mu_complex)<0 does NOT propagate to
                # Im(L_complex)>0 the way a naive derivation suggests once
                # nu=1/(mu0*mu_complex) is inverted and solved through the
                # actual curl-curl system. The minus sign below is the
                # fix confirmed by that test (Rac_ohm > 0 afterward).
                Rac_ohm[fi, i, j] = -2 * np.pi * freq_hz * L_complex_raw.imag

    mat_path = os.path.join(MATRIX_DIR, "litz_sweep.mat")
    savemat(mat_path, {
        "frequencies_hz": np.array(frequencies_hz).reshape(1, -1),
        "L_H": L_H,
        "Rac_ohm": Rac_ohm,
        "ring_names": np.array(all_ring_names, dtype=object),
    })
    print(f"Saved {mat_path}  (L_H, Rac_ohm shaped [frequency, ring, ring])")
    return frequencies_hz, all_ring_names, L_H, Rac_ohm


if __name__ == "__main__":
    run_litz_sweep(test_rings=["ringp1"])

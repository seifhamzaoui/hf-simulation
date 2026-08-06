"""
NGSolve equivalent of simulation.py / simulation_Maxwell.py -- computes the
same three matrices, entirely with the open-source NGSolve FEM stack, no
external commercial software (no AEDT/Q3D/Maxwell) involved.

Status (validated by direct comparison against the Q3D reference data in
"traited Values/", see development notes -- these are NOT re-derived here,
just summarized):

  1. Electrostatics  -> capacitance matrix. VALIDATED: ringp1-ringp2 mutual
                        capacitance came out at 90.1 pF on a reduced 3-body
                        test (Core+ringp1+ringp2, air only, no insulation
                        yet), squarely between Q3D's 75.9 pF and Maxwell's
                        174.7 pF for the same pair.
  2. DC Conduction   -> DC resistance of each ring. VALIDATED: ringp1 came
                        out at 70.64 micro-ohm vs Q3D's 70.77 micro-ohm
                        (<0.2% difference).
  3. Self/mutual inductance -> a genuine bounded-domain curl-curl field
                        solve (mu_r=20000 assigned directly on the Core's
                        mesh region, exactly like permittivity is for
                        capacitance) -- MESH-CONVERGED (core.maxh=0.02/0.01/
                        0.006 all agree to <1%) but still ~2.1-2.3x Q3D's
                        absolute scale for a lone ring; see run_inductance()'s
                        docstring for the full story, including why earlier
                        open-ring current-source attempts diverged and what
                        fixed it (closed rings + an internal EMF "battery"
                        instead of exposed terminal faces). Treat coupling
                        STRUCTURE as more trustworthy than absolute values
                        until that residual ~2x gap is root-caused. Does not
                        produce an AC resistance matrix (Resis.mat) -- that
                        needs a separate frequency-dependent skin/proximity-
                        effect calculation.

Geometry: capacitance and DC resistance load STEP_FILE
(transformer_model.step, same file the Q3D/Maxwell scripts use), which has
each ring as an OPEN conductor with a real terminal-face cut -- both stages
need that cut to define their two-terminal (hi/gnd) boundary conditions.
Inductance loads STEP_FILE_CLOSED (transformer_model_closed.step) instead,
where every ring is a fully CLOSED torus -- required by the EMF-battery
current-source technique described in run_inductance()'s docstring. Both
files carry the same solid names (ringp1..17, rings1..43, Core,
pinsulator_0, sinsulator_0, primary_secondary_insulation -- 64 solids,
uniquely named) via Netgen's OpenCascade (OCC) import. Both are in
millimeters; the whole shape is scaled by 1e-3 once at load time (see
load_solids()) so every subsequent length, area, and volume used in the
physics is genuine SI (meters) -- mixing mm lengths with SI (per-meter)
material constants was an early bug here that silently produced results
1000x too large.

Matrices are saved into "./ngsolve matrices" in the same
{"f_<freq>kHz": matrix, "frequencies_hz": [[..]]} layout used by the
Q3D-based simulation.py in traited Values/.
"""

import os
import re

import numpy as np
from scipy.io import savemat

from netgen.occ import OCCGeometry, Box, Pnt, Glue
from ngsolve import (
    Mesh, H1, HCurl, GridFunction, BilinearForm, LinearForm,
    grad, curl, dx, Integrate, InnerProduct, CoefficientFunction, x, y, z,
)

from config import *

STEP_FILE = "transformer_model.step"  # open rings (real terminal-face cut) -- capacitance, DC resistance
STEP_FILE_CLOSED = "transformer_model_closed.step"  # closed rings (no cut) -- inductance, see run_inductance()
MATRIX_DIR = "./ngsolve matrices"

os.makedirs(MATRIX_DIR, exist_ok=True)

EPS0 = 8.8541878128e-12
MU0 = 4e-7 * np.pi

# Physical properties for each named material now live in config.py's
# MATERIALS dict (edit values there -- shared with the AEDT-based scripts).
# CORE_AC is captured BEFORE the reduction below strips every non-"ngsolve"
# sub-dict -- run_inductance()'s core_fill_factor_aware needs "ac"'s
# "fill_factor", which "ngsolve" alone doesn't carry (mirrors
# simulation_ngsolve_litz.py's own CORE_AC = CONFIG_MATERIALS["Core"]["ac"]).
CORE_AC = MATERIALS["Core"]["ac"]
# Here we only need the "ngsolve" numeric sub-dict per material.
MATERIALS = {name: props["ngsolve"] for name, props in MATERIALS.items()}


# ============================================================
# Geometry
# ============================================================

def load_solids(step_file=STEP_FILE):
    """Returns {name: Solid} for every named solid in the STEP file, scaled
    from the file's native millimeters to SI meters. Defaults to STEP_FILE
    (open rings, real terminal-face cut) -- pass STEP_FILE_CLOSED explicitly
    for the inductance stage, which needs fully closed ring tori (see
    run_inductance()).

    Solids that share a name (e.g. "entrefer" -- one per core column, both
    named identically in transformer_geometry_rectangular.py) are summed
    into a single compound shape under that key -- a plain {s.name: s}
    comprehension would silently keep only the LAST one and drop the rest,
    which for entrefer meant one column's gap was quietly left untagged
    (mesh region defaulted to "air" -- numerically inert only because
    entrefer's material currently equals air's, sigma=0/eps_r=1/mu_r=1)."""
    raw_shape = OCCGeometry(step_file).shape
    shape = raw_shape.Scale(Pnt(0, 0, 0), 1e-3)
    grouped = {}
    for s in shape.solids:
        grouped.setdefault(s.name, []).append(s)
    return {name: (parts[0] if len(parts) == 1 else sum(parts)) for name, parts in grouped.items()}


def entrefer_solid_from(solids):
    """Unions every solid whose name contains 'entrefer' (case-insensitive)
    into one shape, regardless of the exact naming scheme used in
    transformer_geometry_rectangular.py. Needed because that script has
    named the gap solids differently across edits -- a single shared
    'entrefer' (merged by load_solids' own dict-collision fix), or
    per-column names like 'entrefer10'/'entrefer20' -- and a lookup keyed
    on the exact string "entrefer" silently returns None (falls back to
    untagged "air") the moment the naming scheme changes. Matches
    config.py's own MATERIALS["entrefer"]["pattern"] = "entrefer.*"."""
    parts = [s for name, s in solids.items() if "entrefer" in name.lower()]
    if not parts:
        return None
    return parts[0] if len(parts) == 1 else sum(parts)


def core_solid_from(solids):
    """Unions every solid named 'Core' or 'Core<N>' (a numbered suffix,
    e.g. 'Core1'/'Core2') into one shape. Needed because cutting the core
    into disconnected pieces (e.g. around the entrefer gaps) and naming
    each fragment with its own 'Core<N>' name -- so Glue() can't silently
    drop the name off some fragments the way it did when every piece
    shared the single literal name 'Core' (see ring_names()'s own
    docstring for that exact failure mode) -- means a plain
    solids["Core"] lookup no longer finds a single entry once the
    geometry script is fixed to name fragments this way. Mirrors
    entrefer_solid_from()'s same pattern-matching approach; .mat("Core")
    on the returned (possibly multi-solid) shape still tags every
    fragment with the one mesh region name "Core" regardless of each
    fragment's own original OCC solid name."""
    parts = [s for name, s in solids.items() if name is not None and re.match(r"^Core\d*$", name)]
    if not parts:
        return None
    return parts[0] if len(parts) == 1 else sum(parts)


def face_area_bounds(width, thickness, circular):
    if not circular:
        nominal = width * thickness
    else:
        nominal = (thickness / 2) ** 2 * 3.14
    return nominal * 0.9, nominal * 1.1


# config.py's conductor dimensions are in mm -- convert once, to match the
# meters-scaled geometry from load_solids()
P_MIN_AREA, P_MAX_AREA = face_area_bounds(p_cond_width * 1e-3, p_cond_thickness * 1e-3, p_conduc_circular)
S_MIN_AREA, S_MAX_AREA = face_area_bounds(s_cond_width * 1e-3, s_cond_thickness * 1e-3, p_conduc_circular)


def _natural_sort_key(s):
    """Split a string into text/number chunks so 'ringp10' sorts after 'ringp9', not after 'ringp1'."""
    return [int(chunk) if chunk.isdigit() else chunk.lower() for chunk in re.split(r"(\d+)", s)]


def ring_names(step_file=STEP_FILE):
    """Reads the actual ring solid names present in step_file's geometry
    (ringp*/rings*), rather than reconstructing an expected list from
    config.py's primary_n_turns_total/secondary_n_turns_total -- the STEP
    file already names every solid, so this can't silently drift out of
    sync with config.py if the file isn't regenerated after an edit.
    Skips unnamed (None) solids -- transformer_geometry.py's/
    transformer_geometry_rectangular.py's Glue() step doesn't always
    propagate a name to every resulting boundary fragment (seen in
    practice: 2 unnamed solids alongside 2 'Core'-named ones in a closed
    STEP file), and an unnamed solid is by definition not a ringp*/rings*
    ring -- crashing on n.startswith(...) for it would reject the whole
    file over solids this function was never looking for anyway."""
    solids = load_solids(step_file)
    primary = sorted((n for n in solids if n is not None and n.startswith("ringp")), key=_natural_sort_key)
    secondary = sorted((n for n in solids if n is not None and n.startswith("rings")), key=_natural_sort_key)
    return primary, secondary


def area_bounds_for(ring_name):
    return (P_MIN_AREA, P_MAX_AREA) if ring_name.startswith("ringp") else (S_MIN_AREA, S_MAX_AREA)


def refine_entrefer_faces(core, maxh=None, area_ratio_range=(0.9, 1.1)):
    """Locally refines the Core solid's air-gap (entrefer) faces instead of
    relying on core.maxh (0.02, i.e. ~50x the typical <1mm entrefer) to
    resolve them. A gapped magnetic circuit's reluctance -- and therefore
    its inductance -- is dominated by the gap, so under-resolving it here
    plausibly explains far more of the NGSolve-vs-analytical mismatch than
    the domain padding or HCurl order do.

    The gap cut creates two paper-thin horizontal faces (bounding-box
    Y-extent ~ 0) whose area matches a single column's cross-section
    (col_width * col_depth) -- everything else on Core that's also
    horizontal (yoke top/bottom, window ledges) has a distinctly
    different area, so a tight area match reliably isolates just the
    gap-facing faces regardless of column/entrefer count. Verified
    directly against transformer_model_closed.step: the true gap faces
    come out at exactly col_width*col_depth with 0% error, while the
    nearest false-positive candidate (the window ledge) is ~45% off --
    confirming the default 90%-110% area_ratio_range cleanly separates
    them (same nominal*(lo, hi) convention as face_area_bounds() above).

    CAUTION -- face.maxh sets the element size across the face's ENTIRE
    area, not just through the gap's thin normal direction: a first
    attempt at entrefer/4 (~0.1mm) demanded a ~600x600 element grid on
    each ~4550mm^2 gap face, which never finished meshing (11+ minutes,
    pegged at ~2.9GB and climbing before being killed). The default here
    (2mm) is deliberately the same order as this file's other local mesh
    sizes (rings: 4mm, batteries: 1.5mm) -- a ~10x improvement over the
    20mm core default at the one place that matters most, without
    blowing up element count over the whole face."""
    if entrefer == 0:
        return []
    nominal_area = col_width * col_depth * 1e-6  # config values are mm; core is scaled to meters
    gap_h = maxh if maxh is not None else 0.002
    lo, hi = nominal_area * area_ratio_range[0], nominal_area * area_ratio_range[1]
    gap_faces = [
        f for f in core.faces
        if abs(f.bounding_box[1][1] - f.bounding_box[0][1]) < 1e-5
        and lo <= f.mass <= hi
    ]
    for f in gap_faces:
        f.maxh = gap_h
    print(f"Refined {len(gap_faces)} entrefer-facing core face(s) to maxh={gap_h * 1e3:.4f}mm")
    return gap_faces


def terminal_faces(solid):
    """Same area-matching logic as the Q3D/Maxwell scripts: picks the ring's
    two end-cap faces (source/sink) by matching the expected conductor
    cross-section area -- the true end-cap faces are the only ones that
    size (all lateral faces are much larger or much smaller)."""
    min_area, max_area = area_bounds_for(solid.name)
    faces = [f for f in solid.faces if min_area <= f.mass <= max_area]
    print(f"{solid.name}: {len(faces)} candidate terminal face(s) in area range")
    return faces


# ============================================================
# Save helper -- same {"f_<freq>kHz": matrix, "frequencies_hz": [[..]]}
# layout the Q3D-based simulation.py uses in traited Values/
# ============================================================

def freq_label(freq_khz):
    return f"{freq_khz:.6g}kHz"


def save_matrix_like_q3d(matrix, mat_name, freq_khz=min_freq_kHz, display_scale=1.0):
    """`display_scale` matches Q3D's own (inconsistent) per-quantity default
    report unit: traited Values/cap_data.mat stores bare picofarads (not SI
    Farads), traited Values/DCR.mat stores plain SI ohms, and induc.mat
    stores bare nanohenries (not SI Henries) -- pass 1e12/1/1e9 respectively
    so the numbers in the saved file are directly comparable, not just the
    same physical quantity in a different, silently-mismatched unit.
    """
    key = f"f_{freq_label(freq_khz)}"
    mat_dict = {key: np.abs(matrix) * display_scale, "frequencies_hz": np.array([[freq_khz * 1e3]])}
    mat_path = os.path.join(MATRIX_DIR, mat_name)
    savemat(mat_path, mat_dict)
    print(f"Saved {mat_path} with key '{key}' (values x{display_scale:g} vs SI base unit)")


# ============================================================
# 1. CAPACITANCE -- electrostatics
#    Diagonal (after reduction) = ring-to-core, off-diagonal = ring-to-ring.
#    Core is solved as an ordinary signal conductor (like every ring) and
#    then folded into the diagonal -- exactly the same reduction
#    simulation.py's extract_matrix_dict(apply_reduction=True) applies to
#    the Q3D matrix, and simulation_Maxwell.py applies by hand for the same
#    reason (Maxwell's own Matrix boundary has no built-in "ground" role
#    either when Core is passed in as a plain signal source).
#
#    Capacitance is extracted via the energy/reciprocity identity
#    C_ij = integral(eps * grad(u_i) . grad(u_j)) = u_i . (A u_j), computed
#    as a sparse matrix-vector product against the assembled stiffness
#    matrix A rather than N^2 Integrate() calls -- validated equivalent on
#    small cases, but far faster at N=61 conductors over a 500k+ element
#    mesh.
# ============================================================

def run_capacitance():
    print("Loading geometry...")
    solids = load_solids()
    primary, secondary = ring_names()
    all_ring_names = primary + secondary
    conductors = ["Core"] + all_ring_names  # Core first -> index 0, needed for the reduction below

    # Core may be split across several disconnected 'Core<N>'-named
    # fragments (see core_solid_from()'s docstring) -- merge them into
    # ONE solid retagged uniformly "Core" so it's still exactly ONE entry
    # in `conductors` / one row-column in the reduced capacitance matrix,
    # matching this function's own "solved as a single plain conductor"
    # physical assumption (the fragments are all the same physical core,
    # just artificially split by the entrefer cut -- not separate
    # conductors). The other solids keep their own individual names, one
    # mesh region (and one matrix row/column) each, same as before.
    core = core_solid_from(solids)
    core.mat("Core")
    core.faces.name = "Core"
    core.maxh = 0.004
    other_solids = {name: s for name, s in solids.items() if not (name is not None and re.match(r"^Core\d*$", name))}
    all_solids = [core] + list(other_solids.values())
    for s in other_solids.values():
        s.mat(s.name)
        if s.name in conductors:
            s.faces.name = s.name
            s.maxh = 0.004
        else:
            s.maxh = 0.015

    bb = OCCGeometry(STEP_FILE).shape.bounding_box
    bb = ([v * 1e-3 for v in bb[0]], [v * 1e-3 for v in bb[1]])
    pad = 0.03
    box = Box(
        Pnt(bb[0][0] - pad, bb[0][1] - pad, bb[0][2] - pad),
        Pnt(bb[1][0] + pad, bb[1][1] + pad, bb[1][2] + pad),
    )
    box.mat("air")
    box.maxh = 0.03

    print("Meshing (whole assembly, ~2 minutes)...")
    geo = OCCGeometry(Glue([box] + all_solids))
    mesh = Mesh(geo.GenerateMesh(maxh=0.03))
    print(f"mesh: {mesh.ne} elements")

    epsilon = mesh.MaterialCF(
        {
            "ringp.*": MATERIALS["ringp"]["eps_r"],
            "rings.*": MATERIALS["rings"]["eps_r"],
            "Core": MATERIALS["ringp"]["eps_r"],  # solved as a plain copper conductor net for this stage
            "pinsulator.*": MATERIALS["pinsulator"]["eps_r"],
            "sinsulator.*": MATERIALS["sinsulator"]["eps_r"],
            "primary_secondary_insulation": MATERIALS["primary_secondary_insulation"]["eps_r"],
            "p_layer insulator": MATERIALS["p_layer insulator"]["eps_r"],
            "s_layer insulator": MATERIALS["s_layer insulator"]["eps_r"],
        },
        default=MATERIALS["air"]["eps_r"],
    ) * EPS0

    dirichlet_pattern = "|".join(conductors)
    fes = H1(mesh, order=1, dirichlet=dirichlet_pattern)
    print(f"ndof = {fes.ndof}")

    u, v = fes.TnT()
    a = BilinearForm(epsilon * grad(u) * grad(v) * dx)
    a.Assemble()
    print("Factoring (reused for every conductor)...")
    inv = a.mat.Inverse(fes.FreeDofs(), inverse="sparsecholesky")

    input("Geometry/materials/mesh ready for the capacitance solve -- press Enter to continue...")

    N = len(conductors)
    vecs = []
    for k, name in enumerate(conductors):
        gfu = GridFunction(fes)
        gfu.vec[:] = 0.0
        gfu.Set(1.0, definedon=mesh.Boundaries(name))
        res = (-a.mat * gfu.vec).Evaluate()
        gfu.vec.data += inv * res
        vecs.append(gfu.vec.CreateVector())
        vecs[-1].data = gfu.vec
        print(f"  solved conductor {k + 1}/{N}: {name}")

    print("Assembling capacitance matrix...")
    Avecs = [(a.mat * vecs[j]).Evaluate() for j in range(N)]
    C_raw = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            C_raw[i, j] = InnerProduct(vecs[i], Avecs[j])

    first_row = C_raw[0, 1:]
    reduced = C_raw[1:, 1:].copy()
    np.fill_diagonal(reduced, first_row)
    C = np.abs(reduced)

    save_matrix_like_q3d(C, "cap_data.mat", display_scale=1e12)  # F -> pF, matching Q3D's stored convention


# ============================================================
# 2. DC RESISTANCE OF EACH RING -- DC conduction
#    Each ring is solved completely independently (no coupling: they have no
#    galvanic connection to each other), so unlike capacitance this does NOT
#    need the full 64-solid mesh -- each ring gets its own small, fast mesh.
#    Resistance is extracted via the energy identity R = V^2 / P, with
#    P = integral(sigma * |grad(V)|^2) over the ring only, again avoiding any
#    boundary flux integral (which NGSolve can't evaluate directly for a
#    domain-wise/piecewise-defined coefficient function anyway).
# ============================================================

def run_dc_resistance(litz_aware=None):
    """litz_aware=None (default): reads config.py's DC_RESISTANCE_LITZ_AWARE
    (itself defaulting to False -- solid-copper assumption, sigma_copper
    across the full geometric cross-section). False/solid-copper is the
    configuration VALIDATED against Q3D at the top of this file's docstring
    (ringp1: 70.64 vs Q3D's 70.77 micro-ohm, <0.2% difference). Q3D's own
    reference model predates any Litz-wire modeling in this project and
    almost certainly assumed solid copper too, so flipping the config
    default to True would silently break that documented match.
    litz_aware=True (explicitly, or via config): divides sigma_copper by
    config.py's MATERIALS[kind]["litz"]["fill_factor"] before solving, so
    DC resistance reflects the ACTUAL reduced copper cross-section of a
    stranded Litz bundle instead of assuming the geometric cross-section is
    100% copper -- use this if the real winding is genuinely Litz wire and
    you want Rdc consistent with simulation_ngsolve_litz.py /
    simulation_ngsolve_litz_ratio.py's own fill_factor-aware Rdc calc, at
    the cost of no longer matching the solid-copper Q3D comparison above.
    Pass litz_aware explicitly to override config.py for one call without
    editing it."""
    import config as _config
    if litz_aware is None:
        litz_aware = getattr(_config, "DC_RESISTANCE_LITZ_AWARE", False)

    primary, secondary = ring_names()
    all_ring_names = primary + secondary
    N = len(all_ring_names)
    R = np.zeros((N, N))

    input("Ready to compute each ring's DC resistance independently -- press Enter to continue...")

    sigma_copper = MATERIALS["ringp"]["sigma"]  # same copper conductivity as "rings"
    for k, name in enumerate(all_ring_names):
        kind = "ringp" if name.startswith("ringp") else "rings"
        sigma_eff = sigma_copper
        if litz_aware:
            sigma_eff = sigma_copper * _config.MATERIALS[kind]["litz"]["fill_factor"]

        solids = load_solids()
        ring = solids[name]
        min_area, max_area = area_bounds_for(name)
        all_faces = list(ring.faces)
        tf = [f for f in all_faces if min_area <= f.mass <= max_area]
        for f in all_faces:
            f.name = f"{name}_body"
        tf[0].name = f"{name}_hi"
        tf[1].name = f"{name}_gnd"
        ring.mat(name)
        ring.maxh = 0.003

        geo = OCCGeometry(ring)
        mesh = Mesh(geo.GenerateMesh(maxh=0.003))

        fes = H1(mesh, order=2, dirichlet=f"{name}_hi|{name}_gnd")
        u, v = fes.TnT()
        a = BilinearForm(sigma_eff * grad(u) * grad(v) * dx)
        a.Assemble()
        inv = a.mat.Inverse(fes.FreeDofs(), inverse="sparsecholesky")

        gfu = GridFunction(fes)
        gfu.vec[:] = 0.0
        gfu.Set(1.0, definedon=mesh.Boundaries(f"{name}_hi"))
        res = (-a.mat * gfu.vec).Evaluate()
        gfu.vec.data += inv * res

        P = Integrate(sigma_eff * grad(gfu) * grad(gfu), mesh)
        R[k, k] = 1.0 / P
        print(f"  {k + 1}/{N} {name}: R = {R[k, k]*1e6:.4f} micro-ohm ({mesh.ne} elements)")

    save_matrix_like_q3d(R, "DCR.mat")


# ============================================================
# 3. SELF (PROPER) AND MUTUAL INDUCTANCES -- a genuine curl-curl field solve
#    *** Mesh-converged (checked at core.maxh=0.02/0.01/0.006, all agree to
#    <1%), but still ~2.1-2.3x Q3D's absolute values for a lone ring -- see
#    the note at the end of this docstring before trusting absolute scale.
#    Coupling STRUCTURE (which pairs are more/less coupled) should be more
#    reliable than that, as with every earlier approach tried here. ***
#
#    Earlier attempts (documented in prior revisions of this file / dev
#    history) tried to impress a current density J confined to an OPEN
#    ring (two exposed terminal faces, like the real conductor's own cut,
#    bridged or not, excluded or not) directly as the RHS of
#    curl(nu*curl(A)) = J over the whole Core+Ring+Air domain. Every variant
#    of that diverged the same way: as the regularization needed to remove
#    HCurl's null space shrinks, the answer explodes instead of converging,
#    across factor changes of many orders of magnitude. The common cause,
#    confirmed by testing several distinct gap/exclusion geometries (a
#    hand-built cylinder, a CAD-precise cut) with either a natural or a
#    Dirichlet boundary condition on the exposed faces: J has non-zero net
#    flux crossing the conductor boundary at those two faces (current
#    "enters"/"exits" the modeled conductor there), which is not compatible
#    with the curl-curl operator regardless of how that termination is
#    shaped or bounded.
#
#    The fix: STEP_FILE_CLOSED gives every ring as a genuinely CLOSED torus
#    (no cut, no exposed terminal faces anywhere). Current is driven by a
#    small internal "battery" -- a distributed EMF term confined to a thin
#    (~2mm) sub-region of the ring's OWN material (_ring_battery_tool),
#    solving
#
#        integral(sigma*grad(u).grad(v)) = integral_battery(sigma*E_imp.grad(v))
#
#    as a pure-Neumann problem (no Dirichlet anywhere -- testing with v=1
#    makes the RHS vanish identically, so the compatibility condition holds
#    by construction, not approximately). J = sigma*(-grad(u) + E_imp) is
#    then divergence-free EVERYWHERE, including inside the battery itself --
#    there is no face anywhere where current enters or exits the conductor.
#    This is what actually fixed the divergence: regularization factors
#    1e-4..1e-6 now agree to <1% (a genuine converged plateau), instead of
#    blowing up by 5-8 orders of magnitude between adjacent factors like
#    every open-ring variant did.
#
#    Validated this way on a standalone ring-only mesh: self-leakage came
#    out at 150.5nH vs 160.7nH from the old open-ring/Dirichlet-cut PEEC
#    method at the same bin count (<7% apart) -- confirms the technique
#    reproduces the same physics for the leakage-only (no core) case.
#
#    Structure mirrors run_capacitance(): ONE shared mesh (Core + every
#    ring, each split into a passive part + its own battery sub-region +
#    Air), the HCurl curl-curl system assembled and factored ONCE, then
#    reused as N independent right-hand-side solves (one per ring's own
#    current source) -- self/mutual inductance is then the same energy/
#    reciprocity identity used everywhere else in this file:
#    L_ij = integral(A_i . J_j), A_i/J_i from ring i's own unit-current
#    solve, no boundary flux integral needed.
#
#    *** Absolute-scale caveat: for a single ring (ringp1) against the real
#    ferrite core (mu_r=20000), this reproducibly gives ~2.1-2.3x Q3D's
#    5169nH self-inductance, confirmed STABLE across core mesh resolutions
#    from 20mm down to 6mm (i.e. not a discretization error) and using two
#    different but geometrically identical closed-ring STEP files. The
#    residual gap's actual source (domain truncation size, HCurl order
#    sensitivity -- order=2 shifted the answer by a further ~9% without
#    closing the gap -- or something else) has NOT yet been root-caused.
#    Treat this stage's absolute values with that in mind. ***
#
#    This does NOT produce an AC resistance matrix (Resis.mat) -- that needs
#    a frequency-dependent skin/proximity-effect calculation, a different
#    physics problem entirely.
# ============================================================

E_IMP_DIR = CoefficientFunction((0.0, 0.0, 1.0))
E_IMP_MAG = 1.0
BATTERY_HALF_THICKNESS = 0.001  # 2mm total -- also fixes _solve_ring_current's EMF normalization


def _ring_battery_tool(ring):
    """A small Box tool that carves a thin (~2mm) EMF "battery" slab out of
    a ring's own material, at its local x-minimum edge, spanning its full
    y-thickness and centered on its own z-midpoint. This is the same
    relative feature for every ring regardless of stacking position --
    confirmed against the actual CAD: every ringp/rings solid shares its
    neighbors' x/z range exactly, only y (the stacking axis) shifts, and
    the core leg's own z-center (-32.5mm) is identical for primary and
    secondary alike."""
    bb = ring.bounding_box
    zc = (bb[0][2] + bb[1][2]) / 2
    return Box(
        Pnt(bb[0][0] - 0.005, bb[0][1] - 0.005, zc - BATTERY_HALF_THICKNESS),
        Pnt(bb[0][0] + 0.010, bb[1][1] + 0.005, zc + BATTERY_HALF_THICKNESS),
    )


def _solve_ring_current(mesh, name, sigma_copper):
    """EMF-driven steady-conduction solve for one CLOSED ring, restricted to
    its own material (`name` + `name_battery`). No Dirichlet boundary
    anywhere -- see the section docstring above for why that's what makes
    this compatible with the later curl-curl solve. Returns J normalized to
    exactly 1A total current, zero outside this ring's own domain."""
    ring_mats = f"{name}|{name}_battery"
    fes = H1(mesh, order=2, definedon=mesh.Materials(ring_mats))
    u, v = fes.TnT()
    eps_reg = 1e-6 * sigma_copper / (0.01 ** 2)  # pins the pure-Neumann problem's additive constant
    a = BilinearForm(sigma_copper * grad(u) * grad(v) * dx + eps_reg * u * v * dx)
    a.Assemble()
    battery_mask = mesh.MaterialCF({f"{name}_battery": 1.0}, default=0.0)
    Eimp = battery_mask * E_IMP_DIR * E_IMP_MAG
    f = LinearForm(sigma_copper * Eimp * grad(v) * dx)
    f.Assemble()
    inv = a.mat.Inverse(fes.FreeDofs(), inverse="sparsecholesky")
    gfU = GridFunction(fes)
    gfU.vec.data = inv * f.vec
    Jraw = sigma_copper * (-grad(gfU) + Eimp)
    P = Integrate(Jraw * (-grad(gfU) + Eimp), mesh, definedon=mesh.Materials(ring_mats))
    EMF = E_IMP_MAG * (2 * BATTERY_HALF_THICKNESS)
    I_computed = P / EMF
    domain_mask = mesh.MaterialCF({name: 1.0, f"{name}_battery": 1.0}, default=0.0)
    return domain_mask * (Jraw / I_computed)


def run_inductance(test_rings=None, order=1, pad=0.05,
                    core_maxh=0.02, ring_maxh=0.004, battery_maxh=0.0015, air_maxh=0.03,
                    entrefer_maxh=None, entrefer_solid_maxh=0.001, reg_factor=1e-4,
                    core_fill_factor_aware=True):
    """test_rings: pass e.g. ["ringp1"] to solve just that ring's self-
    inductance instead of the full N-ring matrix -- a fast smoke test for
    mesh/solve changes (like refine_entrefer_faces above) without paying
    for the full assembly.
    order: HCurl polynomial order -- order=2 was noted (pre-entrefer-fix) to
    shift the answer by ~9%. WARNING: order=2 at the default mesh sizes
    below needs ~586k DOF for a single ring -- confirmed to exhaust 16GB RAM
    during sparse factorization. Coarsen core_maxh/ring_maxh/battery_maxh/
    air_maxh together with order=2 to keep DOF count within what this
    machine can actually factor.
    pad: air-region padding (meters) beyond the assembly's own bounding box
    before the "outer" Dirichlet (A=0) boundary -- too small artificially
    pins down the leakage field.
    entrefer_solid_maxh: element size (meters) for the entrefer solid itself,
    independent of entrefer_maxh (which only refines Core's gap-facing
    faces). Default 0.001 (1mm) -- confirmed on a single-ring smoke test to
    mesh fine (367k elements, 432k HCurl ndof, ~67s) but that's ~15x the
    element count of the 2mm default for just one ring; a full 60-ring run
    multiplies from there and risks the same RAM wall that killed the
    order=2 attempt, so watch memory on the first full run at this value.
    core_fill_factor_aware=True (default): does NOT touch mu_r -- the
    core's real, linear permeability stays the raw config.py value used in
    the actual curl-curl solve, so the field solution itself is unaffected
    by fill_factor. Instead, the lamination stacking factor (config.py's
    MATERIALS["Core"]["ac"]["fill_factor"]) is applied as a POST-SOLVE
    weight when extracting L: the Core's AND entrefer's contribution to
    integral(nu*curl(A_i).curl(A_j)) (the stored-energy form of the
    inductance integral) both get multiplied by fill_factor -- windings
    and air keep weight 1.0 -- reflecting that only a fill_factor
    fraction of the core's (and its gap's) geometric cross-section is
    actually magnetic material (the rest is inter-lamination insulation
    contributing ~zero stored energy), without perturbing how flux
    actually distributes through the solved field. Mirrors
    run_inductance_gpu()'s same flag in simulation_ngsolve_cuda.py. Pass
    False for the old unweighted integral(A_i.J_j) extraction (equivalent
    to fill_factor=1.0)."""
    print("*** Mesh-converged but still ~2.1-2.3x Q3D's absolute scale for a lone")
    print("*** ring against the real core -- see this function's docstring before")
    print("*** trusting absolute values; coupling structure should be more reliable.")

    if test_rings is not None:
        all_ring_names = test_rings
    else:
        primary, secondary = ring_names(STEP_FILE_CLOSED)
        all_ring_names = primary + secondary
    N = len(all_ring_names)
    sigma_copper = MATERIALS["ringp"]["sigma"]  # same copper conductivity as "rings"
    mu_r_core = MATERIALS["Core"]["mu_r"]  # raw, undiluted -- fill_factor applied to the energy integral below instead

    print("Loading closed-ring geometry...")
    solids = load_solids(STEP_FILE_CLOSED)
    core = core_solid_from(solids)
    core.mat("Core")
    core.maxh = core_maxh
    refine_entrefer_faces(core, maxh=entrefer_maxh)

    entrefer_solid = entrefer_solid_from(solids)
    if entrefer_solid is not None:
        entrefer_solid.mat("entrefer")
        entrefer_solid.maxh = entrefer_solid_maxh

    ring_parts = []
    for name in all_ring_names:
        ring = solids[name]
        battery = ring * _ring_battery_tool(ring)
        ring_passive = ring - battery
        ring_passive.mat(name)
        ring_passive.maxh = ring_maxh
        battery.mat(f"{name}_battery")
        battery.maxh = battery_maxh
        ring_parts.append(ring_passive)
        ring_parts.append(battery)

    bb = OCCGeometry(STEP_FILE_CLOSED).shape.bounding_box
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

    input("Geometry ready (Core + all closed rings/batteries + Air + Entrefer) -- press Enter to mesh...")

    print("Meshing (whole assembly, ~2 minutes)...")
    geo = OCCGeometry(Glue(glued_parts))
    mesh = Mesh(geo.GenerateMesh(maxh=air_maxh))
    print(f"mesh: {mesh.ne} elements")

    print("Building each ring's current source (independent EMF-driven conduction solves)...")
    J = {}
    for k, name in enumerate(all_ring_names):
        J[name] = _solve_ring_current(mesh, name, sigma_copper)
        print(f"  {k + 1}/{N} {name}: current source ready")

    mu_r = mesh.MaterialCF({"Core": mu_r_core, "entrefer": MATERIALS["entrefer"]["mu_r"]}, default=1.0)
    nu = 1.0 / (MU0 * mu_r)
    Ldom = max(hi[i] - lo[i] for i in range(3))
    reg = reg_factor * (1.0 / MU0) / Ldom ** 2
    print(f"reg_factor={reg_factor:g}  ->  reg={reg:.4g}  (nu_core={1.0/(MU0*mu_r_core):.4g})")

    if core_fill_factor_aware:
        core_fill_factor = CORE_AC["fill_factor"]
        energy_weight = mesh.MaterialCF({"Core": core_fill_factor, "entrefer": core_fill_factor}, default=1.0)
    else:
        energy_weight = 1.0

    fes_h = HCurl(mesh, order=order, nograds=True, dirichlet="outer")
    print(f"HCurl ndof = {fes_h.ndof}")
    uu, vv = fes_h.TnT()
    a = BilinearForm(nu * curl(uu) * curl(vv) * dx + reg * uu * vv * dx)
    print("Factoring (reused for every ring's right-hand side)...")
    a.Assemble()
    inv_h = a.mat.Inverse(fes_h.FreeDofs(), inverse="sparsecholesky")

    print("Solving each ring's own field (reusing the same factorization)...")
    A_vecs = {}
    for k, name in enumerate(all_ring_names):
        f = LinearForm(J[name] * vv * dx)
        f.Assemble()
        gfA = GridFunction(fes_h)
        gfA.vec.data = inv_h * f.vec
        A_vecs[name] = gfA.vec.CreateVector()
        A_vecs[name].data = gfA.vec
        print(f"  {k + 1}/{N} {name}: field solved")

    print("Assembling inductance matrix (L_ij = integral(nu*curl(A_i).curl(A_j)*energy_weight), "
          "fill_factor-weighted in Core+entrefer)...")
    L = np.zeros((N, N))
    gfAi = GridFunction(fes_h)
    gfAj = GridFunction(fes_h)
    for i, ni in enumerate(all_ring_names):
        gfAi.vec.data = A_vecs[ni]
        for j, nj in enumerate(all_ring_names):
            if j < i:
                L[i, j] = L[j, i]
                continue
            gfAj.vec.data = A_vecs[nj]
            L[i, j] = Integrate(nu * InnerProduct(curl(gfAi), curl(gfAj)) * energy_weight, mesh)

    save_matrix_like_q3d(L, "induc.mat", display_scale=1e9)  # H -> nH, matching Q3D's stored convention
    print("*** Reminder: absolute values are ~2.1-2.3x Q3D for a lone ring -- see docstring. ***")


# ============================================================
# Choose which simulation(s) to run
# ============================================================

STAGES = {
    "1": ("Capacitance (Electrostatics)", run_capacitance),
    "2": ("DC Resistance (DC Conduction)", run_dc_resistance),
    "3": ("Inductance (curl-curl field solve, closed rings) -- see docstring", run_inductance),
    "3t": ("Inductance -- QUICK TEST (ringp1 self-inductance only)", lambda: run_inductance(test_rings=["ringp1"])),
}

if __name__ == "__main__":
    print("Which simulation(s) do you want to run?")
    for key, (label, _) in STAGES.items():
        print(f"  {key}) {label}")

    choice = input("Enter numbers separated by commas (e.g. 1,2), or 'all' [all]: ").strip().lower()

    # "3t" (single-ring smoke test) is opt-in only -- excluded from "all" since
    # it's redundant with "3" (the real run), not an additional stage to run.
    tokens = [k for k in STAGES if k != "3t"] if choice in ("", "all", "*") else [c.strip() for c in choice.split(",")]
    selected_keys = [t for t in tokens if t in STAGES]
    invalid = [t for t in tokens if t not in STAGES]
    if invalid:
        print(f"[warn] ignoring unrecognized selection(s): {invalid}")
    if not selected_keys:
        raise SystemExit("No valid simulation selected, exiting.")

    print("Running:", ", ".join(STAGES[k][0] for k in selected_keys))
    for key in selected_keys:
        STAGES[key][1]()

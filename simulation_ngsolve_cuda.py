"""
GPU-accelerated variant of simulation_ngsolve.py.

Does NOT modify simulation_ngsolve.py -- reuses its geometry/material/
helper functions (load_solids, ring_names, area_bounds_for, MATERIALS,
MU0/EPS0, save_matrix_like_q3d, STEP_FILE/STEP_FILE_CLOSED,
_ring_battery_tool, _solve_ring_current) via direct import, and only swaps
out the expensive "assemble once, solve many right-hand-sides" step (the
pattern in run_capacitance() and run_inductance()) for a GPU-accelerated
solve using cupy, instead of NGSolve's own CPU sparsecholesky Inverse().

*** MUST BE RUN WITH A SEPARATE, PLAIN-ASCII-PATH PYTHON ENVIRONMENT ***
Root-caused during development: cupy's CUDA JIT compiler (NVRTC) fails to
resolve its own bundled header include path (cupy/complex.cuh) when cupy
is installed under a path containing a non-ASCII character -- confirmed by
reproducing the exact same failure in a fresh venv under this project's
own path (which contains "Thése"), and confirming the identical code
works from a venv under a plain-ASCII path. This is an NVRTC/Windows
path-handling issue, not a bug in this code -- it reproduces on bare `cupy
.array([1.0]) * 2`, before any of this file's own logic runs.

A working environment was set up at C:\\ngsolve_cuda_venv\\venv (plain
ASCII path), with ngsolve + cupy-cuda12x + scipy installed, plus the
nvidia-{cublas,cusparse,cusolver,curand,cuda_runtime,cuda_nvrtc,nvjitlink}
-cu12 companion packages (cupy's own wheel does not bundle those -- they
are separate pip packages cupy dynamically loads at runtime). Run this
file with THAT venv's interpreter, e.g. from this project's directory:

    C:\\ngsolve_cuda_venv\\venv\\Scripts\\python.exe simulation_ngsolve_cuda.py

This script inserts the project directory onto sys.path itself, so plain
Python imports (config.py, simulation_ngsolve.py) work fine regardless of
which venv/interpreter runs it -- only cupy's own CUDA kernel compilation
was ever affected by the accented path, not ordinary Python module imports.

GPU solve method (gpu_solver_for): extracts the assembled NGSolve sparse
matrix via its own .CSR() export, restricts it to FreeDofs (mirroring what
Inverse(freedofs=...) does internally), moves it to the GPU ONCE as a
cupyx.scipy.sparse matrix with a Jacobi (diagonal) preconditioner, then
reuses that GPU-resident matrix for every right-hand-side via
preconditioned CG (cupyx.scipy.sparse.linalg.cg) -- the GPU analogue of
factoring once on CPU and reusing Inverse() for many solves. Valid for SPD
systems only -- true of every system in this codebase (electrostatics, DC
conduction, and the regularized curl-curl inductance solve are all SPD by
construction); an indefinite/non-symmetric system would need a different
Krylov method (GMRES/BiCGStab) instead of CG.

Where this actually helps: capacitance's 61-conductor solve and
inductance's 60-ring solve both already reuse ONE CPU factorization for
many right-hand sides, which is already fast at the mesh sizes seen so
far (tens of thousands to ~600k DOFs -- seconds to factor, near-instant
back-substitution per RHS). GPU offload's real payoff shows up as these
grow larger (finer meshes, more conductors) or with a much higher RHS
count. At current scale, benchmark this against the CPU path rather than
assuming it wins -- CG convergence for the mu_r=20000 curl-curl system in
particular may need many iterations without a stronger preconditioner
than plain Jacobi.
"""
import re
import sys
import time

import numpy as np

sys.path.insert(0, r"C:\Users\hp\OneDrive - CEFEM INDUSTRIES\These\simulations\simulation HF TMF")
import simulation_ngsolve as sim  # noqa: E402 -- must follow the sys.path insert above

from netgen.occ import OCCGeometry, Box, Pnt, Glue  # noqa: E402
from ngsolve import (  # noqa: E402
    Mesh, H1, HCurl, GridFunction, BilinearForm, LinearForm,
    grad, curl, dx, Integrate, InnerProduct, CoefficientFunction, x, y, z,
    BND, specialcf,
)

import scipy.sparse as sp  # noqa: E402
import cupy as cp  # noqa: E402
import cupyx.scipy.sparse as cpsp  # noqa: E402
import cupyx.scipy.sparse.linalg as cpspla  # noqa: E402


# ============================================================
# GPU multi-right-hand-side solver
# ============================================================

def gpu_solver_for(a_mat, freedofs, rtol=1e-10, maxiter=5000, label=""):
    """Builds a GPU-resident, FreeDofs-restricted view of an assembled
    NGSolve sparse matrix ONCE (with a Jacobi/diagonal preconditioner), and
    returns a callable solve(res_vec) that GPU-solves for any number of
    right-hand-side NGSolve vectors afterward, reusing that same GPU
    matrix every time -- see the module docstring for why this mirrors
    a.mat.Inverse(freedofs, "sparsecholesky") but on the GPU via
    preconditioned CG instead of a direct factorization."""
    t0 = time.time()
    vals, cols, rowptr = a_mat.CSR()
    vals = np.array(vals)
    cols = np.array(cols)
    rowptr = np.array(rowptr)
    A_full = sp.csr_matrix((vals, cols, rowptr), shape=a_mat.shape)

    mask = np.array([freedofs[i] for i in range(len(freedofs))], dtype=bool)
    idx_free = np.nonzero(mask)[0]
    A_free = A_full[idx_free, :][:, idx_free].tocsr()

    diag = np.asarray(A_free.diagonal())
    jacobi_inv = cp.asarray(1.0 / diag)
    A_gpu = cpsp.csr_matrix(A_free)
    n_free = len(idx_free)
    print(f"  [gpu{(' ' + label) if label else ''}] matrix on GPU: {n_free} free dofs, "
          f"{A_gpu.nnz} nnz ({time.time()-t0:.1f}s)")

    def M_precond(x):
        return jacobi_inv * x

    M_op = cpspla.LinearOperator(A_gpu.shape, matvec=M_precond)

    def solve(res_vec):
        b_np = res_vec.FV().NumPy()
        b_free_gpu = cp.asarray(b_np[idx_free])
        x_free_gpu, info = cpspla.cg(A_gpu, b_free_gpu, rtol=rtol, maxiter=maxiter, M=M_op)
        if info != 0:
            print(f"    [gpu warn] CG did not fully converge (info={info})")
        x_full = np.zeros(len(mask))
        x_full[idx_free] = cp.asnumpy(x_free_gpu)
        return x_full

    return solve


# ============================================================
# 1. CAPACITANCE -- same geometry/assembly as simulation_ngsolve.run_capacitance(),
#    GPU solve instead of CPU sparsecholesky Inverse().
# ============================================================

def run_capacitance_gpu():
    print("Loading geometry...")
    solids = sim.load_solids()
    primary, secondary = sim.ring_names()
    all_ring_names = primary + secondary
    conductors = ["Core"] + all_ring_names

    # Core may be split across several disconnected 'Core<N>'-named
    # fragments (see sim.core_solid_from()'s docstring) -- merge them
    # into ONE solid retagged uniformly "Core" so it's still exactly ONE
    # conductor, matching this function's own "solved as a single plain
    # conductor" physical assumption. Mirrors simulation_ngsolve.py's own
    # run_capacitance() fix.
    core = sim.core_solid_from(solids)
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

    bb = OCCGeometry(sim.STEP_FILE).shape.bounding_box
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
            "ringp.*": sim.MATERIALS["ringp"]["eps_r"],
            "rings.*": sim.MATERIALS["rings"]["eps_r"],
            "Core": sim.MATERIALS["ringp"]["eps_r"],  # solved as a plain copper conductor net for this stage
            "pinsulator.*": sim.MATERIALS["pinsulator"]["eps_r"],
            "sinsulator.*": sim.MATERIALS["sinsulator"]["eps_r"],
            "primary_secondary_insulation": sim.MATERIALS["primary_secondary_insulation"]["eps_r"],
            "p_layer insulator": sim.MATERIALS["p_layer insulator"]["eps_r"],
            "s_layer insulator": sim.MATERIALS["s_layer insulator"]["eps_r"],
        },
        default=sim.MATERIALS["air"]["eps_r"],
    ) * sim.EPS0

    dirichlet_pattern = "|".join(conductors)
    fes = H1(mesh, order=1, dirichlet=dirichlet_pattern)
    print(f"ndof = {fes.ndof}")

    u, v = fes.TnT()
    a = BilinearForm(epsilon * grad(u) * grad(v) * dx)
    a.Assemble()
    print("Building GPU solver (reused for every conductor)...")
    solve = gpu_solver_for(a.mat, fes.FreeDofs(), label="capacitance")

    N = len(conductors)
    vecs = []
    for k, name in enumerate(conductors):
        gfu = GridFunction(fes)
        gfu.vec[:] = 0.0
        gfu.Set(1.0, definedon=mesh.Boundaries(name))
        res = (-a.mat * gfu.vec).Evaluate()
        gfu.vec.FV().NumPy()[:] += solve(res)
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

    sim.save_matrix_like_q3d(C, "cap_data.mat", display_scale=1e12)


# ============================================================
# 2. DC RESISTANCE -- each ring is an independent, small, single-RHS
#    system (no shared factorization to amortize across many solves), so
#    there is little to gain from GPU offload here -- included for
#    completeness, but expect this to be no faster than (likely slower
#    than) the CPU version at these problem sizes.
# ============================================================

def run_dc_resistance_gpu(litz_aware=None):
    """litz_aware=None (default): reads config.py's DC_RESISTANCE_LITZ_AWARE,
    mirroring simulation_ngsolve.run_dc_resistance()'s own litz_aware
    parameter -- see that function's docstring for the full rationale
    (False/solid-copper is what was validated against Q3D; True divides
    sigma_copper by config.py's MATERIALS[kind]["litz"]["fill_factor"]).
    This GPU version previously had NO litz_aware handling at all (always
    used raw sigma_copper) -- that was a real gap, not an intentional
    solid-copper default like the CPU version's."""
    import config as _config
    if litz_aware is None:
        litz_aware = getattr(_config, "DC_RESISTANCE_LITZ_AWARE", False)

    primary, secondary = sim.ring_names()
    all_ring_names = primary + secondary
    N = len(all_ring_names)
    R = np.zeros((N, N))

    sigma_copper = sim.MATERIALS["ringp"]["sigma"]  # same copper conductivity as "rings"
    for k, name in enumerate(all_ring_names):
        kind = "ringp" if name.startswith("ringp") else "rings"
        sigma_eff = sigma_copper
        if litz_aware:
            sigma_eff = sigma_copper * _config.MATERIALS[kind]["litz"]["fill_factor"]

        solids = sim.load_solids()
        ring = solids[name]
        min_area, max_area = sim.area_bounds_for(name)
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
        solve = gpu_solver_for(a.mat, fes.FreeDofs(), label=name)

        gfu = GridFunction(fes)
        gfu.vec[:] = 0.0
        gfu.Set(1.0, definedon=mesh.Boundaries(f"{name}_hi"))
        res = (-a.mat * gfu.vec).Evaluate()
        gfu.vec.FV().NumPy()[:] += solve(res)

        P = Integrate(sigma_eff * grad(gfu) * grad(gfu), mesh)
        R[k, k] = 1.0 / P
        print(f"  {k + 1}/{N} {name}: R = {R[k, k]*1e6:.4f} micro-ohm ({mesh.ne} elements)")

    sim.save_matrix_like_q3d(R, "DCR.mat")


# ============================================================
# 2b. PEEC (Partial Element Equivalent Circuit) SELF-INDUCTANCE --
#     free-space Neumann double-volume-integral, GPU pairwise sum. See
#     run_inductance_peec_gpu()'s docstring for scope/caveats.
# ============================================================

def run_inductance_peec_gpu(ring_name="ringp1"):
    """Free-space PEEC self-inductance via the classical Neumann formula:

        L = (mu0 / (4*pi*I^2)) * integral_V integral_V (J(r).J(r')/|r-r'|) dV dV'

    evaluated as a cell-collocation sum over ring_name's OWN standalone
    conduction-solve mesh (no core, no air -- PEEC only needs the
    current-carrying volume): element centroids and volumes come from
    NGSolve's own element_wise Integrate (exact per-cell average, not a
    single evaluation point), current density is sim._solve_ring_current()
    -- the SAME EMF-battery technique run_inductance_gpu() uses, already
    normalized to exactly 1A -- so this is a genuinely independent
    cross-check of that current source (different math: a real-space double
    integral, not a vector-potential curl-curl PDE) rather than a
    re-derivation of the same solve.

    The exact self-term (i==j, distance 0) is excluded from the sum --
    standard practice for a weakly (integrable, ~1/r) singular 3D kernel:
    the true self-cell contribution scales as O(h^5) per cell, and the
    error from dropping it vanishes as O(N^-2/3) as the mesh refines (never
    exactly zero, but negligible at any reasonable mesh density).

    *** Scope caveat -- this deliberately has NO ferrite core: a free-space
    Green's function has no way to represent permeable material at all (mu_r
    implicitly = 1 everywhere). This is exactly what Q3D's baseline PEEC
    engine computes internally BEFORE its own magnetic-material correction
    (see run_inductance_gpu's docstring) -- so this number is NOT directly
    comparable to Q3D's full ringp1 figure (5169nH, core-corrected) or to
    run_inductance_gpu's curl-curl-with-core result (~11000nH, non-converged
    as of the last GPU run -- see that function's caveats). What this DOES
    validate: the EMF-battery current source technique itself, against the
    ~150-160nH leakage-only (no-core) figure already cross-checked in
    simulation_ngsolve.run_inductance()'s docstring, via a completely
    independent numerical method. ***

    O(N^2) pairwise sum runs on GPU (cupy) -- the natural fit (trivially
    parallel), and tractable here because N (cells in ONE conductor's own
    standalone mesh, no air/core) stays in the thousands, not the full
    assembly's hundreds of thousands."""
    sigma_copper = sim.MATERIALS["ringp"]["sigma"]  # same copper conductivity as "rings"

    print(f"Loading standalone {ring_name} geometry (no core, no air -- free-space PEEC)...")
    solids = sim.load_solids(sim.STEP_FILE_CLOSED)
    ring = solids[ring_name]
    battery = ring * sim._ring_battery_tool(ring)
    ring_passive = ring - battery
    ring_passive.mat(ring_name)
    ring_passive.maxh = 0.004
    battery.mat(f"{ring_name}_battery")
    battery.maxh = 0.0015

    geo = OCCGeometry(Glue([ring_passive, battery]))
    mesh = Mesh(geo.GenerateMesh(maxh=0.004))
    print(f"mesh: {mesh.ne} elements")

    Jcf = sim._solve_ring_current(mesh, ring_name, sigma_copper)

    vol = Integrate(CoefficientFunction(1.0), mesh, element_wise=True).NumPy()
    cx = Integrate(x, mesh, element_wise=True).NumPy() / vol
    cy = Integrate(y, mesh, element_wise=True).NumPy() / vol
    cz = Integrate(z, mesh, element_wise=True).NumPy() / vol
    Jx = Integrate(Jcf[0], mesh, element_wise=True).NumPy() / vol
    Jy = Integrate(Jcf[1], mesh, element_wise=True).NumPy() / vol
    Jz = Integrate(Jcf[2], mesh, element_wise=True).NumPy() / vol
    N = len(vol)
    print(f"{N} cells -> {N * N:,} pairwise terms (GPU)")

    pts = cp.asarray(np.stack([cx, cy, cz], axis=1))   # (N,3)
    Jvec = cp.asarray(np.stack([Jx, Jy, Jz], axis=1))  # (N,3)
    w = cp.asarray(vol)                                # (N,)

    total = 0.0
    chunk = 2000
    for i0 in range(0, N, chunk):
        i1 = min(i0 + chunk, N)
        diff = pts[i0:i1, None, :] - pts[None, :, :]           # (c,N,3)
        dist = cp.linalg.norm(diff, axis=2)                    # (c,N)
        inv_dist = cp.where(dist > 0, 1.0 / dist, 0.0)         # excludes exact self-term (i==j)
        dot = Jvec[i0:i1] @ Jvec.T                             # (c,N)
        total += float(cp.sum(w[i0:i1, None] * w[None, :] * dot * inv_dist))
        print(f"  pairwise sum: {i1}/{N} rows done")

    L = sim.MU0 / (4 * np.pi) * total  # I = 1A already (Jcf is unit-current-normalized)
    print(f"PEEC self-inductance ({ring_name}, free space, no core): {L * 1e9:.4f} nH")
    return L


# ============================================================
# 2c. PEEC WITH CORE -- extends run_inductance_peec_gpu() with the ferrite
#     core's contribution via the classical equivalent magnetic surface
#     charge (single-layer potential) boundary-element method. See
#     run_inductance_peec_core_gpu()'s docstring for the full derivation.
# ============================================================

def run_inductance_peec_core_gpu(ring_name="ringp1", core_maxh=0.01):
    """Adds the ferrite core's contribution to the free-space PEEC self-
    inductance computed by run_inductance_peec_gpu(), via the classical
    "equivalent magnetic surface charge" boundary-element method for a
    linear, uniform-permeability body in an external field -- the standard
    way a magnetic material gets folded into an otherwise free-space
    (Green's-function-based) magnetostatic calculation, since the plain
    1/(4*pi*r) kernel used for the free-space self-inductance term is only
    valid in a homogeneous medium and has no way to represent mu_r on its
    own (the same fundamental limitation flagged for Q3D's baseline PEEC
    engine).

    DERIVATION (standard result for the magnetic-charge/single-layer-
    potential transmission problem -- directly analogous to the
    electrostatic "dielectric body in an external field" problem with mu_r
    substituted for eps_r):

      Write H = H0 + H_sigma, where H0 is the field of the real (ring)
      current ALONE (as if the whole domain were vacuum -- a plain
      Biot-Savart evaluation), and H_sigma is the field of an unknown
      equivalent magnetic surface charge density sigma_m living on the
      Core's own outer boundary S. All the "bound charge" for a UNIFORM-mu_r
      body concentrates on its surface, not its volume: div(M) = 0 inside a
      uniform, source-free region, since M = (mu_r-1)*H = -(mu_r-1)*grad(psi)
      there and curl(grad(psi)) = 0 -- so a pure single-layer (surface-only)
      representation is exact, not an approximation, for this geometry.
      (This also means the Core's air gap is handled automatically, with no
      special-casing: the gap-facing surfaces are just as much a part of S
      as the rest of the core's boundary.)

      Matching the standard magnetic transmission boundary conditions
      (continuous tangential H, continuous normal B) via a magnetic scalar
      potential psi = psi0 + psi_s (psi_s the single-layer potential of
      sigma_m) gives the Fredholm 2nd-kind integral equation:

          sigma_m(r) = k * [H0_n(r) + H_sigma_avg_n(r)],   k = 2*(mu_r-1)/(mu_r+1)

      where H_sigma_avg_n is the PRINCIPAL VALUE (self-term excluded) of the
      induced field's own normal component at the surface -- the self-term
      is handled analytically via the standard single-layer-potential jump
      relation (a clean, well-known closed form), not via numerical
      singular-kernel quadrature like the free-space volume self-inductance
      term needed.

      Built-in sanity check: mu_r=1 gives k=0, so sigma_m=0 identically and
      the core's contribution vanishes exactly -- i.e. this must reduce to
      plain run_inductance_peec_gpu() with no magnetic contrast, as physics
      requires.

      Once sigma_m is solved for (a dense N_surface x N_surface linear
      system), its TANGENTIAL magnetization M_tang = (mu_r-1)*H_tang gives
      an equivalent BOUND SURFACE CURRENT K_b = M_tang x n_hat (the bound
      VOLUME current is exactly zero, same reasoning as above). K_b produces
      an additional vector potential A_M via the same free-space
      1/(4*pi*r) kernel (equivalent sources always radiate in free space --
      that is the entire point of the equivalent-source method), and the
      core's contribution to the ring's self-inductance follows from the
      SAME reciprocity identity used everywhere else in this file:
      Delta_L = (1/I^2) * integral_ring(A_M . J) dV.

    *** Caveat: mu_r=20000 is exactly the regime where this class of
    equivalent-source method is least accurate -- it is the same kind of
    approximation Q3D's own magnetic-material PEEC correction relies on
    internally. Treat this as a second, independent-but-approximate
    estimate for comparison, not as ground truth. ***
    """
    sigma_copper = sim.MATERIALS["ringp"]["sigma"]  # same copper conductivity as "rings"
    mu_r = sim.MATERIALS["Core"]["mu_r"]

    # ---- Ring: identical to run_inductance_peec_gpu() ----
    print(f"Loading standalone {ring_name} geometry...")
    solids = sim.load_solids(sim.STEP_FILE_CLOSED)
    ring = solids[ring_name]
    battery = ring * sim._ring_battery_tool(ring)
    ring_passive = ring - battery
    ring_passive.mat(ring_name)
    ring_passive.maxh = 0.004
    battery.mat(f"{ring_name}_battery")
    battery.maxh = 0.0015

    ring_geo = OCCGeometry(Glue([ring_passive, battery]))
    ring_mesh = Mesh(ring_geo.GenerateMesh(maxh=0.004))
    print(f"ring mesh: {ring_mesh.ne} elements")

    Jcf = sim._solve_ring_current(ring_mesh, ring_name, sigma_copper)
    r_vol = Integrate(CoefficientFunction(1.0), ring_mesh, element_wise=True).NumPy()
    r_cx = Integrate(x, ring_mesh, element_wise=True).NumPy() / r_vol
    r_cy = Integrate(y, ring_mesh, element_wise=True).NumPy() / r_vol
    r_cz = Integrate(z, ring_mesh, element_wise=True).NumPy() / r_vol
    r_Jx = Integrate(Jcf[0], ring_mesh, element_wise=True).NumPy() / r_vol
    r_Jy = Integrate(Jcf[1], ring_mesh, element_wise=True).NumPy() / r_vol
    r_Jz = Integrate(Jcf[2], ring_mesh, element_wise=True).NumPy() / r_vol
    Nr = len(r_vol)

    r_pts = cp.asarray(np.stack([r_cx, r_cy, r_cz], axis=1))  # (Nr,3)
    r_J = cp.asarray(np.stack([r_Jx, r_Jy, r_Jz], axis=1))    # (Nr,3)
    r_w = cp.asarray(r_vol)                                   # (Nr,)

    # ---- Free-space term (same as run_inductance_peec_gpu, for reference) ----
    diff = r_pts[:, None, :] - r_pts[None, :, :]
    dist = cp.linalg.norm(diff, axis=2)
    inv_dist = cp.where(dist > 0, 1.0 / dist, 0.0)
    dot = r_J @ r_J.T
    L_freespace = float(sim.MU0 / (4 * np.pi) * cp.sum(r_w[:, None] * r_w[None, :] * dot * inv_dist))
    print(f"free-space term: {L_freespace * 1e9:.4f} nH")

    # ---- Core: standalone surface mesh (no ring, no air needed for a BEM) ----
    # The entrefer-facing cells need the SAME local refinement as the curl-curl
    # mesh (see refine_entrefer_faces's docstring), for a different but related
    # reason here: two gap-facing cells sit ~0.4mm apart, but at the bulk
    # core_maxh (10mm) each cell is 25x larger than that separation -- the
    # 1/dist^3 BEM kernel between such a badly-oversized, badly-separated cell
    # pair explodes (confirmed: without this refinement the influence matrix's
    # eigenvalues reach +/-28, when a well-posed double-layer operator on a
    # closed surface should stay well under 1 in magnitude -- an unmistakable
    # sign of an under-resolved near-field interaction, not a formula error).
    print("Loading standalone Core geometry (surface mesh for BEM)...")
    core = sim.core_solid_from(sim.load_solids(sim.STEP_FILE_CLOSED))
    core.mat("Core")
    core.maxh = core_maxh
    sim.refine_entrefer_faces(core)
    core_geo = OCCGeometry(core)
    core_mesh = Mesh(core_geo.GenerateMesh(maxh=core_maxh))

    c_area = Integrate(CoefficientFunction(1.0), core_mesh, BND, element_wise=True).NumPy()
    c_cx = Integrate(x, core_mesh, BND, element_wise=True).NumPy() / c_area
    c_cy = Integrate(y, core_mesh, BND, element_wise=True).NumPy() / c_area
    c_cz = Integrate(z, core_mesh, BND, element_wise=True).NumPy() / c_area
    n_cf = specialcf.normal(3)
    c_nx = Integrate(n_cf[0], core_mesh, BND, element_wise=True).NumPy() / c_area
    c_ny = Integrate(n_cf[1], core_mesh, BND, element_wise=True).NumPy() / c_area
    c_nz = Integrate(n_cf[2], core_mesh, BND, element_wise=True).NumPy() / c_area
    Nc = len(c_area)
    print(f"Core surface mesh: {Nc} boundary cells")

    c_pts = cp.asarray(np.stack([c_cx, c_cy, c_cz], axis=1))   # (Nc,3)
    c_n = cp.asarray(np.stack([c_nx, c_ny, c_nz], axis=1))     # (Nc,3)
    c_area_gpu = cp.asarray(c_area)                            # (Nc,)

    # ---- H0 (Biot-Savart, full vector) from the ring's current at each Core surface cell ----
    # H0(r) = (1/4pi) * sum_k [J_k x (r - r_k)] * vol_k / |r - r_k|^3
    diff_cr = c_pts[:, None, :] - r_pts[None, :, :]              # (Nc,Nr,3)
    dist_cr = cp.linalg.norm(diff_cr, axis=2)                    # (Nc,Nr)
    inv_dist3_cr = cp.where(dist_cr > 0, 1.0 / dist_cr**3, 0.0)
    cross_cr = cp.cross(r_J[None, :, :], diff_cr)                # (Nc,Nr,3): J_k x (r_i - r_k)
    H0 = (1.0 / (4 * np.pi)) * cp.sum(
        cross_cr * (r_w[None, :] * inv_dist3_cr)[:, :, None], axis=1
    )  # (Nc,3)
    H0_n = cp.sum(H0 * c_n, axis=1)  # (Nc,)

    # ---- BEM matrix: A[i,j] = area_j * (1/4pi) * n_i.(r_i-r_j)/|r_i-r_j|^3, zero diagonal ----
    diff_cc = c_pts[:, None, :] - c_pts[None, :, :]              # (Nc,Nc,3)
    dist_cc = cp.linalg.norm(diff_cc, axis=2)
    inv_dist3_cc = cp.where(dist_cc > 0, 1.0 / dist_cc**3, 0.0)
    ndotdiff = cp.sum(c_n[:, None, :] * diff_cc, axis=2)         # (Nc,Nc): n_i . (r_i - r_j)
    A_mat = (1.0 / (4 * np.pi)) * ndotdiff * inv_dist3_cc * c_area_gpu[None, :]
    cp.fill_diagonal(A_mat, 0.0)

    k = 2.0 * (mu_r - 1.0) / (mu_r + 1.0)
    lhs = cp.eye(Nc) - k * A_mat
    rhs = k * H0_n
    print(f"Solving {Nc}x{Nc} dense BEM system for the equivalent surface charge...")
    sigma_m = cp.linalg.solve(lhs, rhs)  # (Nc,)

    # ---- Full H_sigma_avg (vector) at each Core cell, for the tangential field ----
    H_sigma_avg = (1.0 / (4 * np.pi)) * cp.sum(
        diff_cc * (sigma_m[None, :] * c_area_gpu[None, :] * inv_dist3_cc)[:, :, None], axis=1
    )  # (Nc,3)
    H_full_avg = H0 + H_sigma_avg
    H_full_avg_n = cp.sum(H_full_avg * c_n, axis=1)
    H_tang = H_full_avg - H_full_avg_n[:, None] * c_n
    M_tang = (mu_r - 1.0) * H_tang
    K_b = cp.cross(M_tang, c_n)  # (Nc,3) equivalent bound surface current

    # ---- A_M at each ring cell from K_b (free-space vector-potential kernel), reciprocity ----
    diff_rc = r_pts[:, None, :] - c_pts[None, :, :]              # (Nr,Nc,3)
    dist_rc = cp.linalg.norm(diff_rc, axis=2)
    inv_dist_rc = cp.where(dist_rc > 0, 1.0 / dist_rc, 0.0)
    A_M = (sim.MU0 / (4 * np.pi)) * cp.sum(
        K_b[None, :, :] * (c_area_gpu[None, :] * inv_dist_rc)[:, :, None], axis=1
    )  # (Nr,3)

    delta_L = float(cp.sum(cp.sum(A_M * r_J, axis=1) * r_w))  # I = 1A already
    L_total = L_freespace + delta_L

    print(f"core contribution: {delta_L * 1e9:.4f} nH")
    print(f"PEEC self-inductance ({ring_name}, WITH core, mu_r={mu_r:g}): {L_total * 1e9:.4f} nH")
    return L_total


# ============================================================
# 3. INDUCTANCE -- same closed-ring/EMF-battery geometry and current
#    sources as simulation_ngsolve.run_inductance() (reused directly via
#    sim._ring_battery_tool / sim._solve_ring_current), GPU solve for the
#    shared HCurl system instead of CPU sparsecholesky Inverse().
# ============================================================

def run_inductance_gpu(test_rings=None, entrefer_maxh=None, entrefer_solid_maxh=0.001,
                        core_fill_factor_aware=True):
    """test_rings: pass e.g. ["ringp1"] to solve just that ring's self-
    inductance instead of the full N-ring matrix -- a fast smoke test for
    mesh/solve changes (like sim.refine_entrefer_faces below) without
    paying for the full assembly.
    entrefer_maxh: override for the gap-facing CORE faces' element size
    (meters). WARNING: this refines the ENTIRE ~4550mm^2 gap-facing face
    area, not just near the gap -- confirmed that maxh=0.0001 (0.1mm) hangs
    for 11+ minutes climbing past 3GB before ever finishing meshing. Values
    much below the default 0.002 (2mm) risk the same runaway.
    entrefer_solid_maxh: element size (meters) for the entrefer solid
    itself, independent of entrefer_maxh above. Default 0.001 (1mm) --
    see run_inductance's docstring in simulation_ngsolve.py for the
    single-ring smoke-test numbers (367k elements, 432k HCurl ndof,
    ~67s) and the RAM-risk warning for a full multi-ring run.
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
    run_inductance()'s same flag in simulation_ngsolve.py. Pass False for
    the old unweighted integral(A_i.J_j) extraction (equivalent to
    fill_factor=1.0)."""
    if test_rings is not None:
        all_ring_names = test_rings
    else:
        primary, secondary = sim.ring_names(sim.STEP_FILE_CLOSED)
        all_ring_names = primary + secondary
    N = len(all_ring_names)
    sigma_copper = sim.MATERIALS["ringp"]["sigma"]  # same copper conductivity as "rings"
    mu_r_core = sim.MATERIALS["Core"]["mu_r"]  # raw, undiluted -- fill_factor applied to the energy integral below instead

    print("Loading closed-ring geometry...")
    solids = sim.load_solids(sim.STEP_FILE_CLOSED)
    core = sim.core_solid_from(solids)
    core.mat("Core")
    core.maxh = 0.02
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
        ring_passive.maxh = 0.004
        battery.mat(f"{name}_battery")
        battery.maxh = 0.0015
        ring_parts.append(ring_passive)
        ring_parts.append(battery)

    bb = OCCGeometry(sim.STEP_FILE_CLOSED).shape.bounding_box
    bb = ([v * 1e-3 for v in bb[0]], [v * 1e-3 for v in bb[1]])
    pad = 0.05
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
    air.maxh = 0.03

    glued_parts = [air, core] + ring_parts
    if entrefer_solid is not None:
        glued_parts.append(entrefer_solid)

    print("Meshing (whole assembly, ~2 minutes)...")
    geo = OCCGeometry(Glue(glued_parts))
    mesh = Mesh(geo.GenerateMesh(maxh=0.03))
    print(f"mesh: {mesh.ne} elements")

    print("Building each ring's current source (independent EMF-driven conduction solves)...")
    J = {}
    for k, name in enumerate(all_ring_names):
        J[name] = sim._solve_ring_current(mesh, name, sigma_copper)
        print(f"  {k + 1}/{N} {name}: current source ready")

    mu_r = mesh.MaterialCF({"Core": mu_r_core, "entrefer": sim.MATERIALS["entrefer"]["mu_r"]}, default=1.0)
    nu = 1.0 / (sim.MU0 * mu_r)
    Ldom = max(hi[i] - lo[i] for i in range(3))
    reg_factor = 1e-4
    reg = reg_factor * (1.0 / sim.MU0) / Ldom ** 2

    if core_fill_factor_aware:
        core_fill_factor = sim.CORE_AC["fill_factor"]
        energy_weight = mesh.MaterialCF({"Core": core_fill_factor, "entrefer": core_fill_factor}, default=1.0)
    else:
        energy_weight = 1.0

    fes_h = HCurl(mesh, order=1, nograds=True, dirichlet="outer")
    print(f"HCurl ndof = {fes_h.ndof}")
    uu, vv = fes_h.TnT()
    a = BilinearForm(nu * curl(uu) * curl(vv) * dx + reg * uu * vv * dx)
    print("Assembling and building GPU solver (reused for every ring's right-hand side)...")
    a.Assemble()
    solve = gpu_solver_for(a.mat, fes_h.FreeDofs(), label="inductance", maxiter=20000)

    print("Solving each ring's own field (reusing the same GPU matrix)...")
    A_vecs = {}
    for k, name in enumerate(all_ring_names):
        f = LinearForm(J[name] * vv * dx)
        f.Assemble()
        gfA = GridFunction(fes_h)
        gfA.vec.FV().NumPy()[:] = solve(f.vec)
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

    sim.save_matrix_like_q3d(L, "induc.mat", display_scale=1e9)


STAGES = {
    "1": ("Capacitance (Electrostatics) -- GPU solve", run_capacitance_gpu),
    "2": ("DC Resistance (DC Conduction) -- GPU solve (little to gain, see docstring)", run_dc_resistance_gpu),
    "2p": ("PEEC self-inductance (ringp1, free space/no core, cross-check) -- GPU pairwise sum",
           lambda: run_inductance_peec_gpu("ringp1")),
    "2c": ("PEEC self-inductance WITH core (ringp1, BEM magnetic-surface-charge correction) -- GPU",
           lambda: run_inductance_peec_core_gpu("ringp1")),
    "3": ("Inductance (curl-curl field solve, closed rings) -- GPU solve", run_inductance_gpu),
    "3t": ("Inductance -- QUICK TEST (ringp1 self-inductance only) -- GPU solve",
           lambda: run_inductance_gpu(test_rings=["ringp1"])),
}

if __name__ == "__main__":
    print(f"cupy sees GPU: {cp.cuda.runtime.getDeviceProperties(0)['name']}")
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
        t0 = time.time()
        STAGES[key][1]()
        print(f"[{time.time()-t0:.1f}s total]")

"""
GPU-accelerated variant of simulation_ngsolve_litz_ratio.py.

Does NOT modify simulation_ngsolve.py / simulation_ngsolve_litz.py /
simulation_ngsolve_litz_ratio.py / simulation_ngsolve_cuda.py -- reuses
their geometry/material/current-source/GPU-solver machinery directly
(build_ratio_geometry, pick_middle, complex_mu_litz, core_mu_effective,
LITZ_PARAMS, gpu_solver_for), and only swaps the per-frequency
"assemble + factor + solve" step for a GPU-accelerated solve using cupy.

The f=0 DC baseline is real/SPD, so it reuses
simulation_ngsolve_cuda.gpu_solver_for() (Jacobi-preconditioned CG),
already validated this session for the analogous real SPD systems
(capacitance, DC resistance, DC inductance). The AC (f>0) solves are
complex and NOT SPD (see below) and need the separate ILU+GMRES approach
implemented in this file -- they are NOT interchangeable, hence two
different solver functions for the two cases.

*** MUST BE RUN WITH THE SAME SEPARATE, PLAIN-ASCII-PATH PYTHON
ENVIRONMENT documented at the top of simulation_ngsolve_cuda.py *** (cupy's
CUDA JIT compiler fails to resolve its own headers under this project's
accented path "Thése" -- see that file's docstring).

Why an ILU preconditioner instead of plain Jacobi: a first GPU attempt at
this exact problem (single-shared-mesh, combined primary+secondary
excitation, complex nu from Litz/lamination homogenization, mu_r~22500
core) used Jacobi-preconditioned GMRES and NEVER converged (info=2000,
hit max iterations every single solve, all frequencies) -- plain diagonal
preconditioning is too weak for this system's conditioning. This version
builds an incomplete LU factorization (scipy.sparse.linalg.spilu, which
supports complex dtype via SuperLU -- cupy itself has no equivalent
built-in) ONCE on CPU per frequency, then uses it as the preconditioner
for GPU-resident GMRES: the expensive O(n) sparse matrix-vector products
run on GPU every iteration, while the (much cheaper, applied once per
iteration) triangular preconditioner solve runs on CPU. This hybrid
still needs the ILU factorization itself built on CPU (comparable cost to
the direct sparsecholesky factorization the pure-CPU version uses, though
tunable via drop_tol/fill_factor to trade preconditioner quality for
memory), so the actual benefit is confined to the GMRES matvec loop, not
a full replacement of the CPU solve -- treat this as an experiment in
convergence quality first, wall-clock speed second, until proven otherwise
by direct comparison against the CPU version's timings.
"""
import sys
import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla

sys.path.insert(0, r"C:\Users\hp\OneDrive - CEFEM INDUSTRIES\These\simulations\simulation HF TMF")
import simulation_ngsolve as sim  # noqa: E402
import simulation_ngsolve_litz as lz  # noqa: E402
import simulation_ngsolve_cuda as sim_cuda  # noqa: E402 -- reuses its gpu_solver_for (Jacobi+CG),
# already validated this session for real SPD systems (capacitance, DC
# resistance, DC inductance) -- used below for the f=0 DC baseline here,
# which genuinely IS real/SPD, unlike the complex AC solves (see
# gpu_solver_ilu_gmres's docstring for why THOSE need a different approach
# and haven't converged with either Jacobi or ILU preconditioning yet).
from simulation_ngsolve_litz_ratio import (  # noqa: E402
    build_ratio_geometry, pick_middle, MU0, MATRIX_DIR, DC_MATRIX_DIR,
)
from config import sim_frequencies  # noqa: E402
import config as _config  # noqa: E402

from ngsolve import HCurl, GridFunction, BilinearForm, LinearForm, curl, dx, Integrate, Conj, InnerProduct  # noqa: E402

import os  # noqa: E402
from scipy.io import loadmat, savemat  # noqa: E402

import cupy as cp  # noqa: E402
import cupyx.scipy.sparse as cpsp  # noqa: E402
import cupyx.scipy.sparse.linalg as cpspla  # noqa: E402


def gpu_solver_ilu_gmres(a_mat, freedofs, rtol=1e-8, maxiter=2000, restart=200,
                          drop_tol=1e-4, fill_factor=10, label=""):
    """Builds a CPU incomplete-LU preconditioner ONCE for this assembled
    (complex) matrix, then returns a callable solve(rhs) that GPU-solves
    via preconditioned GMRES for any right-hand side -- the matvecs run on
    GPU, the triangular preconditioner solve runs on CPU each iteration
    (transferring a length-n_free vector back and forth, not the full
    matrix, so this overhead is small relative to the matvec cost at
    reasonable problem sizes)."""
    t0 = time.time()
    vals, cols, rowptr = a_mat.CSR()
    vals = np.array(vals, dtype=np.complex128)
    cols = np.array(cols)
    rowptr = np.array(rowptr)
    A_full = sp.csr_matrix((vals, cols, rowptr), shape=a_mat.shape)

    mask = np.array([freedofs[i] for i in range(len(freedofs))], dtype=bool)
    idx_free = np.nonzero(mask)[0]
    A_free = A_full[idx_free, :][:, idx_free].tocsc()
    n_free = len(idx_free)

    print(f"  [gpu-ilu{(' ' + label) if label else ''}] building ILU preconditioner "
          f"({n_free} free dofs, {A_free.nnz} nnz)...")
    # SuperLU's incomplete factorization can hit an exact-zero pivot on
    # this system (confirmed empirically: "Factor is exactly singular" at
    # drop_tol=1e-4) -- a small diagonal shift is the standard, cheap
    # mitigation (shifts eigenvalues away from the factorization's
    # numerically-singular direction without perceptibly changing the
    # preconditioner's quality, since it's already an APPROXIMATE inverse).
    diag_shift = 1e-10 * np.abs(A_free.diagonal()).mean()
    A_free_shifted = A_free + diag_shift * sp.eye(n_free, format="csc", dtype=np.complex128)
    try:
        ilu = sla.spilu(A_free_shifted, drop_tol=drop_tol, fill_factor=fill_factor)
    except RuntimeError as e:
        print(f"    [gpu-ilu warn] spilu failed ({e}) even with diagonal shift -- "
              f"retrying with a larger shift and more conservative drop_tol")
        diag_shift = 1e-6 * np.abs(A_free.diagonal()).mean()
        A_free_shifted = A_free + diag_shift * sp.eye(n_free, format="csc", dtype=np.complex128)
        ilu = sla.spilu(A_free_shifted, drop_tol=drop_tol * 0.1, fill_factor=fill_factor * 2)
    print(f"    ILU built in {time.time() - t0:.1f}s")

    A_gpu = cpsp.csr_matrix(A_free.tocsr())

    def M_precond(x_gpu):
        x_cpu = cp.asnumpy(x_gpu)
        y_cpu = ilu.solve(x_cpu)
        return cp.asarray(y_cpu)

    M_op = cpspla.LinearOperator(A_gpu.shape, matvec=M_precond, dtype=np.complex128)

    def solve(rhs_complex_np):
        b_free_gpu = cp.asarray(rhs_complex_np[idx_free].astype(np.complex128))
        x_free_gpu, info = cpspla.gmres(A_gpu, b_free_gpu, rtol=rtol, maxiter=maxiter, restart=restart, M=M_op)
        if info != 0:
            print(f"    [gpu-ilu warn] GMRES did not fully converge (info={info})")
        x_full = np.zeros(len(mask), dtype=np.complex128)
        x_full[idx_free] = cp.asnumpy(x_free_gpu)
        return x_full

    return solve


J_primary_bundle_names = []
J_secondary_bundle_names = []


def _solve_combined_gpu(mesh, Ldom, J_primary_bundle, J_secondary_bundle, I_primary, I_secondary,
                         freq_hz, order=1, reg_factor=1e-4):
    """GPU/ILU-GMRES analogue of simulation_ngsolve_litz_ratio._solve_combined
    -- same physics, material model, and Rac-from-loss-density extraction;
    only the linear solve is swapped for the ILU-preconditioned GPU GMRES
    above instead of CPU sparsecholesky."""
    J_combined = J_primary_bundle * I_primary + J_secondary_bundle * I_secondary

    mu_complex_p = lz.complex_mu_litz(freq_hz, lz.LITZ_PARAMS["ringp"]["strand_diameter_m"], sim.MATERIALS["ringp"]["sigma"])
    mu_complex_s = lz.complex_mu_litz(freq_hz, lz.LITZ_PARAMS["rings"]["strand_diameter_m"], sim.MATERIALS["ringp"]["sigma"])
    mu_complex_core = lz.core_mu_effective(freq_hz)

    is_complex = freq_hz != 0
    if not is_complex:
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

    fes_h = HCurl(mesh, order=order, nograds=True, dirichlet="outer", complex=is_complex)
    uu, vv = fes_h.TnT()
    a = BilinearForm(nu * curl(uu) * curl(vv) * dx + reg * uu * vv * dx)
    a.Assemble()
    f_lf = LinearForm(J_combined * vv * dx)
    f_lf.Assemble()

    if not is_complex:
        # f=0 baseline: real, SPD system -- use the ALREADY-VALIDATED GPU
        # Jacobi+CG solver from simulation_ngsolve_cuda.py (confirmed
        # working this session for capacitance/DC-resistance/DC-inductance,
        # all real SPD systems just like this one) instead of CPU
        # sparsecholesky.
        gpu_solve = sim_cuda.gpu_solver_for(a.mat, fes_h.FreeDofs(), label="DC baseline")
        gfA = GridFunction(fes_h)
        gfA.vec.FV().NumPy()[:] = gpu_solve(f_lf.vec)
    else:
        rhs_np = np.asarray(f_lf.vec.FV().NumPy(), dtype=np.complex128)
        gpu_solve = gpu_solver_ilu_gmres(a.mat, fes_h.FreeDofs(), label=f"f={freq_hz:g}Hz")
        x_np = gpu_solve(rhs_np)
        gfA = GridFunction(fes_h)
        gfA.vec.FV().NumPy()[:] = x_np

    L_primary = Integrate(gfA * J_primary_bundle, mesh) / I_primary ** 2
    L_secondary = Integrate(gfA * J_secondary_bundle, mesh) / I_secondary ** 2

    if not is_complex:
        return L_primary, L_secondary, 0.0, 0.0

    omega = 2 * np.pi * freq_hz
    B = curl(gfA)
    B_mag_sq = InnerProduct(B, Conj(B)).real
    loss_density = 0.5 * omega * nu.imag * B_mag_sq

    P_loss_primary = Integrate(loss_density, mesh, definedon=mesh.Materials("ringp.*"))
    P_loss_secondary = Integrate(loss_density, mesh, definedon=mesh.Materials("rings.*"))
    Rac_loss_primary = 2 * P_loss_primary / I_primary ** 2
    Rac_loss_secondary = 2 * P_loss_secondary / I_secondary ** 2

    return L_primary, L_secondary, Rac_loss_primary, Rac_loss_secondary


def run_ratio_sweep_gpu(frequencies_hz=None, count=None, primary_count=None, secondary_count=None, **geom_kwargs):
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
    N_primary_total = len(primary)
    N_secondary_total = len(secondary)

    mid_primary = pick_middle(primary, primary_count)
    mid_secondary = pick_middle(secondary, secondary_count)
    J_primary_bundle_names = mid_primary
    J_secondary_bundle_names = mid_secondary
    print(f"Representative sample: primary={mid_primary}  secondary={mid_secondary}")

    mesh, J, Rdc_turn, Ldom = build_ratio_geometry(mid_primary, mid_secondary, **geom_kwargs)

    def _sum_cf(names):
        total = J[names[0]]
        for name in names[1:]:
            total = total + J[name]
        return total

    J_primary_bundle = _sum_cf(mid_primary)
    J_secondary_bundle = _sum_cf(mid_secondary)

    I_primary = 1.0
    I_secondary = -I_primary * (N_primary_total / N_secondary_total)
    print(f"I_primary={I_primary:.4g} A, I_secondary={I_secondary:.4g} A "
          f"(reversed/opposing, ampere-turn balance, N_primary_total={N_primary_total}, N_secondary_total={N_secondary_total})")

    Rdc_primary = sum(Rdc_turn[name] for name in mid_primary)
    Rdc_secondary = sum(Rdc_turn[name] for name in mid_secondary)
    print(f"Rdc_primary_bundle={Rdc_primary:.6g} ohm, Rdc_secondary_bundle={Rdc_secondary:.6g} ohm")

    print("Solving DC (f=0) baseline for Ldc...")
    Ldc_primary, Ldc_secondary, _, _ = _solve_combined_gpu(mesh, Ldom, J_primary_bundle, J_secondary_bundle,
                                                            I_primary, I_secondary, freq_hz=0.0)
    Ldc_primary, Ldc_secondary = Ldc_primary.real, Ldc_secondary.real
    print(f"Ldc_primary_bundle={Ldc_primary * 1e9:.2f} nH, Ldc_secondary_bundle={Ldc_secondary * 1e9:.2f} nH")

    ratio_R_primary = np.zeros(len(frequencies_hz))
    ratio_R_secondary = np.zeros(len(frequencies_hz))
    ratio_L_primary = np.zeros(len(frequencies_hz))
    ratio_L_secondary = np.zeros(len(frequencies_hz))

    for fi, freq_hz in enumerate(frequencies_hz):
        Lc_p, Lc_s, Rac_loss_p, Rac_loss_s = _solve_combined_gpu(
            mesh, Ldom, J_primary_bundle, J_secondary_bundle, I_primary, I_secondary, freq_hz=freq_hz)

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
        diag_R = np.where(is_primary_row, ratio_R_primary[fi], ratio_R_secondary[fi])
        R_ratio_mats[f"f_{sim.freq_label(freq_hz / 1e3)}"] = np.diag(diag_R)

        row_ratio = np.where(is_primary_row, ratio_L_primary[fi], ratio_L_secondary[fi])
        L_ratio_mats[f"f_{sim.freq_label(freq_hz / 1e3)}"] = np.sqrt(np.outer(row_ratio, row_ratio))

    R_ratio_mats["frequencies_hz"] = np.array(frequencies_hz).reshape(1, -1)
    L_ratio_mats["frequencies_hz"] = np.array(frequencies_hz).reshape(1, -1)

    r_path = os.path.join(MATRIX_DIR, "R_ratio_ngsolve_gpu.mat")
    l_path = os.path.join(MATRIX_DIR, "L_ratio_ngsolve_gpu.mat")
    savemat(r_path, R_ratio_mats)
    savemat(l_path, L_ratio_mats)
    print(f"Saved {r_path}, {l_path}")

    return {
        "frequencies_hz": frequencies_hz,
        "ratio_R_primary": ratio_R_primary, "ratio_R_secondary": ratio_R_secondary,
        "ratio_L_primary": ratio_L_primary, "ratio_L_secondary": ratio_L_secondary,
        "Rdc_primary": Rdc_primary, "Rdc_secondary": Rdc_secondary,
        "Ldc_primary": Ldc_primary, "Ldc_secondary": Ldc_secondary,
    }


if __name__ == "__main__":
    run_ratio_sweep_gpu()

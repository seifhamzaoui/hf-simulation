"""
Diagnostic (NOT part of the main pipeline -- simulation_ngsolve.py /
simulation_ngsolve_cuda.py are not touched by this): reproduces
run_inductance_gpu(test_rings=["ringp1"])'s own geometry/mesh/solve
exactly, via that module's own unmodified helper functions, but ALSO
reports the fill_factor-weighted energy integral restricted to JUST the
Core+entrefer regions (definedon=mesh.Materials("Core|entrefer")) --
i.e. how much of ringp1's total self-inductance-as-energy is actually
sitting in the core+gap, versus the windings'/air's own leakage-field
share. This is a different, narrower quantity than L_ij itself (which
integrates over the WHOLE mesh); it's not a replacement for it.
"""
from netgen.occ import OCCGeometry, Box, Pnt, Glue
from ngsolve import HCurl, GridFunction, BilinearForm, LinearForm, curl, dx, Integrate, InnerProduct

import simulation_ngsolve as sim
from simulation_ngsolve_cuda import gpu_solver_for

test_rings = ["ringp1"]
# Coarsened vs. run_inductance_gpu()'s own defaults (0.001/None->0.002/0.02/
# 0.004/0.0015/0.03) -- this machine is currently low on free RAM (~5.4GB)
# and the default single-ring mesh (445k elements) reliably dies during the
# ring current-source solve. The Core+entrefer energy FRACTION this script
# reports should be reasonably robust to this coarsening (it's a diagnostic
# ratio, not a production result) -- this file only, not the main code.
entrefer_maxh = 0.004
entrefer_solid_maxh = 0.003
core_fill_factor_aware = True

all_ring_names = test_rings
N = len(all_ring_names)
sigma_copper = sim.MATERIALS["ringp"]["sigma"]
mu_r_core = sim.MATERIALS["Core"]["mu_r"]

print("Loading closed-ring geometry...")
solids = sim.load_solids(sim.STEP_FILE_CLOSED)
core = sim.core_solid_from(solids)
core.mat("Core")
core.maxh = 0.03
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
    ring_passive.maxh = 0.006
    battery.mat(f"{name}_battery")
    battery.maxh = 0.003
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
air.maxh = 0.05

glued_parts = [air, core] + ring_parts
if entrefer_solid is not None:
    glued_parts.append(entrefer_solid)

print("Meshing (whole assembly, ~2 minutes)...")
geo = OCCGeometry(Glue(glued_parts))
from ngsolve import Mesh
mesh = Mesh(geo.GenerateMesh(maxh=0.05))
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
print("Assembling and building GPU solver...")
a.Assemble()
solve = gpu_solver_for(a.mat, fes_h.FreeDofs(), label="inductance", maxiter=20000)

print("Solving ringp1's own field...")
name = all_ring_names[0]
f = LinearForm(J[name] * vv * dx)
f.Assemble()
gfA = GridFunction(fes_h)
gfA.vec.FV().NumPy()[:] = solve(f.vec)

print()
print("--- Comparison: total L_11 vs. Core+entrefer-only energy share ---")

L_total = Integrate(nu * InnerProduct(curl(gfA), curl(gfA)) * energy_weight, mesh)
print(f"Total self-inductance (whole mesh, fill_factor-weighted in Core+entrefer): {L_total * 1e9:.4f} nH")

L_core_entrefer_only = Integrate(
    nu * InnerProduct(curl(gfA), curl(gfA)) * energy_weight,
    mesh, definedon=mesh.Materials("Core|entrefer"),
)
print(f"Core+entrefer-only share of that energy:                                 {L_core_entrefer_only * 1e9:.4f} nH")
print(f"Fraction of total sitting in Core+entrefer:                              {L_core_entrefer_only / L_total * 100:.2f}%")

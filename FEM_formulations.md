# Finite Element Formulations Used in This Project

This document collects the strong-form PDEs, the weak/finite-element forms actually implemented (`simulation_ngsolve.py`, `simulation_ngsolve_litz.py`), and the Litz-wire and core homogenization formulas used to turn a stranded/laminated winding into a solvable bulk material.

---

## 1. Electrostatics — Capacitance (`run_capacitance`)

**Strong form** (Laplace/Poisson equation for the electric scalar potential φ, no free charge):

$$
\nabla\cdot(\varepsilon \nabla \varphi) = 0 \quad \text{in } \Omega
$$

with Dirichlet conditions `φ = 1` on the driven conductor's surface, `φ = 0` on all other conductor surfaces (each conductor solved as its own driven case to build the full capacitance matrix), and $\varepsilon = \varepsilon_0 \varepsilon_r$ piecewise-constant per material (from `config.py`'s `MATERIALS[...]["ngsolve"]["eps_r"]`).

**Weak form** (H¹ Galerkin, code: [simulation_ngsolve.py:309](simulation_ngsolve.py#L309)):

$$
\int_\Omega \varepsilon\, \nabla u \cdot \nabla v \, dV = 0 \qquad \forall v \in H^1_0(\Omega)
$$

Capacitance is extracted from the stored electrostatic energy of the solved field, then reduced (Core folded into the diagonal) to match Q3D's reported convention.

---

## 2. DC Conduction — DC Resistance (`run_dc_resistance`)

**Strong form** (steady-state current conservation, no volumetric source):

$$
\nabla\cdot(\sigma \nabla \varphi) = 0 \quad \text{in } \Omega_\text{ring}
$$

solved **per ring independently** (each ring's own solid, not the full assembly), with Dirichlet `φ=1` on the `_hi` terminal face, `φ=0` on the `_gnd` terminal face.

**Weak form** ([simulation_ngsolve.py:408](simulation_ngsolve.py#L408)):

$$
\int_{\Omega_\text{ring}} \sigma_\text{eff}\, \nabla u \cdot \nabla v \, dV = 0
$$

Resistance from the dissipated power identity:

$$
R = \frac{V^2}{P}, \qquad P = \int_{\Omega_\text{ring}} \sigma_\text{eff}\, |\nabla \varphi|^2 \, dV, \quad V = 1\text{ V}
$$

$\sigma_\text{eff}$ is either the bulk copper conductivity $\sigma_\text{Cu}$ (solid-conductor assumption, validated against Q3D to <0.2%), or $\sigma_\text{Cu} \cdot \text{fill\_factor}$ when `config.DC_RESISTANCE_LITZ_AWARE = True` — see §4.

---

## 3. DC Magnetostatics — Self/Mutual Inductance (`run_inductance`)

**Strong form** (Ampère's law, magnetoquasistatic, in terms of the magnetic vector potential **A**, $\mathbf B = \nabla\times\mathbf A$):

$$
\nabla\times\!\big(\nu \,\nabla\times \mathbf A\big) = \mathbf J \quad \text{in } \Omega, \qquad \mathbf A = 0 \text{ on } \partial\Omega_\text{outer}
$$

with $\nu = 1/(\mu_0\mu_r)$ piecewise-constant (Core: $\mu_r\approx 30000$; windings, entrefer, air: $\mu_r=1$), and $\mathbf J$ the ring's own current density (built by a separate closed-loop, EMF-driven DC-conduction "battery" sub-solve, normalized to exactly 1 A — see `_solve_ring_current`).

**Weak form**, H(curl) Galerkin, with a small Tikhonov-style regularization to remove the curl-curl operator's null space (gradient fields) ([simulation_ngsolve.py:645](simulation_ngsolve.py#L645)):

$$
\int_\Omega \nu\, (\nabla\times \mathbf u)\cdot(\nabla\times \mathbf v)\, dV \;+\; \text{reg}\!\int_\Omega \mathbf u\cdot \mathbf v\, dV \;=\; \int_\Omega \mathbf J \cdot \mathbf v\, dV
$$

Self/mutual inductance from the standard energy identity, for unit-normalized currents:

$$
L_{ij} = \int_\Omega \mathbf A_i \cdot \mathbf J_j \, dV
$$

(This is mathematically the discrete form of $L_{ij} = \Psi_i/I_j$, and was cross-validated this session against an independent flux calculation, $L = \Phi/I$ through a cut plane, to ~1%.)

---

## 4. Frequency-Domain (Litz) Sweep — AC Inductance & Resistance (`simulation_ngsolve_litz.run_litz_sweep`)

**Strong form** — same Ampère's-law PDE as §3, but $\nu$ is now **complex** and **frequency-dependent** in the winding and core regions, while $\mathbf J$ stays the same real, frequency-independent 1 A current distribution (valid because Litz wire is designed to keep each strand's own current uniform well past the frequencies of interest):

$$
\nabla\times\!\big(\nu(\omega) \,\nabla\times \mathbf A(\omega)\big) = \mathbf J \quad \text{in } \Omega
$$

$$
\nu(\omega) = \frac{1}{\mu_0\,\mu_\text{complex}(\omega)}
$$

**Weak form** — identical structure to §3 but over a **complex** H(curl) space:

$$
\int_\Omega \nu(\omega)\, (\nabla\times \mathbf u)\cdot(\nabla\times \mathbf v)\, dV \;+\; \text{reg}\!\int_\Omega \mathbf u\cdot \mathbf v\, dV \;=\; \int_\Omega \mathbf J \cdot \mathbf v\, dV, \qquad \mathbf u,\mathbf v \in H(\mathrm{curl};\mathbb C)
$$

**Extraction**, for unit currents:

$$
L_\text{complex}(\omega) = \int_\Omega \mathbf A(\omega)\cdot \mathbf J \, dV = L'(\omega) - jL''(\omega)
$$

$$
L(\omega) = \operatorname{Re}\big(L_\text{complex}\big), \qquad R_\text{ac,added}(\omega) = -\,\omega\,\operatorname{Im}\big(L_\text{complex}\big)
$$

(The minus sign was fixed **empirically** this session — the naive `+ω·Im(L_complex)` gave a negative, unphysical added resistance at 1 MHz; the sign above was confirmed to give `Rac ≥ 0` at every tested frequency.)

---

### 4.1 Litz-wire winding model — complex permeability (Bessel formula)

Each winding (`ringp`/`rings`) is a bundle of many thin, individually-insulated round strands. Rather than meshing every strand, the bundle is homogenized into one bulk region whose complex permeability reproduces each strand's own 1-D radial skin effect exactly (classical result, time convention $e^{+j\omega t}$):

$$
k = \frac{1-j}{\delta}, \qquad \delta = \sqrt{\frac{2}{\omega\,\mu_0\,\mu_r\,\sigma}} \quad\text{(skin depth)}
$$

$$
\boxed{\;\mu_\text{complex}(\omega) = \mu_r \cdot \frac{2}{ka}\cdot\frac{J_1(ka)}{J_0(ka)}\;}
$$

where $a$ = strand radius (`strand_diameter_m / 2`), $\sigma$ = copper conductivity, and $J_0, J_1$ are Bessel functions of the first kind. Implemented in `complex_mu_litz()`. As $\omega\to 0$, $\mu_\text{complex}\to\mu_r$ (real, lossless) — verified by `_selftest_complex_mu()`.

**Scope/limitation**: this captures each strand's own skin effect only. It does **not** model strand-to-strand *proximity* effect (the extra loss a strand picks up from its neighbors' field) — a real, documented gap, confirmed this session by comparing against FEMM's reference values (`femm/R_Ratio.mat`), which diverge increasingly from this model at higher frequency.

### 4.2 Core model — complex permeability (lamination formula)

The core is a single solid (or laminated) ferrite block, not a stranded bundle, so it gets a **different** closed form — the classical 1-D skin effect solution for a **slab** of thickness $d$, instead of a round wire:

$$
\boxed{\;\mu_\text{complex}(\omega) = \mu_r \cdot \frac{\tanh(kd/2)}{kd/2}\;}, \qquad k = \frac{1-j}{\delta}
$$

Implemented in `complex_mu_lamination()`, using `config.py`'s `MATERIALS["Core"]["ac"]` (`sigma_ac`, `lamination_thickness_m`, `mu_r`).

**Stacking (fill) factor correction**: a real lamination stack is only partially magnetic material (the rest is inter-lamination insulation). The stacking factor $f$ (`MATERIALS["Core"]["ac"]["fill_factor"]`) dilutes the raw formula:

$$
\boxed{\;\mu_\text{eff}(\omega) = 1 + f\,\big(\mu_\text{complex}(\omega) - 1\big)\;}
$$

Implemented in `core_mu_effective()` — this is the function actually used in the sweep, not the raw `complex_mu_lamination()` output.

**Important caveat found this session**: because most of a gapped transformer's total inductance is stored in the core, even the core's *small* loss tangent ($\operatorname{Im}(\mu)/\operatorname{Re}(\mu)$) can dominate the total `Rac`, more than the windings' own Litz skin effect does. The core's `sigma_ac`/`lamination_thickness_m`/`fill_factor` are currently **placeholder values** — the computed AC/DC ratios are more sensitive to these unvalidated core parameters right now than to the actual wire spec.

---

## 5. AC/DC Ratio Sweep (`simulation_ngsolve_litz_ratio.run_ratio_sweep`)

Not a new PDE — reuses §4's formulation on a small, dynamically-chosen representative sample (3 turns near the middle of each winding, solved **in isolation** per winding, not combined, to avoid mutual-coupling contamination between primary and secondary). For each winding:

$$
\text{ratio}_L(\omega) = \frac{L_\text{series}(\omega)}{L_\text{series}(0)}, \qquad
\text{ratio}_R(\omega) = \frac{R_\text{dc} + R_\text{ac,added}(\omega)}{R_\text{dc}}
$$

where $L_\text{series}$ is the full sum of the 3×3 self+mutual submatrix (turns in series), and $R_\text{dc}$ comes from the Ohmic identity $R = \int J^2/\sigma_\text{eff}\, dV$ applied to the same normalized current field already built for the AC solve. These scalar ratios are then broadcast onto every turn of that winding (diagonal matrix per frequency, same convention as `femm/R_Ratio.mat` / `L_Ratio.mat`) and used to scale the full DC reference matrices (`DCR.mat`, `induc.mat`) into AC-corrected ones — avoiding a full 60-turn resolve at every frequency.

"""
Maxwell 3D equivalent of simulation.py (Q3D version).

Runs three separate solves in the same AEDT project, one design per physics,
since Maxwell 3D (unlike Q3D) does not support Cap/DC/AC results out of a
single "net" excitation model:

  1. Electrostatic  -> capacitance matrix (diagonal = ring-to-core,
                       off-diagonal = ring-to-ring)
  2. DC Conduction  -> DC resistance of each ring
  3. EddyCurrent    -> self (proper) and mutual inductances between rings,
                       plus genuine frequency-dependent skin/proximity-effect
                       AC resistance -- uses the CLOSED-ring geometry (every
                       ring a genuine torus) since the standard AC Magnetic/
                       EddyCurrent solver's Winding excitation needs a
                       genuinely closed current loop -- see
                       run_inductance()'s docstring below.

Terminal faces on each ring are picked the same way as in simulation.py:
by filtering the ring's faces to the ones whose area matches the expected
conductor cross-section (rectangular or circular) from config.py.

Each matrix is pulled out of AEDT via a scratch .txt export (deleted right
after parsing -- only .mat files are kept), then saved into
"./maxwell matrices" in the same {"f_<freq>kHz": matrix, "frequencies_hz":
[[..]]} layout the Q3D-based simulation.py uses in traited Values/. If a
block can't be matched, a warning lists the block titles AEDT actually
produced instead of guessing.

An input() gate is placed right before each of the three analyze_setup()
calls so every simulation can be inspected/validated in the AEDT GUI first.
"""

import os
import re
import tempfile

import numpy as np
from scipy.io import savemat

from ansys.aedt.core import Maxwell3d
from ansys.aedt.core.modules.boundary.maxwell_boundary import (
    MatrixACMagnetic,
    MatrixElectric,
    SourceACMagnetic,
)

from config import *

STEP_FILE = "transformer_model.step"
STEP_FILE_CLOSED = "transformer_model_closed.step"  # closed rings (no terminal cut) -- see run_inductance()
PROJECT_NAME = "my_project_maxwell"
MATRIX_DIR = "./maxwell matrices"

os.makedirs(MATRIX_DIR, exist_ok=True)


# ============================================================
# Helpers shared by all three stages
# ============================================================

def assign_material_by_prefix(app, prefix, material):
    matches = [n for n in app.modeler.object_names if n.startswith(prefix)]
    for name in matches:
        obj = app.modeler.get_object_from_name(name)
        obj.material_name = material
    return matches


def face_area_bounds(width, thickness, circular):
    if not circular:
        nominal = width * thickness
    else:
        nominal = (thickness / 2) ** 2 * 3.14
    return nominal * 0.9, nominal * 1.1


def select_ring_terminal_faces(ring, min_area, max_area):
    faces = [f for f in ring.faces if min_area <= f.area <= max_area]
    print(f"{ring.name}: {len(faces)} candidate terminal face(s) in area range")
    return faces


REGION_PAD_PERCENT = 50  # tune down further if a solve is still slow


def create_air_region(app, pad_percent=REGION_PAD_PERCENT):
    """Bounded air region for FEM volume solves (Electrostatic / AC Magnetic).

    AEDT's own default padding is 300% per side if no region is created
    explicitly, which multiplies the volume mesh for no accuracy benefit --
    a much tighter box around the windings is enough. DC Conduction doesn't
    need this at all since current only flows through the conductors.
    """
    app.modeler.create_region(pad_value=pad_percent, pad_type="Percentage Offset")


def import_and_prepare_geometry(app, step_file=STEP_FILE):
    app.modeler.import_3d_cad(
        input_file=step_file,
        healing=False,
        refresh_all_ids=True,
        import_materials=False,
        create_lightweight_part=False,
        group_by_assembly=False,
        create_group=True,
        merge_planar_faces=True,
    )

    assign_material_by_prefix(app, "ringp", MATERIALS["ringp"]["aedt"])
    assign_material_by_prefix(app, "rings", MATERIALS["rings"]["aedt"])
    assign_material_by_prefix(app, "pinsulator", MATERIALS["pinsulator"]["aedt"])
    assign_material_by_prefix(app, "sinsulator", MATERIALS["sinsulator"]["aedt"])
    assign_material_by_prefix(app, "p_layer insulator", MATERIALS["p_layer insulator"]["aedt"])
    assign_material_by_prefix(app, "s_layer insulator", MATERIALS["s_layer insulator"]["aedt"])
    assign_material_by_prefix(app, "primary_secondary_insulation", MATERIALS["primary_secondary_insulation"]["aedt"])
    assign_material_by_prefix(app, "Core", MATERIALS["ringp"]["aedt"])  # solved as a plain copper conductor net for this stage


def get_rings(app):
    primary_names = [n for n in app.modeler.object_names if n.startswith("ringp")]
    secondary_names = [n for n in app.modeler.object_names if n.startswith("rings")]
    primary = [app.modeler.get_object_from_name(n) for n in primary_names]
    secondary = [app.modeler.get_object_from_name(n) for n in secondary_names]
    return primary, secondary


# ============================================================
# Parsing of AEDT's native Matrix .txt export
#
# Confirmed layout (from an actual CapacitanceMatrix export), tab-delimited:
#
#   Solution : CapSetup : LastAdaptive
#   Parameter : CapacitanceMatrix
#   Capacitance Unit: pF
#
#   Capacitance
#           Core_V  ringp1_V  ringp2_V  ...
#   Core_V  54.59   -6.4967   -2.0526   ...
#   ringp1_V        -6.4967   189.62    -174.67  ...
#   ...
#
# i.e. a bare quantity title ("Capacitance"/"Resistance"/"Inductance"/
# "AC Resistance"), then a tab-separated header row of conductor names,
# then one tab-separated data row per conductor (row label first, matching
# the header order). There is no "Conductor in Row/Column N" line at all.
# ============================================================

_UNIT_LINE_RE = re.compile(r"^(.*)\s+Unit:\s*(\S+)\s*$")
_UNIT_TO_SI = {
    "ff": 1e-15, "pf": 1e-12, "nf": 1e-9, "uf": 1e-6, "mf": 1e-3, "f": 1.0,
    "nh": 1e-9, "uh": 1e-6, "mh": 1e-3, "h": 1.0, "henry": 1.0,
    "microohm": 1e-6, "milliohm": 1e-3, "ohm": 1.0, "kohm": 1e3, "megohm": 1e6,
    "usie": 1e-6, "msie": 1e-3, "sie": 1.0, "ksie": 1e3,
}


def _unit_to_si_factor(unit):
    factor = _UNIT_TO_SI.get(unit.lower())
    if factor is None:
        print(f"[warn] unrecognized unit '{unit}', keeping raw (unconverted) values")
        return 1.0
    return factor


def parse_aedt_matrix_export(filepath):
    """Returns {section_title: (names, matrix_in_SI_units)} for an AEDT Matrix .txt export.

    A section is skipped (with a printed warning) rather than guessed if its
    row labels don't line up exactly with its header row.
    """
    with open(filepath, "r") as f:
        lines = [line.rstrip("\n") for line in f]

    blocks = {}
    idx = 0
    while idx < len(lines):
        stripped = lines[idx].strip()

        is_title = bool(stripped) and ":" not in stripped
        if not is_title:
            idx += 1
            continue

        # A quantity's title is preceded by "<Quantity> Unit: <unit>" if it has
        # one. Dimensionless quantities AEDT also throws in the same export
        # (e.g. "Capacitive Coupling Coefficient") have no such line, so look
        # back only as far as the nearest non-blank line rather than carrying
        # the previous section's unit forward.
        unit_factor = 1.0
        b = idx - 1
        while b >= 0 and not lines[b].strip():
            b -= 1
        if b >= 0:
            unit_match = _UNIT_LINE_RE.match(lines[b].strip())
            if unit_match:
                unit_factor = _unit_to_si_factor(unit_match.group(2))

        j = idx + 1
        while j < len(lines) and not lines[j].strip():
            j += 1
        if j >= len(lines):
            idx += 1
            continue

        names = [c.strip() for c in lines[j].split("\t") if c.strip()]
        if len(names) < 2:
            idx += 1
            continue

        rows, row_names = [], []
        k = j + 1
        while k < len(lines) and len(rows) < len(names):
            cells = [c.strip() for c in lines[k].split("\t") if c.strip()]
            if len(cells) != len(names) + 1:
                break
            try:
                values = [float(x) for x in cells[1:]]
            except ValueError:
                break
            row_names.append(cells[0])
            rows.append(values)
            k += 1

        if len(rows) == len(names) and row_names == names:
            blocks[stripped] = (names, np.array(rows) * unit_factor)
            idx = k
        else:
            print(f"[warn] '{stripped}' block in {filepath} did not line up with its "
                  f"{len(names)}x{len(names)} header, skipped")
            idx += 1

    return blocks


def export_and_parse(app, matrix_name, setup_name):
    """Export the named AEDT matrix to a scratch .txt file, parse it, then
    delete the .txt -- only .mat files are kept in MATRIX_DIR."""
    fd, txt_path = tempfile.mkstemp(suffix=".txt")
    os.close(fd)
    try:
        app.export_matrix(matrix_name=matrix_name, output_file=txt_path, setup=setup_name)
        return parse_aedt_matrix_export(txt_path)
    finally:
        os.remove(txt_path)


def select_primary_block(blocks, keyword):
    """Pick the block matching every word in `keyword`, skipping bonus
    normalized/dimensionless blocks AEDT throws in alongside the physical
    one (e.g. "Capacitive Coupling Coefficient" next to "Capacitance")."""
    tokens = keyword.lower().split()
    candidates = [
        (label, data) for label, data in blocks.items()
        if all(t in label.lower() for t in tokens)
        and "coefficient" not in label.lower()
        and "coupling" not in label.lower()
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda item: len(item[0]))
    return candidates[0]


def freq_label(freq_khz):
    return f"{freq_khz:.6g}kHz"


def save_matrix_like_q3d(blocks, keyword, mat_name, freq_khz, reduce_first=False):
    """Save one parsed block in the same {"f_<freq>kHz": matrix, "frequencies_hz":
    [[..]]} layout used by the Q3D-based simulation.py outputs in traited Values/."""
    picked = select_primary_block(blocks, keyword)
    if picked is None:
        print(f"[warn] no block matching '{keyword}' found, {mat_name} not saved.")
        print(f"       blocks available in this export: {list(blocks.keys())}")
        print("       adjust the keyword to match if AEDT titled this block differently.")
        return

    label, (names, matrix) = picked
    print(f"Using block '{label}' ({len(names)} conductors) for {mat_name}")

    if reduce_first:
        first_row = matrix[0, 1:]
        reduced = matrix[1:, 1:].copy()
        np.fill_diagonal(reduced, first_row)
        matrix = reduced

    matrix = np.abs(matrix)

    key = f"f_{freq_label(freq_khz)}"
    mat_dict = {key: matrix, "frequencies_hz": np.array([[freq_khz * 1e3]])}
    mat_path = os.path.join(MATRIX_DIR, mat_name)
    savemat(mat_path, mat_dict)
    print(f"Saved {mat_path} with key '{key}' (same format as traited Values/{mat_name})")


def save_dc_resistance_like_q3d(blocks, mat_name, freq_khz):
    """DC Conduction reports a Conductance matrix (Siemens), not resistance
    directly. Since the rings have no galvanic connection to each other, this
    matrix is diagonal by construction (verified: off-diagonal ~ 0), so each
    ring's DC resistance is simply 1 / its own diagonal conductance term."""
    picked = select_primary_block(blocks, "Conductance")
    if picked is None:
        print(f"[warn] no 'Conductance' block found, {mat_name} not saved.")
        print(f"       blocks available in this export: {list(blocks.keys())}")
        return

    label, (names, g_matrix) = picked
    print(f"Using block '{label}' ({len(names)} conductors) for {mat_name}")

    diag_g = np.diagonal(g_matrix)
    off_diag_max = np.abs(g_matrix - np.diag(diag_g)).max()
    if off_diag_max > 1e-6 * np.abs(diag_g).max():
        print(f"[warn] conductance matrix has non-negligible off-diagonal terms "
              f"(max {off_diag_max:.3e} S) -- rings may not be as galvanically "
              "isolated as assumed, resistance values below ignore that coupling")

    r_matrix = np.diag(1.0 / diag_g)

    key = f"f_{freq_label(freq_khz)}"
    mat_dict = {key: r_matrix, "frequencies_hz": np.array([[freq_khz * 1e3]])}
    mat_path = os.path.join(MATRIX_DIR, mat_name)
    savemat(mat_path, mat_dict)
    print(f"Saved {mat_path} with key '{key}' (same format as traited Values/{mat_name})")


# Face area thresholds, same formulas as simulation.py (note: the secondary
# block re-uses p_conduc_circular too, mirroring simulation.py exactly)
p_min_area, p_max_area = face_area_bounds(p_cond_width, p_cond_thickness, p_conduc_circular)
s_min_area, s_max_area = face_area_bounds(s_cond_width, s_cond_thickness, p_conduc_circular)

_desktop_started = False


def _open_design(design_name, solution_type):
    """Open (or attach to) the shared project. The first stage that actually
    runs starts a fresh desktop; whichever stages run after it attach to that
    same session -- this works no matter which stage(s) the user picked."""
    global _desktop_started
    m3d = Maxwell3d(
        project=PROJECT_NAME,
        design=design_name,
        solution_type=solution_type,
        non_graphical=False,
        new_desktop=not _desktop_started,
    )
    _desktop_started = True
    return m3d


# ============================================================
# 1. CAPACITANCE -- Electrostatic solver
#    Core is treated as just another signal source, and the reduction
#    (fold Core's row/col into the diagonal) is applied afterwards,
#    exactly like apply_reduction=True did in the Q3D script.
# ============================================================

def run_capacitance():
    m3d = _open_design("Capacitance_Electrostatic", "Electrostatic")

    import_and_prepare_geometry(m3d)
    create_air_region(m3d)
    primary_rings, secondary_rings = get_rings(m3d)

    cap_signal_names = []

    for ring in primary_rings:
        faces = select_ring_terminal_faces(ring, p_min_area, p_max_area)
        v_name = f"{ring.name}_V"
        m3d.assign_voltage(assignment=[faces[0].id], amplitude="1V", name=v_name)
        cap_signal_names.append(v_name)

    for ring in secondary_rings:
        faces = select_ring_terminal_faces(ring, s_min_area, s_max_area)
        v_name = f"{ring.name}_V"
        m3d.assign_voltage(assignment=[faces[0].id], amplitude="1V", name=v_name)
        cap_signal_names.append(v_name)

    core_v_name = "Core_V"
    m3d.assign_voltage(assignment=["Core"], amplitude="0V", name=core_v_name)

    cap_matrix_name = "CapacitanceMatrix"
    m3d.assign_matrix(MatrixElectric(
        signal_sources=[core_v_name] + cap_signal_names,
        matrix_name=cap_matrix_name,
    ))

    m3d.create_setup(
        name="CapSetup",
        setup_type="Electrostatic",
        MaximumPasses=10,
        MinimumPasses=1,
        MinimumConvergedPasses=1,
        PercentError=1,
        UseIterativeSolver=True,
    )

    input("Validate the Electrostatic (capacitance) setup in AEDT, then press Enter to run the simulation...")
    if not m3d.analyze_setup("CapSetup"):
        raise RuntimeError(
            "CapSetup did not solve successfully -- check AEDT's Message Manager / "
            "Progress window for the actual error before re-running."
        )

    cap_blocks = export_and_parse(m3d, cap_matrix_name, "CapSetup")
    save_matrix_like_q3d(
        cap_blocks, keyword="Capacitance", mat_name="cap_data.mat",
        freq_khz=min_freq_kHz, reduce_first=True,
    )


# ============================================================
# 2. DC RESISTANCE OF EACH RING -- DC Conduction solver
#    Each ring is its own isolated 2-terminal path (hi -> gnd), so one
#    combined matrix over all rings gives a diagonal = self DC resistance,
#    off-diagonal ~= 0 since the rings have no galvanic connection.
# ============================================================

def run_dc_resistance():
    m3d = _open_design("Resistance_DCConduction", "DC Conduction")

    import_and_prepare_geometry(m3d)
    primary_rings, secondary_rings = get_rings(m3d)

    dcr_hi_names, dcr_gnd_names = [], []

    for ring in primary_rings:
        faces = select_ring_terminal_faces(ring, p_min_area, p_max_area)
        hi_name, gnd_name = f"{ring.name}_hi", f"{ring.name}_gnd"
        m3d.assign_voltage(assignment=[faces[0].id], amplitude="1V", name=hi_name)
        m3d.assign_voltage(assignment=[faces[1].id], amplitude="0V", name=gnd_name)
        dcr_hi_names.append(hi_name)
        dcr_gnd_names.append(gnd_name)

    for ring in secondary_rings:
        faces = select_ring_terminal_faces(ring, s_min_area, s_max_area)
        hi_name, gnd_name = f"{ring.name}_hi", f"{ring.name}_gnd"
        m3d.assign_voltage(assignment=[faces[0].id], amplitude="1V", name=hi_name)
        m3d.assign_voltage(assignment=[faces[1].id], amplitude="0V", name=gnd_name)
        dcr_hi_names.append(hi_name)
        dcr_gnd_names.append(gnd_name)

    dcr_matrix_name = "ResistanceMatrix"
    m3d.assign_matrix(MatrixElectric(
        signal_sources=dcr_hi_names,
        ground_sources=dcr_gnd_names,
        matrix_name=dcr_matrix_name,
    ))

    m3d.create_setup(
        name="DCRSetup",
        setup_type="DC Conduction",
        MaximumPasses=10,
        MinimumPasses=1,
        MinimumConvergedPasses=1,
        PercentError=1,
        UseIterativeSolver=True,
    )

    input("Validate the DC Conduction (resistance) setup in AEDT, then press Enter to run the simulation...")
    if not m3d.analyze_setup("DCRSetup"):
        raise RuntimeError(
            "DCRSetup did not solve successfully -- check AEDT's Message Manager / "
            "Progress window for the actual error before re-running."
        )

    dcr_blocks = export_and_parse(m3d, dcr_matrix_name, "DCRSetup")
    save_dc_resistance_like_q3d(dcr_blocks, mat_name="DCR.mat", freq_khz=min_freq_kHz)


# ============================================================
# 3. SELF (PROPER) AND MUTUAL INDUCTANCES + genuine EDDY CURRENT (skin/
#    proximity-effect AC resistance) -- standard AC Magnetic/EddyCurrent
#    solver, using the CLOSED-ring geometry (STEP_FILE_CLOSED, every ring a
#    genuine torus, no terminal cut).
#
#    An earlier A-Phi-formulation/open-ring approach was dropped entirely:
#    "AC Magnetic APhi" is not a solution type this AEDT installation
#    (2023.1) recognizes at all (confirmed by AEDT's own error, not a naming
#    issue -- that formulation was added in a later Maxwell release). The
#    standard solver used here needs Winding sources, and a solid Winding's
#    Coil Terminal must either sit on a genuinely CLOSED current loop or
#    touch the Region's outer boundary ("Illegal external terminal"
#    otherwise) -- an open ring (real terminal-face cut, fully enclosed in
#    air) satisfies neither, which is exactly why the old approach needed
#    open rings + A-Phi in the first place. A genuinely closed ring
#    satisfies the first condition directly.
#
#    Coil terminal: a closed ring has no natural terminal face (there is no
#    surface to inject current on), so one is created per ring in
#    _create_ring_coil_terminal() via Object3d.split() (see that function's
#    docstring for why split() and not section()), at the SAME location
#    simulation_ngsolve.py's EMF-battery technique uses (the ring's own local
#    X-minimum edge, at its own Z-center): a plane of constant Z there is
#    exactly perpendicular to the local current-flow (circumferential)
#    direction, giving a clean cross-sectional terminal face.
#
#    This solution_type string ("EddyCurrent") and setup_type match what
#    this AEDT installation (2023.1) actually accepts -- confirmed directly
#    against a disposable test session, since pyaedt's own solution-type
#    table only remaps "AC Magnetic"/"ACMagnetic" to "EddyCurrent" for AEDT
#    2025.1 specifically (older versions, like this one, need the raw
#    "EddyCurrent" string passed straight through). If assign_matrix()'s
#    solution_type-based dispatch ever rejects MatrixACMagnetic the same way
#    it once rejected MatrixACMagneticAPhi (removed), the fix is the same:
#    bypass it and call app._create_boundary(...) directly.
# ============================================================

def _create_ring_coil_terminal(m3d, ring):
    """Creates a cross-sectional face on `ring` (a genuinely closed torus)
    usable as a Winding's coil terminal, by physically SPLITTING the ring
    into two solid pieces at the cut plane rather than sectioning it.

    Object3d.section() was tried first and rejected by AEDT at solve time
    ("Illegal inner terminal ... must be a cross section of a 3D model
    object"), even when sectioning the real ring directly (not a disposable
    duplicate): section() creates its cross-section as a topologically
    INDEPENDENT new sheet object, merely coincident with the ring, not an
    actual member of the ring's own face list -- apparently not what AEDT's
    inner-terminal validation means by "a cross section of a 3D model
    object". Object3d.split(), confirmed against a disposable test object in
    this exact AEDT installation, instead divides the solid into two pieces
    that share a genuine coincident face (a real, distinct face belonging to
    each resulting solid's own topology, not a separate object) -- current
    still flows continuously between the two pieces since they touch and
    share the same conductor material, exactly like the rest of the ring's
    own volume.

    A plane at the ring's own Z-center crosses the closed annulus at BOTH
    sides of the loop (it's an infinite plane), so each resulting piece gets
    a NEW face at both crossing locations -- new faces are found by diffing
    face IDs before/after the split, and only the one nearest the ring's own
    local X-minimum (matching simulation_ngsolve.py's battery location) is
    kept as the terminal.
    """
    bbox = ring.bounding_box  # [xmin, ymin, zmin, xmax, ymax, zmax]
    zc = (bbox[2] + bbox[5]) / 2
    x_target = bbox[0]
    before_face_ids = {f.id for f in ring.faces}

    cs_name = f"{ring.name}_cut_cs"
    m3d.modeler.create_coordinate_system(origin=[0, 0, zc], name=cs_name)
    m3d.modeler.set_working_coordinate_system(cs_name)
    split_names = ring.split(plane="XY", sides="Both")
    m3d.modeler.set_working_coordinate_system("Global")

    piece = m3d.modeler.get_object_from_name(ring.name)
    new_faces = [f for f in piece.faces if f.id not in before_face_ids]
    if not new_faces:
        raise RuntimeError(
            f"{ring.name}: split() produced no new faces on '{ring.name}' (result objects: "
            f"{split_names}) -- inspect the cut location/geometry in AEDT before re-running."
        )

    terminal_face = min(new_faces, key=lambda f: abs(f.center[0] - x_target))
    return terminal_face.id


def run_inductance():
    m3d = _open_design("Inductance_EddyCurrent", "EddyCurrent")

    import_and_prepare_geometry(m3d, step_file=STEP_FILE_CLOSED)
    create_air_region(m3d)
    assign_material_by_prefix(m3d, "Core", MATERIALS["Core"]["aedt"])
    primary_rings, secondary_rings = get_rings(m3d)

    winding_names = []
    for ring in primary_rings + secondary_rings:
        terminal_face_id = _create_ring_coil_terminal(m3d, ring)
        winding = m3d.assign_winding(
            assignment=[terminal_face_id], winding_type="Current",
            is_solid=True, current=1, name=f"{ring.name}_winding",
        )
        winding_names.append(winding.name)

    ind_matrix_name = "InductanceMatrix"
    m3d.assign_matrix(MatrixACMagnetic(
        signal_sources=[SourceACMagnetic(name=n) for n in winding_names],
        matrix_name=ind_matrix_name,
    ))

    induc_setup = m3d.create_setup(
        name="InducEddySetup",
        setup_type="EddyCurrent",
        MaximumPasses=10,
        MinimumPasses=1,
        MinimumConvergedPasses=1,
        PercentError=1,
        UseIterativeSolver=True,
        Frequency=f"{min_freq_kHz}kHz",
        HasSweepSetup=sweep_active,
    )

    if sweep_active:
        induc_setup.add_eddy_current_sweep(
            sweep_type="LinearCount",
            start_frequency=min_freq_kHz,
            stop_frequency=max_freq_kHz,
            step_size=number_of_point,
            units="kHz",
        )
        print("Sweep configured: matrix export/parsing below only covers the nominal "
              "adaptive frequency, use AEDT's report tools for the full swept data.")

    input("Validate the coil terminal placement AND the EddyCurrent (inductance) setup "
          "in AEDT -- check each ring's terminal sheet sits where expected before solving "
          "-- then press Enter to run the simulation...")
    if not m3d.analyze_setup("InducEddySetup"):
        raise RuntimeError(
            "InducEddySetup did not solve successfully -- check AEDT's Message Manager / "
            "Progress window for the actual error before re-running."
        )

    ind_blocks = export_and_parse(m3d, ind_matrix_name, "InducEddySetup")
    save_matrix_like_q3d(
        ind_blocks, keyword="Inductance", mat_name="induc_eddy.mat",
        freq_khz=min_freq_kHz,
    )
    save_matrix_like_q3d(
        ind_blocks, keyword="AC Resistance", mat_name="Resis_eddy.mat",
        freq_khz=min_freq_kHz,
    )


# ============================================================
# Choose which simulation(s) to run
# ============================================================

STAGES = {
    "1": ("Capacitance (Electrostatic)", run_capacitance),
    "2": ("DC Resistance (DC Conduction)", run_dc_resistance),
    "3": ("Inductance + Eddy Current (EddyCurrent solver, closed rings)", run_inductance),
}

if __name__ == "__main__":
    print("Which simulation(s) do you want to run?")
    for key, (label, _) in STAGES.items():
        print(f"  {key}) {label}")

    choice = input("Enter numbers separated by commas (e.g. 1,3), or 'all' [all]: ").strip().lower()

    tokens = list(STAGES.keys()) if choice in ("", "all", "*") else [c.strip() for c in choice.split(",")]
    selected_keys = [t for t in tokens if t in STAGES]
    invalid = [t for t in tokens if t not in STAGES]
    if invalid:
        print(f"[warn] ignoring unrecognized selection(s): {invalid}")
    if not selected_keys:
        raise SystemExit("No valid simulation selected, exiting.")

    print("Running:", ", ".join(STAGES[k][0] for k in selected_keys))
    for key in selected_keys:
        STAGES[key][1]()

# m3d.release_desktop()

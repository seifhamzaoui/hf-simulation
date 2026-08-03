"""
Post-processing utilities for computing derived quantities from the
inductance matrices saved by simulation.py / simulation_Maxwell.py /
simulation_ngsolve.py / simulation_ngsolve_cuda.py (induc.mat,
induc_gpu.mat, induc_eddy.mat, ...).

Every one of those scripts builds its inductance matrix with the same
row/column order: all primary (ringp*) rows first, then all secondary
(rings*) rows -- confirmed for the Q3D path via traitement.py's
natural_sort_key ("Core_V" < "ringp*" < "rings*" alphabetically,
numeric-aware), and by construction for the ngsolve/Maxwell paths (each
builds entry_names/all_ring_names as primary + secondary in that literal
order). This module relies on that ordering rather than re-deriving
conductor names from the .mat file, since none of those save functions
store the names alongside the matrix -- so how many of the N rows are
primary vs. secondary is NOT stored anywhere and can't be inferred from
the matrix itself (it depends on how many rings that particular run
covered, e.g. a full run vs. a test_rings=[...] subset), which is why
every function below takes primary_count/secondary_count as mandatory
arguments rather than defaulting to config.py's full design totals.

Quantities computed:
  - Equivalent self-inductance of the primary bank (all ringp) connected
    in series, and of the secondary bank (all rings) in series.
  - Mutual inductance between the two series-connected banks.
  - Coupling coefficient and leakage inductance (short-circuit-test
    definition) between them.
"""

import sys

import numpy as np
from scipy.io import loadmat


def load_inductance_matrix(mat_path, freq_key=None):
    """Loads an inductance matrix from one of this project's saved .mat
    files. These store {"f_<freq>kHz": matrix, "frequencies_hz": [[..]]},
    with the matrix ALREADY in nH (every save_matrix_like_q3d call for
    inductance uses display_scale=1e9 -- "H -> nH, matching Q3D's stored
    convention" -- so nothing here re-scales it; every quantity this module
    computes/prints is in nH, not SI Henries).
    freq_key picks a specific frequency's matrix when the file has more
    than one (a sweep); defaults to the first one found, with a printed
    note if there were others to choose from."""
    data = loadmat(mat_path)
    keys = [k for k in data if k.startswith("f_")]
    if not keys:
        raise ValueError(f"No 'f_<freq>' matrix key found in {mat_path}")
    if freq_key is None:
        freq_key = keys[0]
        if len(keys) > 1:
            print(f"[info] multiple frequencies in {mat_path}, using '{freq_key}' "
                  f"(pass freq_key= to pick another: {keys})")
    elif freq_key not in keys:
        raise ValueError(f"freq_key '{freq_key}' not found in {mat_path}, available: {keys}")
    return np.asarray(data[freq_key], dtype=float)


def primary_secondary_blocks(L, primary_count, secondary_count):
    """Splits an (N x N) inductance matrix (rows/cols ordered ringp1..Np,
    rings1..Ns) into its primary-primary, secondary-secondary, and
    primary-secondary sub-blocks.

    primary_count/secondary_count are mandatory and must match how THIS
    particular matrix was actually built -- there is no default, since a
    matrix's own size doesn't tell you the split (a Q3D capacitance matrix
    has an extra leading Core row/col; a test_rings=[...] run has fewer
    than the full turn count; config.py's turn totals only describe the
    FULL design, not necessarily whatever subset a given .mat file covers)."""
    N = L.shape[0]
    expected = primary_count + secondary_count
    if N != expected:
        raise ValueError(
            f"matrix is {N}x{N} but primary_count+secondary_count={expected} -- "
            "pass primary_count/secondary_count matching how this particular "
            "matrix was actually built."
        )
    Lpp = L[:primary_count, :primary_count]
    Lss = L[primary_count:, primary_count:]
    Lps = L[:primary_count, primary_count:]
    return Lpp, Lss, Lps


def series_equivalent_inductance(L_block):
    """Equivalent self-inductance of every conductor in L_block connected
    in series (same current through all of them). Each turn sees the same
    current I, so the total flux linkage is the sum over EVERY (i,j) pair
    -- self AND mutual terms -- of L_ij * I: summing the entire submatrix
    (not just its diagonal) gives the series bank's own self-inductance."""
    return float(np.sum(L_block))


def mutual_between_series_windings(Lps_block):
    """Mutual inductance between two windings, each itself a bank of
    conductors connected in series -- same all-pairs-sum logic as
    series_equivalent_inductance, over the cross (primary x secondary)
    block instead of a self block."""
    return float(np.sum(Lps_block))


def coupling_coefficient(L_primary_series, L_secondary_series, M_series):
    return M_series / np.sqrt(L_primary_series * L_secondary_series)


def leakage_inductance(L_primary_series, L_secondary_series, M_series, referred_to="primary"):
    """Two-winding-transformer leakage inductance, from the standard
    short-circuit-test definition: the inductance seen from one winding
    with the other short-circuited.
        referred to primary:   L_leak = L_P - M^2/L_S
        referred to secondary: L_leak = L_S - M^2/L_P
    """
    if referred_to == "primary":
        return L_primary_series - M_series ** 2 / L_secondary_series
    elif referred_to == "secondary":
        return L_secondary_series - M_series ** 2 / L_primary_series
    raise ValueError("referred_to must be 'primary' or 'secondary'")


def summarize_inductance(mat_path, primary_count, secondary_count, freq_key=None):
    """Prints and returns every derived quantity above for one inductance
    matrix file. primary_count/secondary_count are mandatory -- see
    primary_secondary_blocks()."""
    L = load_inductance_matrix(mat_path, freq_key)
    Lpp, Lss, Lps = primary_secondary_blocks(L, primary_count, secondary_count)

    L_P = series_equivalent_inductance(Lpp)
    L_S = series_equivalent_inductance(Lss)
    M = mutual_between_series_windings(Lps)
    k = coupling_coefficient(L_P, L_S, M)
    L_leak_p = leakage_inductance(L_P, L_S, M, referred_to="primary")
    L_leak_s = leakage_inductance(L_P, L_S, M, referred_to="secondary")

    print(f"--- {mat_path} ---")
    print(f"Primary  (ringp1..{primary_count}) in series:  L_P = {L_P:.4f} nH")
    print(f"Secondary (rings1..{secondary_count}) in series: L_S = {L_S:.4f} nH")
    print(f"Mutual primary<->secondary:                     M   = {M:.4f} nH")
    print(f"Coupling coefficient:                           k   = {k:.6f}")
    print(f"Leakage inductance, referred to primary:        L_leak_p = {L_leak_p:.4f} nH")
    print(f"Leakage inductance, referred to secondary:      L_leak_s = {L_leak_s:.4f} nH")
    print()

    return {
        "L_primary_series": L_P,
        "L_secondary_series": L_S,
        "M": M,
        "k": k,
        "L_leakage_primary": L_leak_p,
        "L_leakage_secondary": L_leak_s,
    }


# Known inductance-matrix outputs from this project's simulation scripts --
# only the ones that actually exist on disk are processed when run directly.
KNOWN_INDUCTANCE_FILES = [
    # "./traited Values/induc.mat",       # simulation.py (Q3D)
    # "./maxwell matrices/induc_eddy.mat",  # simulation_Maxwell.py
    "./ngsolve matrices/induc.mat",     # simulation_ngsolve.py (CPU)
    # "./ngsolve matrices/induc_gpu.mat",  # simulation_ngsolve_cuda.py (GPU)
]


if __name__ == "__main__":
    import os

    # primary_count/secondary_count have no default (see primary_secondary_blocks) --
    # a matrix's own size can't tell you the split, so they must be supplied here.
    if len(sys.argv) == 4:
        paths = [sys.argv[1]]
        primary_count, secondary_count = int(sys.argv[2]), int(sys.argv[3])
    else:
        raise SystemExit(
            "Usage: python traitement_inductance.py <mat_path> <primary_count> <secondary_count>\n"
            "  e.g. python traitement_inductance.py \"./ngsolve matrices/induc.mat\" 17 43\n"
            f"Known inductance files on disk: {[p for p in KNOWN_INDUCTANCE_FILES if os.path.exists(p)]}"
        )

    for path in paths:
        try:
            summarize_inductance(path, primary_count, secondary_count)
        except Exception as e:
            print(f"[warn] skipping {path}: {e}\n")

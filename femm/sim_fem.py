"""
FEMM harmonic sweep: extract R and L for each circuit ("primaire", "secondaire")
across a fundamental frequency and its harmonics.

DC (freq = 0) is always simulated once to get Rdc and Ldc for each circuit.
Neither is included in R_matrices / L_matrices; instead they are used to
compute the Rac/Rdc and Lac/Ldc ratios for every harmonic, stored in
Ratio_R_matrices and Ratio_L_matrices.

Requires: Windows + FEMM installed + `pip install pyfemm`
"""
import sys, os

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

import femm
import numpy as np
from scipy.io import savemat
from config import *

# ------------------------- USER SETTINGS -------------------------
FEMM_FILE = "./femm/RL.fem"

F0 = 100000            # fundamental frequency [Hz]
N_HARMONICS = 100      # number of harmonics to run (1 = only the fundamental)



CIRCUITS = ["primaire", "secondaire"]   # order matters: [primary, secondary]

N1 = primary_turns_per_layer * primary_n_layers
N2 = secondary_turns_per_layer * secondary_n_layers

# mi_probdef(frequency, units, type, precision, depth, minangle, acsolver)
UNITS = "millimeters"
PROBTYPE = "planar"     # or "axi"
PRECISION = 1e-8
DEPTH = 1000             # model depth (planar) -- ignored for axi
MINANGLE = 30
ACSOLVER = 0             # 0 = successive approximation, 1 = Newton
# -------------------------------------------------------------------


def freq_key(freq_hz):
    """
    Build a MATLAB-safe key like 'f_1kHz' or 'f_0p5kHz' from a frequency in Hz,
    expressed in kHz.
    """
    freq_khz = freq_hz / 1000.0
    if freq_khz == int(freq_khz):
        val_str = str(int(freq_khz))
    else:
        val_str = f"{freq_khz:g}".replace(".", "p").replace("-", "m")
    return f"f_{val_str}kHz"


def build_diag_matrix(primary_total, secondary_total, N1, N2):
    """
    Split a lumped value (R or L) into N1 identical primary-branch values and
    N2 identical secondary-branch values, stack as [Xp ; Xs], and return the
    diagonal matrix.

    Returns:
        Xp        : (N1,) vector, each entry = primary_total / N1
        Xs        : (N2,) vector, each entry = secondary_total / N2
        X_vector  : (N1+N2,) concatenated vector [Xp, Xs]
        X_matrix  : (N1+N2, N1+N2) diagonal matrix of X_vector
    """
    Xp = np.full(N1, primary_total / N1)
    Xs = np.full(N2, secondary_total / N2)
    X_vector = np.concatenate([Xp, Xs])
    X_matrix = np.diag(X_vector)
    return Xp, Xs, X_vector, X_matrix


def build_ratio_diag_matrix(ratio_primary, ratio_secondary, N1, N2):
    """
    Build an Xac/Xdc ratio diagonal matrix (works for R or L). The ratio is
    NOT divided by N1/N2 -- every branch of a given circuit shares the same
    ratio (since Xp_ac/Xp_dc = (X_ac/N)/(X_dc/N) = X_ac/X_dc).
    """
    ratio_p = np.full(N1, ratio_primary)
    ratio_s = np.full(N2, ratio_secondary)
    ratio_vector = np.concatenate([ratio_p, ratio_s])
    ratio_matrix = np.diag(ratio_vector)
    return ratio_matrix


def get_dc_values():
    """
    Run the DC (freq = 0) case once and extract Rdc and Ldc for each circuit.
    Not stored in R_matrices/L_matrices -- only used as the ratio reference.

    Rdc  : real part of V/I (purely resistive at DC)
    Ldc  : flux_linkage / I (static/DC inductance, no eddy effects at freq=0)
    """
    femm.mi_probdef(0, UNITS, PROBTYPE, PRECISION, DEPTH, MINANGLE, ACSOLVER)
    femm.mi_analyze(1)
    femm.mi_loadsolution()

    R_dc, L_dc = {}, {}
    for c in CIRCUITS:
        I, V, flux = femm.mo_getcircuitproperties(c)
        if I == 0:
            R_dc[c] = None
            L_dc[c] = None
        else:
            R_dc[c] = (V / I).real
            L_dc[c] = (flux / I).real if hasattr(flux, "real") else flux / I
    return R_dc, L_dc


def run_sweep():
    femm.openfemm()
    femm.opendocument(FEMM_FILE)

    # --- Always run the DC point first, used only as the ratio reference ---
    R_dc, L_dc = get_dc_values()
    print(f"Rdc primary   : {R_dc['primaire']}")
    print(f"Rdc secondary : {R_dc['secondaire']}")
    print(f"Ldc primary   : {L_dc['primaire']}")
    print(f"Ldc secondary : {L_dc['secondaire']}")

    results = {c: [] for c in CIRCUITS}
    R_matrices = {}         # keyed by "f_{freq_kHz}kHz" -> (N1+N2, N1+N2) ndarray
    L_matrices = {}         # keyed by "f_{freq_kHz}kHz" -> (N1+N2, N1+N2) ndarray
    Ratio_R_matrices = {}   # keyed by "f_{freq_kHz}kHz" -> Rac/Rdc diagonal matrix
    Ratio_L_matrices = {}   # keyed by "f_{freq_kHz}kHz" -> Lac/Ldc diagonal matrix

    for freq in sim_frequencies:
        # freq = h * F0

        femm.mi_probdef(freq, UNITS, PROBTYPE, PRECISION, DEPTH, MINANGLE, ACSOLVER)
        femm.mi_analyze(1)      # 1 = run solver silently in background
        femm.mi_loadsolution()

        R_by_circuit = {}
        L_by_circuit = {}

        for c in CIRCUITS:
            I, V, flux = femm.mo_getcircuitproperties(c)

            if I == 0:
                R, L = None, None
            else:
                Z = V / I                # complex impedance (R + jX)
                omega = 2 * np.pi * freq
                R = Z.real
                L = Z.imag / omega

            R_by_circuit[c] = R
            L_by_circuit[c] = L

            results[c].append({
                "harmonic": freq,
                "freq_Hz": freq,
                "I": I,
                "V": V,
                "flux_linkage": flux,
                "R_ohm": R,
                "L_H": L,
            })

        R_primary_total = R_by_circuit["primaire"]
        R_secondary_total = R_by_circuit["secondaire"]
        L_primary_total = L_by_circuit["primaire"]
        L_secondary_total = L_by_circuit["secondaire"]

        if R_primary_total is None or R_secondary_total is None:
            continue

        key = freq_key(freq)

        # [Rp, Rs] and [Lp, Ls] diagonal matrices for this frequency
        _, _, _, R_matrix = build_diag_matrix(
            R_primary_total, R_secondary_total, N1, N2
        )
        R_matrices[key] = R_matrix

        _, _, _, L_matrix = build_diag_matrix(
            L_primary_total, L_secondary_total, N1, N2
        )
        L_matrices[key] = L_matrix

        # Rac/Rdc ratio diagonal matrix for this frequency
        if R_dc["primaire"] and R_dc["secondaire"]:
            ratio_R_primary = R_primary_total / R_dc["primaire"]
            ratio_R_secondary = R_secondary_total / R_dc["secondaire"]
            Ratio_R_matrices[key] = build_ratio_diag_matrix(
                ratio_R_primary, ratio_R_secondary, N1, N2
            )

        # Lac/Ldc ratio diagonal matrix for this frequency
        if L_dc["primaire"] and L_dc["secondaire"]:
            ratio_L_primary = L_primary_total / L_dc["primaire"]
            ratio_L_secondary = L_secondary_total / L_dc["secondaire"]
            Ratio_L_matrices[key] = build_ratio_diag_matrix(
                ratio_L_primary, ratio_L_secondary, N1, N2
            )

    femm.mo_close()
    femm.mi_close()
    femm.closefemm()
    return (results, R_matrices, L_matrices,
            Ratio_R_matrices, Ratio_L_matrices, R_dc, L_dc)


def print_and_save(results, R_matrices, L_matrices,
                    Ratio_R_matrices, Ratio_L_matrices, R_dc, L_dc):
    for c, rows in results.items():
        print(f"\n--- {c} ---")
        for row in rows:
            print(row)

    print("\n--- Rac/Rdc ratio matrices, keyed by frequency ---")
    for key, ratio_matrix in Ratio_R_matrices.items():
        print(f"\n{key}:")
        print(ratio_matrix)

    print("\n--- Lac/Ldc ratio matrices, keyed by frequency ---")
    for key, ratio_matrix in Ratio_L_matrices.items():
        print(f"\n{key}:")
        print(ratio_matrix)

    # frequencies_hz array shared by both files, matching the target format
    frequencies_hz = np.array([[h * F0 for h in range(1, N_HARMONICS + 1)]])

    # ---- Flat dict for R_ratio.mat: top-level "f_XkHz" keys + frequencies_hz ----
    R_ratio_dict = dict(Ratio_R_matrices)
    R_ratio_dict["frequencies_hz"] = frequencies_hz
    savemat("./femm/R_ratio.mat", R_ratio_dict)

    # ---- Flat dict for L_Ratio.mat: top-level "f_XkHz" keys + frequencies_hz ----
    L_ratio_dict = dict(Ratio_L_matrices)
    L_ratio_dict["frequencies_hz"] = frequencies_hz
    savemat("./femm/L_Ratio.mat", L_ratio_dict)

    print("\nSaved: ./femm/R_ratio.mat")
    print("Saved: ./femm/L_Ratio.mat")


if __name__ == "__main__":
    (res, R_mats, L_mats,
     Ratio_R_mats, Ratio_L_mats, R_dc, L_dc) = run_sweep()
    print_and_save(res, R_mats, L_mats, Ratio_R_mats, Ratio_L_mats, R_dc, L_dc)
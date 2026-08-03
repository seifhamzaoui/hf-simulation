
import numpy as np
import pandas as pd
import os
from ansys.aedt.core import Q3d
from config import *
import fnmatch
import numpy as np
import pandas as pd
import re
import pickle
from scipy.io import *


# capa = loadmat("./traited Values/cap_data.mat")
# print(capa)

# Induc = loadmat("./traited Values/Induc.mat")
# print(Induc)

# Rssis = loadmat("./traited Values/DCR.mat")
# print(Rssis)

def natural_sort_key(s):
    """Split a string into text/number chunks so 'ringp10' sorts after 'ringp9', not after 'ringp1'."""
    return [int(chunk) if chunk.isdigit() else chunk.lower() for chunk in re.split(r'(\d+)', s)]



def format_freq_key(freq):
    s = f"{freq:.3e}"
    mantissa, exponent = s.split("e")
    mantissa = mantissa.rstrip("0").rstrip(".")
    return f"{mantissa}e{int(exponent)}"


def extract_matrix_dict(q3d, quantity_symbol, setup_sweep_name, quantities_category,
                         matrix_context="Original", apply_reduction=False, target_freq_unit="kHz"):
    """
    Returns:
        freq_dict   : {"<freq_label>": NxN matrix, ...}
        net_names   : list of net names (row/col order)
        freqs_hz    : numpy array of the actual frequency values, in Hz (ground truth,
                       use this for anything precision-sensitive instead of the string keys)
    """
    expressions = q3d.post.available_report_quantities(
        report_category="Matrix",
        quantities_category=quantities_category,
        context=matrix_context,
    )
    if not expressions:
        raise RuntimeError(f"No expressions found for quantities_category='{quantities_category}', context='{matrix_context}'.")

    solution_data = q3d.post.get_solution_data(
        expressions=expressions,
        setup_sweep_name=setup_sweep_name,
        report_category="Matrix",
        context=matrix_context,
    )
    # solution_data.enable_pandas_output = True
    real_dict, imag_dict = solution_data.full_matrix_real_imag

    # --- Get the REAL unit and convert to Hz (ground truth) ---
    raw_unit = solution_data.units_sweeps.get("Freq", "GHz")
    unit_to_hz = {"Hz": 1, "kHz": 1e3, "MHz": 1e6, "GHz": 1e9, "THz": 1e12}
    if raw_unit not in unit_to_hz:
        raise RuntimeError(f"Unrecognized frequency unit '{raw_unit}' — check solution_data.units_sweeps.")

    pattern = rf"{quantity_symbol}\((.*?)\)"
    
    # net_names = sorted(set(
    #     n.strip()
    #     for expr in real_dict.keys()
    #     for n in re.findall(pattern, expr)[0].split(",")
    # ))
    net_names = sorted(set(
        n.strip()
        for expr in real_dict.keys()
        for n in re.findall(pattern, expr)[0].split(",")
    ), key=natural_sort_key)
    n_nets = len(net_names)

    sample_expr = list(real_dict.keys())[0]
    freqs_raw = real_dict[sample_expr][:, 0]          # in raw_unit (e.g. GHz)
    freqs_hz = freqs_raw * unit_to_hz[raw_unit]        # ground-truth Hz values
    n_freq = len(freqs_hz)

    def label_for(freq_hz):
        val = freq_hz / unit_to_hz[target_freq_unit]
        return f"{val:.6g}{target_freq_unit}"

    freq_dict = {}
    for f_idx in range(n_freq):
        raw_matrix = np.zeros((n_nets, n_nets))
        for expr, arr in real_dict.items():
            match = re.findall(pattern, expr)
            if not match:
                continue
            ni, nj = [x.strip() for x in match[0].split(",")]
            i, j = net_names.index(ni), net_names.index(nj)
            val = arr[f_idx, 1]
            raw_matrix[i, j] = val
            raw_matrix[j, i] = val

        if apply_reduction:
            first_row = raw_matrix[0, 1:]
            reduced = raw_matrix[1:, 1:].copy()
            np.fill_diagonal(reduced, first_row)
            matrix = np.abs(reduced)
            # matrix = reduced
        else:
            matrix = np.abs(raw_matrix)

        freq_dict[label_for(freqs_hz[f_idx])] = matrix

    return freq_dict, net_names, freqs_hz
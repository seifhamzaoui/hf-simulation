import numpy as np
import matplotlib.pyplot as plt
from scipy.special import jv
from scipy.io import loadmat
from config import *
import math

def create_km(m):
    """
    Creates the k_m matrix of shape (m, m-1) as defined in Eq. 36.
    It features 1s on the main diagonal and -1s on the sub-diagonal.
    """
    # Create an (m x m-1) matrix with 1s on the main diagonal (k=0)
    main_diag = np.eye(m, m - 1, k=0)
    
    # Create an (m x m-1) matrix with 1s on the sub-diagonal (k=-1)
    sub_diag = np.eye(m, m - 1, k=-1)
    
    # Subtract to get the 1 and -1 pattern
    k_m = main_diag - sub_diag
    return k_m

def create_K(N1, N2):
    
    # Generate the k_m sub-matrices
    k_N1 = create_km(N1)
    k_N2 = create_km(N2)
    
    # Generate the zero-matrices with correct dimensions
    zero_top_right = np.zeros((N1, N2 - 1))
    zero_bottom_left = np.zeros((N2, N1 - 1))
    
    # Assemble the final block matrix K
    K = np.block([
        [k_N1,             zero_top_right],
        [zero_bottom_left, k_N2          ]
    ])
    
    return K

def create_CH(C, N1, N2):
    """
    Constructs the CH matrix from the capacitance matrix C as defined in Eq. 40.
    C is assumed to be an N x N matrix where N = N1 + N2.
    """
    N = N1 + N2
    
    # 1. Identify the rows to extract from C (using 0-based indexing)
    # The image skips row N1 and row N (which are indices N1-1 and N-1)
    all_row_indices = np.arange(N)
    extract_rows = np.delete(all_row_indices, [N1 - 1, N - 1])
    
    # Initialize an (N-2) x 4 matrix with zeros
    CH = np.zeros((N - 2, 4))
    
    # 2. Populate the 2nd column (index 1) 
    # Using the N1-th column of C (index N1 - 1)
    CH[:, 1] = C[extract_rows, N1 - 1]
    
    # 3. Populate the 4th column (index 3)
    # Using the N-th column of C (index N - 1)
    CH[:, 3] = C[extract_rows, N - 1]
    
    # Columns 0 and 2 remain zeros as initialized
    return abs(CH)

def create_H(N1, N2):
    """
    Constructs the sparse matrix H as defined in Eq. 37.
    Assumes total rows N = N1 + N2.
    """
    N = N1 + N2
    
    # Initialize an N x 4 matrix with zeros
    H = np.zeros((N, 4))
    
    # Set the specific nonzero entries using 0-based indexing
    H[0, 0] = 1         # corresponds to row 1
    H[N1 - 1, 1] = -1   # corresponds to row N1
    H[N1, 2] = 1        # corresponds to row N1 + 1
    H[N - 1, 3] = -1    # corresponds to row N
    
    return H

def create_CK(C, N1, N2):
    """
    Constructs the CK matrix from the capacitance matrix C as defined in Eq. 39.
    C is assumed to be an N x N matrix where N = N1 + N2.
    """
    N = N1 + N2
    
    # 1. Identify the row/column indices to retain (0-based indexing)
    # The image skips indices N1-1 and N-1
    keep_indices = np.delete(np.arange(N), [N1 - 1, N - 1])
    
    # 2. Calculate the diagonal values 
    # Sum across all columns (axis=1) of the original C matrix for the kept rows
    diag_values = np.sum(C[keep_indices, :], axis=1)
    
    # 3. Extract the submatrix and negate it for the off-diagonal entries
    # np.ix_ allows us to safely extract a subgrid by retaining specific rows and columns
    CK = -C[np.ix_(keep_indices, keep_indices)]
    
    # 4. Overwrite the main diagonal with the calculated sums
    np.fill_diagonal(CK, diag_values)
    
    return CK


# ==========================================
# Turn-number <-> array-index mapping
# ==========================================
# The N=N1+N2 rows/columns of C/L/R (and the N-length `i` branch-current
# vector solve_system_spectrum returns) are ordered turn-by-turn: primary
# turns 1..N1 first (0-based indices 0..N1-1), then secondary turns
# 1..N2 (0-based indices N1..N-1) -- matching DCR.mat/induc.mat/
# cap_data.mat's own ring ordering. The (N-2)-length `v` node-voltage
# vector is the SAME numbering with exactly two entries removed (see
# create_CK's keep_indices above: original indices N1-1 and N-1, the
# HV,2 and LV,2 terminal nodes, held at this model's reference
# potential rather than solved as unknowns) -- these helpers translate a
# human turn number into the right index for each vector, including that
# shift, instead of hand-picking raw array indices per plot.

def turn_original_index(winding, turn_number, N1, N2):
    """0-based index into the full N=N1+N2 turn numbering for a 1-based
    turn_number within `winding` ('primary' or 'secondary')."""
    if winding == "primary":
        if not (1 <= turn_number <= N1):
            raise ValueError(f"primary turn_number must be in [1,{N1}], got {turn_number}")
        return turn_number - 1
    elif winding == "secondary":
        if not (1 <= turn_number <= N2):
            raise ValueError(f"secondary turn_number must be in [1,{N2}], got {turn_number}")
        return N1 + turn_number - 1
    raise ValueError(f"winding must be 'primary' or 'secondary', got {winding!r}")


def turn_current_index(winding, turn_number, N1, N2):
    """Index into the N-length branch-current vector `i` -- i is NOT
    reduced (create_K/create_H keep all N rows), so this is just the
    plain turn numbering."""
    return turn_original_index(winding, turn_number, N1, N2)


def turn_voltage_index(winding, turn_number, N1, N2):
    """Index into the (N-2)-length node-voltage vector `v`, or None if
    this turn's node is one of the two reference nodes excluded from the
    unknown vector (see module note above) -- callers should treat that
    case as 0V by this model's own convention."""
    N = N1 + N2
    orig = turn_original_index(winding, turn_number, N1, N2)
    removed = (N1 - 1, N - 1)
    if orig in removed:
        return None
    return orig - sum(1 for r in removed if r < orig)


def get_turn_voltage(v_array, winding, turn_number, N1, N2):
    """v_array: an (N-2, ...) array indexable along its first axis, e.g.
    reconstruct_time_domain()'s time-domain output -- returns that turn's
    own trace (all-zero if it's one of the two reference nodes)."""
    idx = turn_voltage_index(winding, turn_number, N1, N2)
    if idx is None:
        return np.zeros_like(np.asarray(v_array[0]))
    return v_array[idx]


def get_turn_voltage_freq(v_freq_list, winding, turn_number, N1, N2):
    """v_freq_list: the list of (N-2,1) complex column vectors
    solve_system_spectrum() returns (one per swept frequency) -- returns
    a 1D complex array of that turn's voltage phasor across the sweep
    (all-zero if it's one of the two reference nodes)."""
    idx = turn_voltage_index(winding, turn_number, N1, N2)
    if idx is None:
        return np.zeros(len(v_freq_list), dtype=complex)
    return np.array([v[idx, 0] for v in v_freq_list])


def build_system_matrix(K, R, L, C, omega,G_K):
    """
    Builds the block matrix:
        [  K        R + jωL ]
        [  jωC      -K.T    ]

    K : (n_branches, n_nodes) topology/incidence matrix
    R, L : (n_branches, n_branches) resistance/inductance matrices
    C : (n_nodes, n_nodes) capacitance matrix
    omega : angular frequency (2*pi*f), in rad/s
    """
    

    top_right = R + 1j * omega * L
    bottom_left = 1j * omega * C + G_K
    bottom_right = -K.T

    M = np.block([
        [K,          top_right],
        [bottom_left, bottom_right]
    ])
    return M
# ==========================================
# 1. System Matrix Construction
# ==========================================

# ==========================================
# 2. Fourier Series & Input Generation
# ==========================================
def get_fourier_components(t, A, T, tr, num_harmonics):
    """Calculates frequencies, amplitudes, and the time-domain waveform of a trapezoidal wave."""
    f0 = 1.0 / T
    omega_0 = 2 * np.pi * f0
    
    # Initialize an array of zeros with the same shape as t
    waveform = np.zeros_like(t) 
    frequencies = []
    amplitudes = []
    
    for n in range(1, num_harmonics * 2, 2):
        fn = n * f0
        ideal_coeff = (4 * A) / (np.pi * n)
        
        attenuation = 1.0
        if tr != 0:
            x = (n * np.pi * tr) / T
            attenuation = np.sin(x) / x
            
        actual_amplitude = ideal_coeff * attenuation
        
        frequencies.append(fn)
        amplitudes.append(actual_amplitude)
        
        # Build the time-domain waveform by accumulating each harmonic
        waveform += actual_amplitude * np.sin(n * omega_0 * t)
        
    return waveform, np.array(frequencies), np.array(amplitudes)

def construct_vg_vectors(amplitudes, excitation_nodes):
    """Maps harmonic amplitudes to the 4x1 vg column vectors."""
    vg_list = []
    exc_vector = np.array(excitation_nodes, dtype=complex).reshape(4, 1)
    for amp in amplitudes:
        vg_list.append(exc_vector * amp)
    return vg_list

# ==========================================
# 3. Frequency Domain Solver
# ==========================================
DEFAULT_R_RATIO_PATH = "./ngsolve matrices/R_ratio_ngsolve.mat"
DEFAULT_L_RATIO_PATH = "./ngsolve matrices/L_ratio_ngsolve.mat"

Resistance_ratio = None
Inductance_ratio = None


def load_ratio_matrices(r_ratio_path=DEFAULT_R_RATIO_PATH, l_ratio_path=DEFAULT_L_RATIO_PATH):
    """(Re)loads the AC/DC ratio matrices solve_system_spectrum() reads
    from, into this module's own globals -- call this to point at a
    different R_ratio/L_ratio .mat file (e.g. from a UI file picker)
    before calling solve_system_spectrum(). Runs once at import with the
    defaults above so existing callers (HF_model.py) keep working
    unmodified."""
    global Resistance_ratio, Inductance_ratio
    Resistance_ratio = loadmat(r_ratio_path)
    Inductance_ratio = loadmat(l_ratio_path)
    return Resistance_ratio, Inductance_ratio


load_ratio_matrices()


def solve_system_spectrum(frequencies, vg_list, C_K, K, C_H, H, R,L,G_K=None,G_H=None):
    """Solves the matrix equations iteratively for each harmonic."""
    v_responses = []
    i_responses = []

    for freq, vg in zip(frequencies, vg_list):
        omega = 2 * np.pi * freq
        extracted_freq = nearest_frequency(freq, sim_frequencies)


        R_ratio = Resistance_ratio[f'f_{int(extracted_freq/1000)}kHz']

        # NOTE: this used to be loaded then immediately overwritten with
        # L_ratio = 1 (a debug leftover that silently discarded
        # Inductance_ratio entirely, making L_ratio_ngsolve.mat a no-op
        # below) -- restored to actually apply the loaded matrix, since a
        # UI control to pick this file is pointless otherwise.
        L_ratio = Inductance_ratio[f'f_{int(extracted_freq/1000)}kHz']
        # R = np.array(R)
        # R[:N1] *= calculate_Fr(ds = 0.012,f = freq,sigma = 59e6,mu_r = 1)

        # # 2. Multiply rows FROM N2 to the end (includes row N2)
        # R[N1+1:] *= calculate_Fr(ds = 0.007,f = freq,sigma = 59e6,mu_r = 1)
        # calculate_Fr(ds = 0.007,f = freq,sigma = 59e6,mu_r = 1)*
       
        # calculate_Fr(ds = 0.004,f = freq,sigma = 58e6,mu_r = 1)*
        # Z = R + 1j*L*omega #------------------------------- freq/100*
        # Z_inv = np.linalg.inv(Z)
        
        # Calculate v (N-2 x 1) and i (N x 1) exactly as requested
        # term1 = (1j * omega * C_K) + (K_T @ Z_inv @ K)
        # term2 = (1j * omega * C_H) + (K_T @ Z_inv @ H)
        # v = term1 @ term2 @ vg
        # i = Z_inv @ ((H @ vg) - (K @ v))
        # print(build_system_matrix(K,R,L,C_K,omega))
        
        # solution_vector = np.vstack((v, i))
        # is_verified = np.allclose(calculated_rhs, rhs_vector)
        # calculated_rhs = system_matrix @ solution_vector
        # if G_K == None or G_H == None :
        G_K=omega*C_K*(0.005+(0.060-0.045)/(2e6-1e6)*(freq))
        G_H=omega*C_H*(0.005+(0.060-0.045)/(2e6-1e6)*(freq))

        # G_K=omega*C_K*(0.05)
        # G_H=omega*C_H*(0.05)

        rhs_top = H @ vg
        rhs_bottom = ( 1j * omega * C_H + G_H ) @ vg
        rhs_vector = np.vstack((rhs_top, rhs_bottom))
        
        # calculate_Fr(ds = 0.004,f = freq,sigma = 58e6,mu_r = 1)
        system_matrix = build_system_matrix(K,R_ratio*R,L_ratio*L,C_K,omega,G_K)
        system_matrix_inv = np.linalg.inv(system_matrix)
        
    
        
        N_v  =len(R[0]) -2
        solution_vector = system_matrix_inv @ rhs_vector

        v = solution_vector[:N_v, :]  # Takes everything from the top down to N_v
        i = solution_vector[N_v:, :]
        v_responses.append(v)
        i_responses.append(i)
    return v_responses, i_responses

# ==========================================
# 4. Time-Domain Reconstruction
# ==========================================
def reconstruct_time_domain(t, frequencies, phasor_responses):
    """Converts frequency-domain complex phasors back to time-domain."""
    num_nodes = phasor_responses[0].shape[0]
    global_time_response = np.zeros((num_nodes, len(t)))
    for freq, phasor in zip(frequencies, phasor_responses):
        omega = 2 * np.pi * freq
        # v(t) = Re(V)*sin(wt) + Im(V)*cos(wt)
        harmonic_time_wave = phasor.real * np.sin(omega * t) + phasor.imag * np.cos(omega * t)
        global_time_response += harmonic_time_wave
    return global_time_response




def plot_vector_ratio(freqs, vec_num, vec_den, title="Frequency Vector Ratio",):
    """
    Calculates and plots the magnitude and phase ratio of two frequency-domain vectors.
    
    Parameters:
    freqs   : 1D array of frequencies (Hz).
    vec_num : 1D array of complex values (Numerator).
    vec_den : 1D array of complex values (Denominator).
    title   : Title for the plot.
    """
    # 1. Ensure inputs are numpy arrays
    f = np.array(freqs)
    num = np.array(vec_num)
    den = np.array(vec_den)
    
    # 2. Prevent division by zero
    epsilon = 1e-15
    den_safe = np.where(den == 0, epsilon, den)
    
    # 3. Calculate complex ratio, magnitude, and phase
    ratio = num / den_safe
    mag = np.abs(ratio)
    phase = np.angle(ratio, deg=True)
    
    # --- 4. Plotting ---
   
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    fig.suptitle(title, fontsize=14)
    
    # Magnitude Subplot (Log frequency scale)
    ax1.loglog(f, mag, color='blue', linewidth=1.5)
    ax1.set_ylabel("Magnitude (|Num / Den|)")
    ax1.set_title("Magnitude Response")
    ax1.grid(False, which="both", ls="--")
    
    # Phase Subplot (Log frequency scale)
    ax2.semilogx(f, phase, color='red', linewidth=1.5)
    ax2.set_ylabel("Phase (Degrees)")
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_title("Phase Response")
    ax2.grid(False, which="both", ls="--")
    
    plt.tight_layout()
    # plt.show()
    
    return ratio, mag, phase


def calculate_Fr(ds, f, sigma, mu_r=1.0):
    """
    Calculates the Froude/Resistance factor Fr based on skin depth and Kelvin functions.
    
    Parameters:
    ds    : float - Conductor diameter (m)
    f     : array_like - Frequency or array of frequencies (Hz)
    sigma : float - Electrical conductivity (S/m)
    mu_r  : float - Relative permeability (default is 1.0 for copper)
    
    Returns:
    Fr    : ndarray - Calculated factor (same size as f)
    """
    # 1. Convert input to numpy array for vectorized operations
    f = np.asarray(f)
    
    # Optional: Prevent division by zero warnings if a DC frequency (0 Hz) is passed
    f_safe = np.where(f == 0, 1e-15, f)
    
    # 2. Physical Constants
    mu_0 = 4 * np.pi * 1e-7  # Vacuum permeability (H/m)
    
    # 3. Calculate Skin Depth (delta)
    delta = 1 / np.sqrt(np.pi * mu_0 * mu_r * sigma * f_safe)
    
    # 4. Calculate Xi
    xi = ds / (np.sqrt(2) * delta)
    
    # 5. Calculate Kelvin Functions (ber and bei)
    # Z = xi * exp(j * 3*pi / 4)
    Z = xi * np.exp(1j * 3 * np.pi / 4)
    
    # Calculate Bessel functions of order 0 and 1
    J0 = jv(0, Z)
    J1 = jv(1, Z)
    
    # Extract real (ber) and imaginary (bei) parts
    ber0 = np.real(J0)
    bei0 = np.imag(J0)
    ber1 = np.real(J1)
    bei1 = np.imag(J1)
    
    # 6. Calculate Fr
    # Pre-calculate the common denominator
    denom = ber1**2 + bei1**2
    
    # Calculate the two main terms in the parentheses
    term1 = (ber0 * (bei1 - ber1)) / denom
    term2 = (bei0 * (bei1 + ber1)) / denom
    
    # Final assembly
    Fr = (xi / (4 * np.sqrt(2))) * (term1 - term2)
    
    # Force DC (0 Hz) result to theoretical limit if 0 was in the array
    Fr = np.where(f == 0, 0, Fr)
    
    return Fr

def calculate_Gr(ds, f, sigma, mu_r=1.0):
    """
    Calculates the Gr factor based on skin depth and Kelvin functions.
    
    Parameters:
    ds    : float - Conductor diameter (m)
    f     : array_like - Frequency or array of frequencies (Hz)
    sigma : float - Electrical conductivity (S/m)
    mu_r  : float - Relative permeability (default is 1.0 for copper)
    
    Returns:
    Gr    : ndarray - Calculated factor (same size as f)
    """
    # 1. Convert input to numpy array for vectorized operations
    f = np.asarray(f)
    
    # Optional: Prevent division by zero warnings if a DC frequency (0 Hz) is passed
    f_safe = np.where(f == 0, 1e-15, f)
    
    # 2. Physical Constants
    mu_0 = 4 * np.pi * 1e-7  # Vacuum permeability (H/m)
    
    # 3. Calculate Skin Depth (delta)
    delta = 1 / np.sqrt(np.pi * mu_0 * mu_r * sigma * f_safe)
    
    # 4. Calculate Xi
    xi = ds / (np.sqrt(2) * delta)
    
    # 5. Calculate Kelvin Functions (ber and bei)
    # Z = xi * exp(j * 3*pi / 4)
    Z = xi * np.exp(1j * 3 * np.pi / 4)
    
    # Calculate Bessel functions of order 0, 1, and 2
    J0 = jv(0, Z)
    J1 = jv(1, Z)
    J2 = jv(2, Z)
    
    # Extract real (ber) and imaginary (bei) parts
    ber0 = np.real(J0)
    bei0 = np.imag(J0)
    ber1 = np.real(J1)
    bei1 = np.imag(J1)
    ber2 = np.real(J2)
    bei2 = np.imag(J2)
    
    # 6. Calculate Gr
    # Pre-calculate the common denominator (Note: Gr uses order 0 for the denominator)
    denom = ber0**2 + bei0**2
    
    # Calculate the two main terms in the parentheses
    term1 = (ber2 * (ber1 + bei1)) / denom
    term2 = (bei2 * (ber1 - bei1)) / denom
    
    # Calculate the multiplication coefficient
    coeff = - (xi * np.pi**2 * ds**2) / (2 * np.sqrt(2))
    
    # Final assembly
    Gr = coeff * (term1 - term2)
    
    # Force DC (0 Hz) result to theoretical limit if 0 was in the array
    Gr = np.where(f == 0, 0, Gr)
    
    return Gr

def get_flat_spectrum(A, T, num_harmonics):
    """
    Calculates frequencies and constant amplitudes based on period T.
    """
    f0 = 1.0 / T
    frequencies = []
    amplitudes = []
    
    for n in range(1, num_harmonics + 1):
        fn = n * f0
        frequencies.append(fn)
        amplitudes.append(A)
        
    return np.array(frequencies), np.array(amplitudes)

def multiply_non_diagonal(matrix, multiplier):
    # Create a copy so we don't modify the original matrix
    result = matrix.copy()
    
    # Create a boolean mask where True represents non-diagonal elements
    # np.eye creates a diagonal of 1s (True), and the '~' operator inverts it
    mask = ~np.eye(result.shape[0], result.shape[1], dtype=bool)
    
    # Multiply only the elements located at the True positions in the mask
    result[mask] *= multiplier
    
    return result


def nearest_frequency(freq, frequencies):
    return min(frequencies, key=lambda f: abs(f - freq))
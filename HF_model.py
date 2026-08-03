from scipy.io import loadmat
import numpy as np
from config import *
from SWF import *


capa = loadmat("./ngsolve matrices/cap_data.mat")
induc = loadmat("./ngsolve matrices/induc.mat")
resis = loadmat("./ngsolve matrices/DCR.mat")

for freq_key in capa.keys():
    if freq_key.startswith("f_"):
        C = capa[freq_key]
        C = C  * 1e-12
        # C = multiply_non_diagonal(C, 1.4) 
        L = induc[freq_key]
        L = L*1e-9*1.5
        R = resis[freq_key]
        R = R
        # R = np.diag(np.diag(R))
        G = None
        print(C)
        # print(L)
        # print(R)
        # K = create_K(primary_turns_per_layer*primary_n_layers, secondary_turns_per_layer*secondary_n_layers)
        # Ck = create_CK(C,primary_turns_per_layer*primary_n_layers, secondary_turns_per_layer*secondary_n_layers)
        
        # omega = 1000 * 2 * 3.14

        # print(Ck)
        # print(build_system_matrix(K,R,L,Ck,omega));

        # build_system_matrix(K, R, L, C, omega)
        

if __name__ == "__main__":
    print("--- Starting Full Pipeline ---")
    
    # --- 5a. Define Dimensions ---
    N1 = primary_turns_per_layer*primary_n_layers
    N2 = secondary_turns_per_layer*secondary_n_layers
    N = N1 + N2
    
    # --- 5b. Generate Base Matrices ---
    print(f"Generating system matrices for N1={N1}, N2={N2}...")
    K = create_K(N1, N2)
    H = create_H(N1, N2)
    C_K = create_CK(C, N1, N2)
    # G_k = create_CK(G,N1,N2)
    C_H = create_CH(C, N1, N2)
    # G_H = create_CH(G,N1,N2)
    
    
    # --- 5c. Signal Definitions ---
    A = 1000.0          # 100V Peak
    T = 1/10000           # 20ms period (50 Hz fundamental)
    tr = 0      # 2ms rise time
    num_harm = 500      # Calculate 50 harmonics
    # Define the time array FIRST so we can pass it to the function
    t = np.linspace(0, 1 * T, 100000) 

    
    
    
# Mesure de l'impédance
    freqs, amps =  get_flat_spectrum(A, T, 1000)
   
    print("Constructing Excitation Vectors (Applied to HV,1)...")
    vg_list = construct_vg_vectors(amps, excitation_nodes=[0.85, 0, 2.2, 0])
    # Extract Fourier and build excitation vectors (Applying 100V wave to HV,1)--------------------------------------------------------------
    print("Extracting Fourier Components...")
    # print(vg_list)
    
    # --- 5d. Solve Frequency Spectrum ---
    print("Solving Frequency-Domain equations across spectrum...")
    v_freq_res, i_freq_res = solve_system_spectrum(freqs, vg_list, C_K, K, C_H, H, R,L)
    
    midpoint = len(freqs)

    plot_vector_ratio(freqs[:midpoint], np.array(vg_list)[:midpoint, 2, 0], np.array(i_freq_res)[:midpoint, 17, 0], title="Impédance CC vue au secondaire")
    
    plot_vector_ratio(freqs[:midpoint], np.array(i_freq_res)[:midpoint, 0, 0], np.array(i_freq_res)[:midpoint, 17, 0], title="Rapport de transformation")
    
# SOlution de la tension---------------------------------------------------------------------------------------------------------------------------------------------
    # Extract Fourier components AND the original waveform
    print("Extracting Fourier Components and generating waveform...")
    original_waveform, freqs, amps = get_fourier_components(t, A, T, tr, num_harm)
   
    print("Constructing Excitation Vectors (Applied to HV,1)...")
    vg_list = construct_vg_vectors(amps, excitation_nodes=[0.869, 0, 2.2, 0])
    # Extract Fourier and build excitation vectors (Applying 100V wave to HV,1)--------------------------------------------------------------
    print("Extracting Fourier Components...")
    # print(vg_list)
    
    # --- 5d. Solve Frequency Spectrum ---
    print("Solving Frequency-Domain equations across spectrum...")
    v_freq_res, i_freq_res = solve_system_spectrum(freqs, vg_list, C_K, K, C_H, H, R,L)
    plot_vector_ratio(freqs[:midpoint], np.array(vg_list)[:midpoint, 2, 0]/max(np.array(vg_list)[:midpoint, 2, 0]), 1, title="contenue fréquenciel de l'onde de tension")
    plot_vector_ratio(freqs[:midpoint], np.array(v_freq_res)[:midpoint, 17, 0] - np.array(v_freq_res)[:midpoint, 18, 0], np.array(vg_list)[:midpoint, 2, 0], title="amplification factor (Vs1-Vs2)/Vhv")
    # plt.show()
    # --- 5e. Reconstruct in Time Domain ---
    print("Reconstructing time-domain signals via superposition...")
    
    
    vg_time1 = reconstruct_time_domain(t, freqs, vg_list)[0, :] # Extract input at HV,1
    vg_time2 = reconstruct_time_domain(t, freqs, vg_list)[1, :] # Extract input at HV,2

    v_time_global = reconstruct_time_domain(t, freqs, v_freq_res)
    i_time_global = reconstruct_time_domain(t, freqs, i_freq_res)
    
    # --- 5f. Plotting ---
    print("Plotting results...")
    fig, (ax1, ax2, ax3,ax4,ax5,ax6,ax7) = plt.subplots(7, 1, figsize=(10, 10), sharex=True)
    
    # Plot 1: Input Wave
    ax1.plot(t, vg_time1, color='black', linewidth=2)
    ax1.plot(t, vg_time2, color='red', linewidth=2)
    ax1.set_title(f"Reconstructed Input $v_G(t)$ at HV,1 (Peak={A}V, tr={tr}s)")
    ax1.set_ylabel("Voltage (V)")
    ax1.grid(True)
    
    # Plot 2: Output Voltage (Arbitrarily picking Node index 0 of v vector)
    
    ax2.plot(t, v_time_global[17, :]  , color='blue', linewidth=2)
    ax2.plot(t, v_time_global[18, :]  , color='red', linewidth=2)
    # ax2.plot(t, v_time_global[2, :]  , color='green', linewidth=2)
    ax2.set_title(f"Global Voltage Response $v(t)$ at Vs 1,2,3")
    ax2.set_ylabel("Voltage (V)")
    ax2.grid(True)
    
    node_v = 0
    ax3.plot(t, v_time_global[node_v, :] - v_time_global[node_v+1, :], color='blue', linewidth=2)
    ax3.set_title(f"Global Voltage Response $v(t)$ at Vp1-Vp2")
    ax3.set_ylabel("Voltage (V)")
    ax3.grid(True)

    node_v = 17
    ax4.plot(t, v_time_global[node_v, :] - v_time_global[node_v+1, :], color='blue', linewidth=2)
    ax4.set_title(f"Global Voltage Response $v(t)$ at Vs1-Vs2 ")
    ax4.set_ylabel("Voltage (V)")
    ax4.grid(True)
    
    
    # Plot 3: Output Current (Arbitrarily picking Node index 0 of i vector)

    node_i = 18
    ax5.plot(t, i_time_global[node_i, :], color='red', linewidth=2)
    ax5.set_title(f"Global Current Response $i(t)$ Is1")
    ax5.set_xlabel("Time (s)")
    ax5.set_ylabel("Current (A)")
    ax5.grid(True)

    
    

    plt.tight_layout()
    plt.show()

    print("--- Pipeline Complete ---")
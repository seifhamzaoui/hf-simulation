"""
Tkinter UI front-end for the HF_model.py ladder-network transformer
simulation. Replaces hand-editing HF_model.py for every experiment with:

  - file pickers for the C/L/R matrices (cap_data.mat/induc.mat/DCR.mat
    or any alternative, e.g. *_ac_corrected.mat) and for SWF.py's
    R_ratio/L_ratio AC/DC scaling matrices,
  - a "Time Waveforms" tab: build the trapezoidal excitation (amplitude,
    period, rise time, harmonic count), pick any number of turn-to-turn
    voltage pairs (by PRIMARY/SECONDARY turn number, read from config.py's
    N1/N2 -- not raw array indices) and any number of branch currents to
    plot in the time domain,
  - a "Flat Spectrum" tab: sweep a flat (constant-amplitude) harmonic
    spectrum and plot impedance, transformation ratio, and a selected
    turn pair's amplification factor vs. frequency.

Reuses SWF.py's own functions (create_K/create_H/create_CK/create_CH,
solve_system_spectrum, get_fourier_components, get_flat_spectrum,
reconstruct_time_domain, and the turn_*_index/get_turn_voltage* helpers)
rather than re-deriving any of the circuit math here.
"""

import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
from scipy.io import loadmat

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

import config
import SWF as swf

MATRIX_DIR = "./ngsolve matrices"
DEFAULT_C_PATH = os.path.join(MATRIX_DIR, "cap_data.mat")
DEFAULT_L_PATH = os.path.join(MATRIX_DIR, "induc.mat")
DEFAULT_R_PATH = os.path.join(MATRIX_DIR, "DCR.mat")
DEFAULT_R_RATIO_PATH = os.path.join(MATRIX_DIR, "R_ratio_ngsolve.mat")
DEFAULT_L_RATIO_PATH = os.path.join(MATRIX_DIR, "L_ratio_ngsolve.mat")


def load_clr(c_path, l_path, r_path):
    """Mirrors HF_model.py's own C/L/R extraction exactly (same 1e-12
    F/pF and 1e-9*1.5 H/nH-with-correction-factor scaling it already
    uses) -- just parameterized on file path. These files only ever
    carry ONE 'f_...' entry (a single DC/base matrix, not a sweep,
    unlike R_ratio/L_ratio below)."""
    capa = loadmat(c_path)
    induc = loadmat(l_path)
    resis = loadmat(r_path)
    C = L = R = None
    for key in capa.keys():
        if key.startswith("f_"):
            C = capa[key] * 1e-12
    for key in induc.keys():
        if key.startswith("f_"):
            L = induc[key] * 1e-9 * 1.5
    for key in resis.keys():
        if key.startswith("f_"):
            R = resis[key]
    if C is None or L is None or R is None:
        raise ValueError("One of the C/L/R files has no 'f_...' matrix entry.")
    return C, L, R


def turn_labels(N1, N2):
    return [f"P{k}" for k in range(1, N1 + 1)] + [f"S{k}" for k in range(1, N2 + 1)]


def parse_turn_label(label):
    winding = "primary" if label[0] == "P" else "secondary"
    return winding, int(label[1:])


class HFModelFrame(ttk.Frame):
    """Everything HF_model.py's UI needs, as an embeddable ttk.Frame --
    use HFModelUI below to run it standalone, or instantiate this
    directly and .pack()/.grid() it as one screen of a larger app (see
    simulation_ui.py)."""

    def __init__(self, master=None):
        super().__init__(master)

        self.N1 = config.primary_turns_per_layer * config.primary_n_layers
        self.N2 = config.secondary_turns_per_layer * config.secondary_n_layers
        self.turn_choices = turn_labels(self.N1, self.N2)
        first_secondary = self.turn_choices[self.N1] if self.N2 >= 1 else self.turn_choices[0]
        second_secondary = self.turn_choices[self.N1 + 1] if self.N2 >= 2 else first_secondary

        self.voltage_pairs = []  # list of (labelA, labelB)

        self._build_file_selectors()
        self._build_notebook(first_secondary, second_secondary)

    # ------------------------------------------------------------------
    # Matrix file selectors
    # ------------------------------------------------------------------
    def _build_file_selectors(self):
        frame = ttk.LabelFrame(self, text=f"Matrix files  (N1={self.N1} primary turns, N2={self.N2} secondary turns)")
        frame.pack(fill="x", padx=8, pady=6)

        self.c_path = tk.StringVar(value=DEFAULT_C_PATH)
        self.l_path = tk.StringVar(value=DEFAULT_L_PATH)
        self.r_path = tk.StringVar(value=DEFAULT_R_PATH)
        self.r_ratio_path = tk.StringVar(value=DEFAULT_R_RATIO_PATH)
        self.l_ratio_path = tk.StringVar(value=DEFAULT_L_RATIO_PATH)

        self._file_row(frame, 0, "C matrix (capacitance)", self.c_path)
        self._file_row(frame, 1, "L matrix (inductance)", self.l_path)
        self._file_row(frame, 2, "R matrix (resistance)", self.r_path)
        self._file_row(frame, 3, "R_ratio (SWF AC/DC)", self.r_ratio_path)
        self._file_row(frame, 4, "L_ratio (SWF AC/DC)", self.l_ratio_path)

    def _file_row(self, parent, row, label, var):
        ttk.Label(parent, text=label, width=22).grid(row=row, column=0, sticky="w", padx=4, pady=2)
        ttk.Entry(parent, textvariable=var, width=65).grid(row=row, column=1, sticky="we", padx=4)
        ttk.Button(parent, text="Browse...", command=lambda: self._browse(var)).grid(row=row, column=2, padx=4)
        parent.columnconfigure(1, weight=1)

    def _browse(self, var):
        path = filedialog.askopenfilename(initialdir=MATRIX_DIR, filetypes=[("MAT files", "*.mat"), ("All files", "*.*")])
        if path:
            var.set(path)

    # ------------------------------------------------------------------
    # Notebook
    # ------------------------------------------------------------------
    def _build_notebook(self, first_secondary, second_secondary):
        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True, padx=8, pady=6)

        self.time_tab = ttk.Frame(nb)
        self.flat_tab = ttk.Frame(nb)
        nb.add(self.time_tab, text="Time Waveforms")
        nb.add(self.flat_tab, text="Flat Spectrum")

        self._build_time_tab()
        self._build_flat_tab(first_secondary, second_secondary)

    def _param_row(self, parent, row, label, var, width=12):
        ttk.Label(parent, text=label, width=18).grid(row=row, column=0, sticky="w", padx=4, pady=2)
        ttk.Entry(parent, textvariable=var, width=width).grid(row=row, column=1, padx=4, pady=2)

    def _common_matrices(self):
        C, L, R = load_clr(self.c_path.get(), self.l_path.get(), self.r_path.get())
        swf.load_ratio_matrices(self.r_ratio_path.get(), self.l_ratio_path.get())
        N1, N2 = self.N1, self.N2
        K = swf.create_K(N1, N2)
        H = swf.create_H(N1, N2)
        C_K = swf.create_CK(C, N1, N2)
        C_H = swf.create_CH(C, N1, N2)
        return C, L, R, K, H, C_K, C_H

    @staticmethod
    def _parse_excitation(text):
        parts = [p.strip() for p in text.split(",")]
        if len(parts) != 4:
            raise ValueError("Excitation must be exactly 4 comma-separated values: HV1,HV2,LV1,LV2")
        return [float(p) for p in parts]

    def _make_scrollable_controls(self, parent, width=340):
        """Wraps a controls column in a vertically scrollable Canvas so
        every widget -- including the Run & Plot button at the bottom --
        stays reachable even when the window is shorter than the
        controls' natural height (e.g. when this frame is embedded in
        simulation_ui.py's master app, which leaves much less vertical
        room than running HF_model_ui.py standalone)."""
        container = ttk.Frame(parent)
        container.pack(side="left", fill="y", padx=6, pady=6)
        canvas = tk.Canvas(container, highlightthickness=0, width=width)
        vscroll = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        inner = ttk.Frame(canvas)
        inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=vscroll.set)
        canvas.pack(side="left", fill="both", expand=True)
        vscroll.pack(side="left", fill="y")
        canvas.bind("<MouseWheel>", lambda e: canvas.yview_scroll(int(-e.delta / 120), "units"))
        return inner

    # ------------------------------------------------------------------
    # Time Waveforms tab
    # ------------------------------------------------------------------
    def _build_time_tab(self):
        controls = self._make_scrollable_controls(self.time_tab)

        wf = ttk.LabelFrame(controls, text="Trapezoidal excitation waveform")
        wf.pack(fill="x", pady=4)
        self.A_var = tk.DoubleVar(value=1000.0)
        self.T_var = tk.DoubleVar(value=1e-4)
        self.tr_var = tk.DoubleVar(value=0.0)
        self.numharm_var = tk.IntVar(value=500)
        self.exc_var = tk.StringVar(value="0.869, 0, 2.2, 0")
        self._param_row(wf, 0, "Amplitude A (V)", self.A_var)
        self._param_row(wf, 1, "Period T (s)", self.T_var)
        self._param_row(wf, 2, "Rise time tr (s)", self.tr_var)
        self._param_row(wf, 3, "Num. harmonics", self.numharm_var)
        ttk.Label(wf, text="Excitation [HV1,HV2,LV1,LV2]").grid(row=4, column=0, sticky="w", padx=4, pady=2)
        ttk.Entry(wf, textvariable=self.exc_var, width=22).grid(row=4, column=1, padx=4, pady=2)

        vp = ttk.LabelFrame(controls, text="Turn-to-turn voltages (pick 2 turns per trace)")
        vp.pack(fill="x", pady=4)
        self.vpair_a = tk.StringVar(value=self.turn_choices[0])
        self.vpair_b = tk.StringVar(value=self.turn_choices[1] if len(self.turn_choices) > 1 else self.turn_choices[0])
        ttk.Combobox(vp, textvariable=self.vpair_a, values=self.turn_choices, width=8, state="readonly").grid(row=0, column=0, padx=4, pady=2)
        ttk.Label(vp, text="-").grid(row=0, column=1)
        ttk.Combobox(vp, textvariable=self.vpair_b, values=self.turn_choices, width=8, state="readonly").grid(row=0, column=2, padx=4, pady=2)
        ttk.Button(vp, text="Add pair", command=self._add_voltage_pair).grid(row=0, column=3, padx=6)
        self.vpair_list = tk.Listbox(vp, height=6, width=26)
        self.vpair_list.grid(row=1, column=0, columnspan=4, sticky="we", padx=4, pady=2)
        ttk.Button(vp, text="Remove selected", command=self._remove_voltage_pair).grid(row=2, column=0, columnspan=4, pady=2)

        cur = ttk.LabelFrame(controls, text="Branch currents (ctrl/shift-click multiple)")
        cur.pack(fill="both", pady=4, expand=True)
        self.current_list = tk.Listbox(cur, selectmode="extended", height=10, exportselection=False)
        for lab in self.turn_choices:
            self.current_list.insert("end", lab)
        self.current_list.pack(fill="both", expand=True, padx=4, pady=2)

        ttk.Button(controls, text="Run & Plot", command=self._run_time_domain).pack(fill="x", pady=8)

        plot_frame = ttk.Frame(self.time_tab)
        plot_frame.pack(side="right", fill="both", expand=True)
        self.time_fig = Figure(figsize=(8, 8))
        self.time_canvas = FigureCanvasTkAgg(self.time_fig, master=plot_frame)
        toolbar = NavigationToolbar2Tk(self.time_canvas, plot_frame)
        toolbar.update()
        self.time_canvas.get_tk_widget().pack(fill="both", expand=True)

    def _add_voltage_pair(self):
        a, b = self.vpair_a.get(), self.vpair_b.get()
        if a == b:
            messagebox.showwarning("Invalid pair", "Pick two different turns.")
            return
        self.voltage_pairs.append((a, b))
        self.vpair_list.insert("end", f"{a} - {b}")

    def _remove_voltage_pair(self):
        for i in reversed(self.vpair_list.curselection()):
            self.vpair_list.delete(i)
            del self.voltage_pairs[i]

    def _run_time_domain(self):
        try:
            C, L, R, K, H, C_K, C_H = self._common_matrices()
            N1, N2 = self.N1, self.N2

            A = self.A_var.get()
            T = self.T_var.get()
            tr = self.tr_var.get()
            num_harm = self.numharm_var.get()
            exc_nodes = self._parse_excitation(self.exc_var.get())

            t = np.linspace(0, T, 100000)
            _, freqs, amps = swf.get_fourier_components(t, A, T, tr, num_harm)
            vg_list = swf.construct_vg_vectors(amps, excitation_nodes=exc_nodes)
            v_freq_res, i_freq_res = swf.solve_system_spectrum(freqs, vg_list, C_K, K, C_H, H, R, L)

            vg_time = swf.reconstruct_time_domain(t, freqs, vg_list)
            v_time = swf.reconstruct_time_domain(t, freqs, v_freq_res)
            i_time = swf.reconstruct_time_domain(t, freqs, i_freq_res)

            selected_currents = list(self.current_list.curselection())
            n_plots = 1 + len(self.voltage_pairs) + len(selected_currents)

            self.time_fig.clear()
            axes = self.time_fig.subplots(n_plots, 1, sharex=True)
            if n_plots == 1:
                axes = [axes]

            axes[0].plot(t, vg_time[0, :], label="HV,1", color="black")
            axes[0].plot(t, vg_time[2, :], label="LV,1", color="red")
            axes[0].set_title(f"Excitation waveform (A={A:g}V, T={T:g}s, tr={tr:g}s)")
            axes[0].set_ylabel("V")
            axes[0].legend(loc="upper right", fontsize=8)
            axes[0].grid(True)

            ax_i = 1
            for a, b in self.voltage_pairs:
                wa, ta = parse_turn_label(a)
                wb, tb = parse_turn_label(b)
                va = swf.get_turn_voltage(v_time, wa, ta, N1, N2)
                vb = swf.get_turn_voltage(v_time, wb, tb, N1, N2)
                axes[ax_i].plot(t, va - vb, color="blue")
                axes[ax_i].set_title(f"V({a}) - V({b})")
                axes[ax_i].set_ylabel("V")
                axes[ax_i].grid(True)
                ax_i += 1

            for idx in selected_currents:
                label = self.turn_choices[idx]
                w, tn = parse_turn_label(label)
                cidx = swf.turn_current_index(w, tn, N1, N2)
                axes[ax_i].plot(t, i_time[cidx, :], color="darkgreen")
                axes[ax_i].set_title(f"Current I({label})")
                axes[ax_i].set_ylabel("A")
                axes[ax_i].grid(True)
                ax_i += 1

            axes[-1].set_xlabel("Time (s)")
            self.time_fig.tight_layout()
            self.time_canvas.draw()
        except Exception as exc:
            messagebox.showerror("Error", str(exc))

    # ------------------------------------------------------------------
    # Flat Spectrum tab
    # ------------------------------------------------------------------
    def _build_flat_tab(self, first_secondary, second_secondary):
        controls = self._make_scrollable_controls(self.flat_tab)

        p = ttk.LabelFrame(controls, text="Flat spectrum parameters")
        p.pack(fill="x", pady=4)
        self.flatA_var = tk.DoubleVar(value=1000.0)
        self.flatT_var = tk.DoubleVar(value=1e-4)
        self.flat_numharm_var = tk.IntVar(value=1000)
        self.flat_exc_var = tk.StringVar(value="0.85, 0, 2.2, 0")
        self._param_row(p, 0, "Amplitude A", self.flatA_var)
        self._param_row(p, 1, "Period T (s)", self.flatT_var)
        self._param_row(p, 2, "Num. harmonics", self.flat_numharm_var)
        ttk.Label(p, text="Excitation [HV1,HV2,LV1,LV2]").grid(row=3, column=0, sticky="w", padx=4, pady=2)
        ttk.Entry(p, textvariable=self.flat_exc_var, width=22).grid(row=3, column=1, padx=4, pady=2)

        metrics = ttk.LabelFrame(controls, text="Metrics to plot")
        metrics.pack(fill="x", pady=4)

        self.show_impedance = tk.BooleanVar(value=True)
        ttk.Checkbutton(metrics, text="Impedance (Vexc,LV1 / I(turn))", variable=self.show_impedance).grid(row=0, column=0, columnspan=3, sticky="w", pady=(4, 0))
        self.imp_turn = tk.StringVar(value=first_secondary)
        ttk.Label(metrics, text="turn:").grid(row=1, column=0, sticky="e")
        ttk.Combobox(metrics, textvariable=self.imp_turn, values=self.turn_choices, width=8, state="readonly").grid(row=1, column=1, sticky="w", pady=2)

        self.show_ratio = tk.BooleanVar(value=True)
        ttk.Checkbutton(metrics, text="Transformation ratio  I(A)/I(B)", variable=self.show_ratio).grid(row=2, column=0, columnspan=3, sticky="w", pady=(8, 0))
        self.ratio_a = tk.StringVar(value=self.turn_choices[0])
        self.ratio_b = tk.StringVar(value=first_secondary)
        ttk.Combobox(metrics, textvariable=self.ratio_a, values=self.turn_choices, width=8, state="readonly").grid(row=3, column=0, padx=2, pady=2)
        ttk.Label(metrics, text="/").grid(row=3, column=1)
        ttk.Combobox(metrics, textvariable=self.ratio_b, values=self.turn_choices, width=8, state="readonly").grid(row=3, column=2, padx=2, pady=2)

        self.show_amplification = tk.BooleanVar(value=True)
        ttk.Checkbutton(metrics, text="Amplification factor  (V(A)-V(B))/Vexc,LV1", variable=self.show_amplification).grid(row=4, column=0, columnspan=3, sticky="w", pady=(8, 0))
        self.amp_a = tk.StringVar(value=first_secondary)
        self.amp_b = tk.StringVar(value=second_secondary)
        ttk.Combobox(metrics, textvariable=self.amp_a, values=self.turn_choices, width=8, state="readonly").grid(row=5, column=0, padx=2, pady=2)
        ttk.Label(metrics, text="-").grid(row=5, column=1)
        ttk.Combobox(metrics, textvariable=self.amp_b, values=self.turn_choices, width=8, state="readonly").grid(row=5, column=2, padx=2, pady=2)

        ttk.Button(controls, text="Run & Plot", command=self._run_flat_spectrum).pack(fill="x", pady=8)

        plot_frame = ttk.Frame(self.flat_tab)
        plot_frame.pack(side="right", fill="both", expand=True)
        self.flat_fig = Figure(figsize=(8, 8))
        self.flat_canvas = FigureCanvasTkAgg(self.flat_fig, master=plot_frame)
        toolbar = NavigationToolbar2Tk(self.flat_canvas, plot_frame)
        toolbar.update()
        self.flat_canvas.get_tk_widget().pack(fill="both", expand=True)

    def _run_flat_spectrum(self):
        try:
            C, L, R, K, H, C_K, C_H = self._common_matrices()
            N1, N2 = self.N1, self.N2

            A = self.flatA_var.get()
            T = self.flatT_var.get()
            num_harm = self.flat_numharm_var.get()
            exc_nodes = self._parse_excitation(self.flat_exc_var.get())

            freqs, amps = swf.get_flat_spectrum(A, T, num_harm)
            vg_list = swf.construct_vg_vectors(amps, excitation_nodes=exc_nodes)
            v_freq_res, i_freq_res = swf.solve_system_spectrum(freqs, vg_list, C_K, K, C_H, H, R, L)

            vg_arr = np.array(vg_list)
            i_arr = np.array(i_freq_res)

            plots = []  # (title, numerator, denominator)

            if self.show_impedance.get():
                w, tn = parse_turn_label(self.imp_turn.get())
                cidx = swf.turn_current_index(w, tn, N1, N2)
                plots.append((f"Impedance seen at {self.imp_turn.get()}  (Vexc,LV1 / I)",
                               vg_arr[:, 2, 0], i_arr[:, cidx, 0]))

            if self.show_ratio.get():
                wa, ta = parse_turn_label(self.ratio_a.get())
                wb, tb = parse_turn_label(self.ratio_b.get())
                ia = swf.turn_current_index(wa, ta, N1, N2)
                ib = swf.turn_current_index(wb, tb, N1, N2)
                plots.append((f"Transformation ratio  I({self.ratio_a.get()}) / I({self.ratio_b.get()})",
                               i_arr[:, ia, 0], i_arr[:, ib, 0]))

            if self.show_amplification.get():
                wa, ta = parse_turn_label(self.amp_a.get())
                wb, tb = parse_turn_label(self.amp_b.get())
                va = swf.get_turn_voltage_freq(v_freq_res, wa, ta, N1, N2)
                vb = swf.get_turn_voltage_freq(v_freq_res, wb, tb, N1, N2)
                plots.append((f"Amplification factor  (V({self.amp_a.get()})-V({self.amp_b.get()})) / Vexc,LV1",
                               va - vb, vg_arr[:, 2, 0]))

            if not plots:
                messagebox.showinfo("Nothing to plot", "Check at least one metric.")
                return

            self.flat_fig.clear()
            axes = self.flat_fig.subplots(2 * len(plots), 1, sharex=True)
            if len(plots) == 1:
                axes = [axes[0], axes[1]]

            epsilon = 1e-15
            for k, (title, num, den) in enumerate(plots):
                den_safe = np.where(den == 0, epsilon, den)
                ratio = num / den_safe
                mag = np.abs(ratio)
                phase = np.angle(ratio, deg=True)

                ax_mag = axes[2 * k]
                ax_phase = axes[2 * k + 1]
                ax_mag.loglog(freqs, mag, color="blue", linewidth=1.5)
                ax_mag.set_title(title, fontsize=10)
                ax_mag.set_ylabel("Magnitude")
                ax_mag.grid(True, which="both", ls="--")
                ax_phase.semilogx(freqs, phase, color="red", linewidth=1.5)
                ax_phase.set_ylabel("Phase (deg)")
                ax_phase.grid(True, which="both", ls="--")

            axes[-1].set_xlabel("Frequency (Hz)")
            self.flat_fig.tight_layout()
            self.flat_canvas.draw()
        except Exception as exc:
            messagebox.showerror("Error", str(exc))


class HFModelUI(tk.Tk):
    """Standalone window wrapping HFModelFrame -- kept so `python
    HF_model_ui.py` still works on its own. simulation_ui.py's master app
    embeds HFModelFrame directly instead of using this."""

    def __init__(self):
        super().__init__()
        self.title("HF Transformer Model")
        self.geometry("1300x850")
        HFModelFrame(self).pack(fill="both", expand=True)


if __name__ == "__main__":
    HFModelUI().mainloop()

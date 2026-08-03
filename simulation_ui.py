"""
Master Tkinter application tying together every stage of this project's
workflow into one multi-screen UI, so none of it requires hand-editing a
script or retyping CLI prompts for every experiment:

  - Config Editor      : edit config.py's VALUES through a form (numbers,
                          booleans, strings) -- never its code. Each field
                          is written back into the exact character span
                          its literal occupies in the source, so every
                          comment and everything not being edited is
                          preserved byte-for-byte (see ConfigField/
                          parse_config_fields below).
  - Geometry Builder    : launch transformer_geometry.py / _rectangular.py
                          to (re)build the STEP/DXF geometry from
                          config.py's current parameters.
  - NGSolve Simulations : launch simulation_ngsolve.py's and
                          simulation_ngsolve_cuda.py's own CLI stages
                          (Capacitance/DC Resistance/Inductance, CPU or
                          GPU), the AC Litz frequency sweep, and the R/L
                          AC-DC ratio sweeps (CPU and GPU).
  - HF Model            : the existing HF_model_ui.HFModelFrame (time-
                          domain waveforms and flat-spectrum transfer
                          functions).

Every long-running / heavy operation (geometry meshing, GPU/CPU field
solves) is launched as a SEPARATE SUBPROCESS -- both because several of
those scripts open their own native GUI (Netgen's viewer, a separate
Tcl/Tk runtime) that would conflict with this app's own Tkinter mainloop
if imported in-process, and because a crash or out-of-memory condition in
one of those heavy solves should not be able to take this whole UI down
with it. All such launches share ONE ProcessRunnerPanel (a persistent
console pinned at the bottom of the window) so at most one heavy job runs
at a time.
"""

import ast
import importlib
import io
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import tkinter as tk
import tokenize
from tkinter import messagebox, ttk

import config
import HF_model_ui as hf_ui

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(PROJECT_DIR, "config.py")

CPU_STAGES = [
    ("1", "Capacitance (Electrostatics)"),
    ("2", "DC Resistance (DC Conduction)"),
    ("3", "Inductance (curl-curl field solve, closed rings)"),
    ("3t", "Inductance -- QUICK TEST (ringp1 self-inductance only)"),
]
GPU_STAGES = [
    ("1", "Capacitance (Electrostatics) -- GPU solve"),
    ("2", "DC Resistance (DC Conduction) -- GPU solve"),
    ("2p", "PEEC self-inductance (ringp1, free space/no core) -- GPU"),
    ("2c", "PEEC self-inductance WITH core (ringp1, BEM) -- GPU"),
    ("3", "Inductance (curl-curl field solve, closed rings) -- GPU solve"),
    ("3t", "Inductance -- QUICK TEST (ringp1 self-inductance only) -- GPU solve"),
]


# ======================================================================
# Shared process console
# ======================================================================
class ProcessRunnerPanel(ttk.LabelFrame):
    """Console for launching project scripts as subprocesses. Streams
    stdout+stderr live into a log, and lets you answer a script's own
    input() prompts (the "press Enter to continue" safety gates several
    simulation functions use before an expensive meshing/factorization
    step) via the Send Enter button -- a subprocess's stdin here is a
    pipe, not a real interactive terminal, so those prompts would
    otherwise block forever with no way to satisfy them."""

    def __init__(self, master):
        super().__init__(master, text="Process console")
        self.proc = None
        self.out_queue = queue.Queue()

        top = ttk.Frame(self)
        top.pack(fill="x", padx=4, pady=2)
        self.status_var = tk.StringVar(value="Idle")
        ttk.Label(top, textvariable=self.status_var).pack(side="left")
        ttk.Button(top, text="Clear log", command=self.clear_log).pack(side="right", padx=2)
        ttk.Button(top, text="Stop", command=self.stop).pack(side="right", padx=2)
        ttk.Button(top, text="Send Enter", command=self.send_enter).pack(side="right", padx=2)

        text_frame = ttk.Frame(self)
        text_frame.pack(fill="both", expand=True, padx=4, pady=4)
        self.text = tk.Text(text_frame, height=12, wrap="word", state="disabled",
                             font=("Consolas", 9), background="#111318", foreground="#dddddd",
                             insertbackground="#dddddd")
        scroll = ttk.Scrollbar(text_frame, orient="vertical", command=self.text.yview)
        self.text.configure(yscrollcommand=scroll.set)
        self.text.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")

        self.after(100, self._poll_queue)

    def is_running(self):
        return self.proc is not None and self.proc.poll() is None

    def run(self, cmd, cwd=None, initial_input=None, label=None):
        if self.is_running():
            messagebox.showwarning(
                "Busy", "A process is already running in the console below -- "
                        "stop it or wait for it to finish first.")
            return
        self._log(f"$ {' '.join(cmd)}\n")
        self.status_var.set(f"Running: {label or os.path.basename(cmd[-1])}")
        try:
            # Binary pipes (no text=True) -- see _read_output()'s docstring
            # for why: text-mode line iteration hides any prompt that
            # doesn't end in '\n', which is exactly what input()'s prompt
            # text always looks like, making a script correctly WAITING
            # at a prompt look identical to a hung/broken one.
            self.proc = subprocess.Popen(
                cmd, cwd=cwd or PROJECT_DIR,
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                bufsize=0,
            )
        except Exception as exc:
            self._log(f"[failed to start] {exc}\n")
            self.status_var.set("Idle")
            return

        if initial_input:
            try:
                self.proc.stdin.write(initial_input.encode("utf-8"))
                self.proc.stdin.flush()
            except Exception:
                pass

        threading.Thread(target=self._read_output, args=(self.proc,), daemon=True).start()

    def _read_output(self, proc):
        """Reads raw bytes as soon as they're available (os.read returns
        whatever the pipe currently has, it does NOT wait for a full
        buffer or a newline) instead of iterating proc.stdout line by
        line. Line iteration silently withholds any partial/unterminated
        line from the log until either more data or EOF arrives -- and
        every input() prompt (e.g. transformer_geometry.py's trailing
        "click to end", or the CPU stages' "press Enter to continue"
        gates) is written WITHOUT a trailing newline, so with line-based
        reading the log would show nothing at all while the process sat
        correctly waiting for Send Enter, making it look hung."""
        try:
            while True:
                chunk = os.read(proc.stdout.fileno(), 4096)
                if not chunk:
                    break
                self.out_queue.put(chunk.decode("utf-8", errors="replace"))
        except Exception as exc:
            self.out_queue.put(f"[reader error] {exc}\n")
        finally:
            proc.wait()
            self.out_queue.put(f"\n[process exited with code {proc.returncode}]\n")
            self.out_queue.put(None)  # sentinel -> back to Idle

    def _poll_queue(self):
        try:
            while True:
                item = self.out_queue.get_nowait()
                if item is None:
                    self.status_var.set("Idle")
                    continue
                self._log(item)
        except queue.Empty:
            pass
        self.after(100, self._poll_queue)

    def _log(self, text):
        self.text.configure(state="normal")
        self.text.insert("end", text)
        self.text.see("end")
        self.text.configure(state="disabled")

    def clear_log(self):
        self.text.configure(state="normal")
        self.text.delete("1.0", "end")
        self.text.configure(state="disabled")

    def run_own_console(self, cmd, cwd=None, label=None):
        """Launches cmd with its OWN new console window (Windows) instead
        of piping stdio into this panel's log. Some native-GUI scripts
        (Netgen's 3D viewer via `from netgen.gui import *` / Draw()) open
        a window whose background Tcl/Tk event-pump thread apparently
        needs a real attached console/TTY to stay responsive -- with
        piped stdio the surrounding input()/exit flow works fine (see
        _read_output's docstring for that separate, already-fixed bug),
        but the viewer window itself sat frozen/"Not Responding" the
        whole time it was open. Giving the child its own real console
        (exactly like launching it by hand from a terminal) fixes that.
        Trade-off: no live log capture and no Send Enter here -- the
        script's own console window shows its output, and you answer its
        "press Enter" prompts by clicking into THAT window directly."""
        if self.is_running():
            messagebox.showwarning(
                "Busy", "A process is already running -- "
                        "stop it or wait for it to finish first.")
            return
        self._log(f"$ {' '.join(cmd)}  (opened in its own console window)\n")
        self.status_var.set(f"Running: {label or os.path.basename(cmd[-1])}")
        try:
            self.proc = subprocess.Popen(
                cmd, cwd=cwd or PROJECT_DIR,
                creationflags=subprocess.CREATE_NEW_CONSOLE,
            )
        except Exception as exc:
            self._log(f"[failed to start] {exc}\n")
            self.status_var.set("Idle")
            return
        threading.Thread(target=self._wait_only, args=(self.proc,), daemon=True).start()

    def _wait_only(self, proc):
        proc.wait()
        self.out_queue.put(f"\n[process exited with code {proc.returncode}]\n")
        self.out_queue.put(None)

    def send_enter(self):
        if self.is_running() and self.proc.stdin is None:
            self._log("[this process has its own console window -- click into it and press Enter there]\n")
            return
        if self.is_running():
            try:
                self.proc.stdin.write(b"\n")
                self.proc.stdin.flush()
            except Exception as exc:
                self._log(f"[send enter failed] {exc}\n")
        else:
            self._log("[no process running]\n")

    def stop(self):
        if self.is_running():
            self._log("[stopping process...]\n")
            try:
                self.proc.terminate()
            except Exception as exc:
                self._log(f"[stop failed] {exc}\n")
        else:
            self._log("[no process running]\n")


# ======================================================================
# Config Editor screen -- values only, never the surrounding code
# ======================================================================
# config.py's MATERIALS dict and nearly every numeric parameter carry
# extensive inline comments recording *why* each value is set the way it
# is (validation notes, placeholders-to-edit warnings, physical
# justifications). The editor below never touches that: it parses
# config.py's AST to find each editable literal's exact character span in
# the source, renders one widget per literal, and on save replaces ONLY
# those spans -- byte-for-byte identical everywhere else, comments
# included, no matter what gets edited.

_SECTION_HEADER_RE = re.compile(r'^#\s*-{3,}.*-{3,}\s*$')


def _build_line_offsets(source):
    """offsets[i] = absolute character offset of the start of line i+1
    (ast reports 1-indexed line numbers, 0-indexed columns)."""
    offsets = [0]
    for line in source.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(line))
    return offsets


def _section_headers_by_line(source):
    """Maps each line number to the nearest preceding '# --- WORD ---'
    header comment's WORD, so General fields can be grouped the same way
    config.py already visually groups itself -- uses tokenize (not the
    AST, which discards comments) to find them."""
    mapping = {}
    current = "General"
    try:
        for tok in tokenize.generate_tokens(io.StringIO(source).readline):
            if tok.type == tokenize.COMMENT and _SECTION_HEADER_RE.match(tok.string.strip()):
                current = tok.string.strip().lstrip("#").strip().strip("-").strip() or current
            mapping[tok.start[0]] = current
    except (tokenize.TokenizeError, IndentationError, SyntaxError):
        pass
    return mapping


class ConfigField:
    """One editable literal in config.py: a label to show, a Tk variable
    to edit it with, and the exact (start,end) absolute character span
    its value occupies in the source -- everything needed to both render
    a widget and write the edit back precisely."""

    def __init__(self, label, kind, start, end, value):
        self.label = label
        self.kind = kind  # "bool" | "number" | "string"
        self.start = start
        self.end = end
        self.var = tk.BooleanVar(value=value) if kind == "bool" else tk.StringVar(value=value)

    def validate(self):
        if self.kind == "number":
            text = self.var.get().strip()
            try:
                value = ast.literal_eval(text)
            except Exception:
                return f"{self.label}: {text!r} is not a valid number"
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                return f"{self.label}: {text!r} is not a number"
        return None

    def replacement_text(self):
        if self.kind == "bool":
            return "True" if self.var.get() else "False"
        if self.kind == "string":
            return repr(self.var.get())
        return self.var.get().strip()  # number: written back verbatim, already validated


def _field_from_constant(label, value_node, source, offsets):
    if not isinstance(value_node, ast.Constant):
        return None  # not a plain literal (e.g. a list comprehension) -- not editable here
    start = offsets[value_node.lineno - 1] + value_node.col_offset
    end = offsets[value_node.end_lineno - 1] + value_node.end_col_offset
    raw = value_node.value
    if isinstance(raw, bool):
        return ConfigField(label, "bool", start, end, raw)
    if isinstance(raw, (int, float)):
        return ConfigField(label, "number", start, end, source[start:end])  # keep original formatting (e.g. "5.8e7")
    if isinstance(raw, str):
        return ConfigField(label, "string", start, end, raw)
    return None  # None/bytes/etc. -- not a value this form edits


def _dict_value(dict_node, key):
    for k_node, v_node in zip(dict_node.keys, dict_node.values):
        if isinstance(k_node, ast.Constant) and k_node.value == key:
            return v_node
    return None


def _parse_materials(dict_node, source, offsets):
    """MATERIALS = {"ringp": {"ngsolve": {...}, "litz": {...}}, "Core":
    {"ngsolve": {...}, "ac": {...}}, ...} -- only the numeric/bool/string
    leaves inside each material's "ngsolve"/"litz"/"ac" sub-dicts are
    exposed (pattern/aedt are code-level identifiers matched elsewhere in
    the codebase, not physical values to tune here)."""
    materials = {}
    for key_node, val_node in zip(dict_node.keys, dict_node.values):
        if not isinstance(key_node, ast.Constant) or not isinstance(val_node, ast.Dict):
            continue
        fields = []
        for sub_key in ("ngsolve", "litz", "ac"):
            sub_dict = _dict_value(val_node, sub_key)
            if not isinstance(sub_dict, ast.Dict):
                continue
            for k_node, v_node in zip(sub_dict.keys, sub_dict.values):
                if isinstance(k_node, ast.Constant):
                    field = _field_from_constant(f"{sub_key}.{k_node.value}", v_node, source, offsets)
                    if field:
                        fields.append(field)
        if fields:
            materials[key_node.value] = fields
    return materials


def parse_config_fields(source):
    """Returns (general_fields, material_fields):
      general_fields  : list of (section_title, ConfigField) for every
                         top-level `name = <literal>` assignment.
      material_fields : {material_name: [ConfigField, ...]} from MATERIALS.
    Anything that isn't a plain literal (MATERIALS' own dict structure,
    sim_frequencies' list comprehension, etc.) is either handled
    specially (MATERIALS) or simply not editable through this form."""
    tree = ast.parse(source, CONFIG_PATH)
    offsets = _build_line_offsets(source)
    section_for_line = _section_headers_by_line(source)

    general_fields = []
    material_fields = {}
    current_section = "General"
    for node in tree.body:
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)):
            continue
        name = node.targets[0].id
        current_section = section_for_line.get(node.lineno, current_section)
        if name == "MATERIALS" and isinstance(node.value, ast.Dict):
            material_fields = _parse_materials(node.value, source, offsets)
            continue
        field = _field_from_constant(name, node.value, source, offsets)
        if field:
            general_fields.append((current_section, field))
    return general_fields, material_fields


def apply_fields(source, fields):
    """Applies every field's replacement_text() to its own span, working
    from the END of the file backwards so earlier spans' offsets stay
    valid as later edits shift the text around them."""
    buf = source
    for field in sorted(fields, key=lambda f: f.start, reverse=True):
        buf = buf[:field.start] + field.replacement_text() + buf[field.end:]
    return buf


class ConfigEditorFrame(ttk.Frame):
    """Value-only form editor for config.py -- see the module comment
    above this class for how it avoids touching anything but the literal
    values themselves. Save validates every edited value (numbers must
    still parse as numbers) and re-compiles the resulting source before
    writing, and keeps a config.py.bak backup of whatever was on disk."""

    def __init__(self, master, on_config_reloaded=None):
        super().__init__(master)
        self.on_config_reloaded = on_config_reloaded
        self.source = ""
        self.general_fields = []
        self.material_fields = {}

        toolbar = ttk.Frame(self)
        toolbar.pack(fill="x", padx=6, pady=4)
        ttk.Button(toolbar, text="Reload from disk", command=self.reload_from_disk).pack(side="left", padx=2)
        ttk.Button(toolbar, text="Save", command=self.save).pack(side="left", padx=2)
        ttk.Button(toolbar, text="Save and reload into app", command=self.save_and_reload).pack(side="left", padx=2)
        self.status_var = tk.StringVar(value="")
        ttk.Label(toolbar, textvariable=self.status_var).pack(side="left", padx=10)

        outer = ttk.Frame(self)
        outer.pack(fill="both", expand=True, padx=6, pady=4)
        self._canvas = tk.Canvas(outer, highlightthickness=0)
        vscroll = ttk.Scrollbar(outer, orient="vertical", command=self._canvas.yview)
        self.form = ttk.Frame(self._canvas)
        self.form.bind("<Configure>", lambda e: self._canvas.configure(scrollregion=self._canvas.bbox("all")))
        self._canvas.create_window((0, 0), window=self.form, anchor="nw")
        self._canvas.configure(yscrollcommand=vscroll.set)
        self._canvas.pack(side="left", fill="both", expand=True)
        vscroll.pack(side="right", fill="y")
        self._canvas.bind("<MouseWheel>", lambda e: self._canvas.yview_scroll(int(-e.delta / 120), "units"))

        self.reload_from_disk()

    def reload_from_disk(self):
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            self.source = f.read()
        self.general_fields, self.material_fields = parse_config_fields(self.source)
        self._build_form()
        self.status_var.set(f"Loaded {os.path.basename(CONFIG_PATH)}")

    def _build_form(self):
        for child in self.form.winfo_children():
            child.destroy()

        sections, order = {}, []
        for section, field in self.general_fields:
            if section not in sections:
                sections[section] = []
                order.append(section)
            sections[section].append(field)

        for section in order:
            box = ttk.LabelFrame(self.form, text=section)
            box.pack(fill="x", padx=4, pady=4, anchor="w")
            self._render_fields(box, sections[section])

        if self.material_fields:
            mat_box = ttk.LabelFrame(self.form, text="MATERIALS")
            mat_box.pack(fill="x", padx=4, pady=4, anchor="w")
            for mat_name, fields in self.material_fields.items():
                sub = ttk.LabelFrame(mat_box, text=mat_name)
                sub.pack(fill="x", padx=4, pady=4, anchor="w")
                self._render_fields(sub, fields)

        ttk.Label(
            self.form, foreground="#666", wraplength=700, justify="left",
            text="Note: sim_frequencies (a computed list) and any other non-literal "
                 "expression aren't shown here -- this form only edits plain values.",
        ).pack(anchor="w", padx=8, pady=(4, 10))

    def _render_fields(self, parent, fields):
        for i, field in enumerate(fields):
            ttk.Label(parent, text=field.label, width=28).grid(row=i, column=0, sticky="w", padx=4, pady=2)
            if field.kind == "bool":
                ttk.Checkbutton(parent, variable=field.var).grid(row=i, column=1, sticky="w", padx=4, pady=2)
            else:
                ttk.Entry(parent, textvariable=field.var, width=24).grid(row=i, column=1, sticky="w", padx=4, pady=2)

    def _all_fields(self):
        fields = [f for _, f in self.general_fields]
        for mat_fields in self.material_fields.values():
            fields.extend(mat_fields)
        return fields

    def save(self):
        fields = self._all_fields()
        errors = [msg for msg in (f.validate() for f in fields) if msg]
        if errors:
            messagebox.showerror("Invalid value(s)", "\n".join(errors))
            return False

        new_source = apply_fields(self.source, fields)
        try:
            compile(new_source, CONFIG_PATH, "exec")
        except SyntaxError as exc:
            messagebox.showerror("Syntax error", f"Resulting config.py would not be valid, NOT saved:\n{exc}")
            return False

        backup_path = CONFIG_PATH + ".bak"
        try:
            shutil.copy2(CONFIG_PATH, backup_path)
        except Exception:
            pass
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            f.write(new_source)
        self.source = new_source
        self.status_var.set(f"Saved (backup: {os.path.basename(backup_path)})")
        return True

    def save_and_reload(self):
        if not self.save():
            return
        importlib.reload(config)
        if self.on_config_reloaded:
            self.on_config_reloaded()
        self.status_var.set("Saved and reloaded into running app")


# ======================================================================
# Geometry Builder screen
# ======================================================================
class GeometryBuilderFrame(ttk.Frame):
    def __init__(self, master, runner):
        super().__init__(master)
        self.runner = runner

        ttk.Label(self, justify="left", wraplength=700, text=(
            "Builds the transformer geometry from config.py's CURRENT parameters and "
            "writes transformer_model.step / transformer_model_closed.step (plus a 2D "
            "DXF export). Opens Netgen's own 3D viewer in its OWN console window (not "
            "piped into the process console below) -- Netgen's viewer needs a real "
            "attached console to stay responsive, otherwise it shows as frozen/'Not "
            "Responding'. When you're done looking at it, click into that console "
            "window and press Enter there to let the script exit (or use Stop below "
            "to force it closed)."
        )).pack(anchor="w", padx=10, pady=10)

        row = ttk.Frame(self)
        row.pack(fill="x", padx=10, pady=6)

        round_box = ttk.LabelFrame(row, text="Round conductors")
        round_box.pack(side="left", fill="both", expand=True, padx=(0, 6))
        ttk.Label(round_box, text="transformer_geometry.py", foreground="#555").pack(anchor="w", padx=6, pady=(4, 0))
        ttk.Button(round_box, text="Build geometry",
                   command=lambda: self._build("transformer_geometry.py")).pack(anchor="w", padx=6, pady=6)

        rect_box = ttk.LabelFrame(row, text="Rectangular conductors")
        rect_box.pack(side="left", fill="both", expand=True, padx=(6, 0))
        ttk.Label(rect_box, text="transformer_geometry_rectangular.py", foreground="#555").pack(anchor="w", padx=6, pady=(4, 0))
        ttk.Button(rect_box, text="Build geometry",
                   command=lambda: self._build("transformer_geometry_rectangular.py")).pack(anchor="w", padx=6, pady=6)

    def _build(self, script_name):
        self.runner.run_own_console([sys.executable, "-u", script_name], cwd=PROJECT_DIR, label=script_name)


# ======================================================================
# NGSolve Simulations screen
# ======================================================================
class StagesTab(ttk.Frame):
    """Mirrors a script's own STAGES dict (simulation_ngsolve.py /
    simulation_ngsolve_cuda.py) as Checkbuttons. Run launches that
    script unmodified as a subprocess and answers its "which stage(s)?"
    input() prompt with the selected keys -- exactly what typing them at
    a real terminal would do. Any FURTHER input() prompts inside the
    stage functions themselves (the CPU stages' own "press Enter to
    continue" gates before an expensive step -- the GPU stages have
    none) need Send Enter in the process console to advance."""

    def __init__(self, master, runner, script_name, stages):
        super().__init__(master)
        self.runner = runner
        self.script_name = script_name
        self.vars = {}
        for key, label in stages:
            var = tk.BooleanVar(value=False)
            ttk.Checkbutton(self, text=f"{key}) {label}", variable=var).pack(anchor="w", padx=10, pady=2)
            self.vars[key] = var
        ttk.Button(self, text=f"Run selected on {script_name}", command=self._run).pack(anchor="w", padx=10, pady=10)

    def _run(self):
        selected = [k for k, v in self.vars.items() if v.get()]
        if not selected:
            messagebox.showinfo("Nothing selected", "Check at least one stage first.")
            return
        self.runner.run([sys.executable, "-u", self.script_name], cwd=PROJECT_DIR,
                         initial_input=",".join(selected) + "\n", label=self.script_name)


class LitzSweepTab(ttk.Frame):
    """simulation_ngsolve_litz.py's full AC frequency sweep (CPU only --
    see that file's docstring for why there's no GPU equivalent here)."""

    def __init__(self, master, runner):
        super().__init__(master)
        self.runner = runner
        ttk.Label(self, justify="left", wraplength=560, text=(
            "Full N-ring x frequency-sweep AC inductance/resistance "
            "(simulation_ngsolve_litz.py). A full sweep is FAR more expensive than the "
            "quick single-ring smoke test -- see that file's own docstring/memory "
            "warnings. Uncheck 'quick test' only once you actually mean to run the "
            "full sweep."
        )).pack(anchor="w", padx=10, pady=10)
        self.quick_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(self, text="Quick test only (ringp1)", variable=self.quick_var).pack(anchor="w", padx=10)
        ttk.Button(self, text="Run AC sweep", command=self._run).pack(anchor="w", padx=10, pady=10)

    def _run(self):
        if self.quick_var.get():
            code = "import simulation_ngsolve_litz as lz; lz.run_litz_sweep(test_rings=['ringp1'])"
        else:
            if not messagebox.askyesno(
                "Full sweep",
                "This solves every ring at every swept frequency -- can take a very "
                "long time and a lot of RAM. Continue?",
            ):
                return
            code = "import simulation_ngsolve_litz as lz; lz.run_litz_sweep()"
        self.runner.run([sys.executable, "-u", "-c", code], cwd=PROJECT_DIR, label="simulation_ngsolve_litz.py")


class RatioSweepTab(ttk.Frame):
    """simulation_ngsolve_litz_ratio(.py|_cuda.py)'s run_ratio_sweep* --
    the small-representative-sample AC/DC ratio sweep used to scale the
    full DCR.mat/induc.mat matrices, far cheaper than LitzSweepTab's full
    sweep. primary_count/secondary_count default to config.py's own
    LITZ_RATIO_SAMPLE_COUNT_PRIMARY/_SECONDARY but are overridable here
    per run without editing config.py."""

    def __init__(self, master, runner, module_name, function_name, warning=None):
        super().__init__(master)
        self.runner = runner
        self.module_name = module_name
        self.function_name = function_name

        if warning:
            ttk.Label(self, text=warning, justify="left", wraplength=560,
                      foreground="#b30000").pack(anchor="w", padx=10, pady=(10, 4))

        form = ttk.Frame(self)
        form.pack(anchor="w", padx=10, pady=6)
        self.primary_count = tk.IntVar(value=getattr(config, "LITZ_RATIO_SAMPLE_COUNT_PRIMARY", 3))
        self.secondary_count = tk.IntVar(value=getattr(config, "LITZ_RATIO_SAMPLE_COUNT_SECONDARY", 3))
        ttk.Label(form, text="primary_count").grid(row=0, column=0, sticky="w", padx=4, pady=2)
        ttk.Entry(form, textvariable=self.primary_count, width=8).grid(row=0, column=1, padx=4, pady=2)
        ttk.Label(form, text="secondary_count").grid(row=1, column=0, sticky="w", padx=4, pady=2)
        ttk.Entry(form, textvariable=self.secondary_count, width=8).grid(row=1, column=1, padx=4, pady=2)

        self.quick_freq_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            self, text="Quick test: only 2 frequencies (1kHz, 1MHz) instead of the full config.sim_frequencies sweep",
            variable=self.quick_freq_var,
        ).pack(anchor="w", padx=10, pady=4)

        ttk.Button(self, text="Run ratio sweep", command=self._run).pack(anchor="w", padx=10, pady=10)

    def _run(self):
        freq_arg = ", frequencies_hz=[1e3, 1e6]" if self.quick_freq_var.get() else ""
        code = (f"import {self.module_name} as m; "
                f"m.{self.function_name}(primary_count={self.primary_count.get()}, "
                f"secondary_count={self.secondary_count.get()}{freq_arg})")
        self.runner.run([sys.executable, "-u", "-c", code], cwd=PROJECT_DIR, label=self.module_name)


class NGSolveSimulationsFrame(ttk.Frame):
    def __init__(self, master, runner):
        super().__init__(master)
        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True, padx=6, pady=6)

        nb.add(StagesTab(nb, runner, "simulation_ngsolve.py", CPU_STAGES), text="CPU Stages")
        nb.add(StagesTab(nb, runner, "simulation_ngsolve_cuda.py", GPU_STAGES), text="GPU Stages")
        nb.add(LitzSweepTab(nb, runner), text="AC Sweep (Litz, CPU)")
        nb.add(RatioSweepTab(nb, runner, "simulation_ngsolve_litz_ratio", "run_ratio_sweep"),
               text="R/L Ratio (CPU)")
        nb.add(RatioSweepTab(
            nb, runner, "simulation_ngsolve_litz_ratio_cuda", "run_ratio_sweep_gpu",
            warning=("Known issue (see this project's own dev history): the AC (f>0) GPU "
                     "solve currently fails to converge/factor (ILU+GMRES never got a "
                     "working preconditioner). Only the DC (f=0) baseline is confirmed "
                     "working -- expect a real sweep to error out partway through."),
        ), text="R/L Ratio (GPU)")


# ======================================================================
# Master app
# ======================================================================
SCREENS = ["Config Editor", "Geometry Builder", "NGSolve Simulations", "HF Model"]


class MasterApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Transformer HF/TMF Simulation Suite")
        self.geometry("1400x950")

        paned = ttk.PanedWindow(self, orient="vertical")
        paned.pack(fill="both", expand=True)

        top = ttk.Frame(paned)
        paned.add(top, weight=4)

        self.runner = ProcessRunnerPanel(paned)
        paned.add(self.runner, weight=1)

        sidebar = ttk.Frame(top, width=190)
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)

        self.content = ttk.Frame(top)
        self.content.pack(side="left", fill="both", expand=True)
        self.content.rowconfigure(0, weight=1)
        self.content.columnconfigure(0, weight=1)

        self.screens = {}
        self.hf_frame = None
        self._build_screens()

        ttk.Label(sidebar, text="Screens", font=("Segoe UI", 10, "bold")).pack(fill="x", padx=8, pady=(10, 4))
        for name in SCREENS:
            ttk.Button(sidebar, text=name, command=lambda n=name: self._show(n)).pack(fill="x", padx=8, pady=4)

        self._show(SCREENS[0])

    def _build_screens(self):
        self.screens["Config Editor"] = ConfigEditorFrame(self.content, on_config_reloaded=self._on_config_reloaded)
        self.screens["Geometry Builder"] = GeometryBuilderFrame(self.content, self.runner)
        self.screens["NGSolve Simulations"] = NGSolveSimulationsFrame(self.content, self.runner)
        self.hf_frame = hf_ui.HFModelFrame(self.content)
        self.screens["HF Model"] = self.hf_frame
        for frame in self.screens.values():
            frame.grid(row=0, column=0, sticky="nsew")

    def _show(self, name):
        self.screens[name].tkraise()

    def _on_config_reloaded(self):
        """config.py itself was already reloaded in-place by
        ConfigEditorFrame (importlib.reload mutates the same module
        object every importer shares) -- but the HF Model screen copied
        N1/N2 and built its turn-choice widgets at CONSTRUCTION time, so
        those need re-deriving from the fresh values by rebuilding the
        screen, not just reloading the module."""
        old = self.hf_frame
        new_frame = hf_ui.HFModelFrame(self.content)
        new_frame.grid(row=0, column=0, sticky="nsew")
        self.screens["HF Model"] = new_frame
        self.hf_frame = new_frame
        new_frame.tkraise()
        old.destroy()


if __name__ == "__main__":
    MasterApp().mainloop()

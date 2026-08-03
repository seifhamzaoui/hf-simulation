"""
Tiny launcher stub, frozen via PyInstaller into the distributable's
SimulationSuite.exe. Deliberately has almost no dependencies of its own
(just stdlib) so PyInstaller has nothing complicated to analyze -- all
the real work (tkinter, matplotlib, ngsolve/netgen, numpy/scipy, cupy)
runs as a normal, UNFROZEN Python program via the bundled portable
interpreter in runtime\\python.exe, which is a full, ordinary CPython
install copied wholesale rather than something PyInstaller had to freeze
or understand. This sidesteps needing PyInstaller to trace NGSolve's
native GUI/Tcl-Tk/plugin-loading internals at all -- see build_portable.ps1's
own header comment for why that would otherwise be a serious risk.
"""

import os
import subprocess
import sys


def _base_dir():
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


def main():
    base = _base_dir()
    python_exe = os.path.join(base, "runtime", "python.exe")
    app_script = os.path.join(base, "app", "simulation_ui.py")
    app_dir = os.path.join(base, "app")

    if not os.path.isfile(python_exe) or not os.path.isfile(app_script):
        _show_error(
            f"Missing bundled files.\n\nExpected:\n  {python_exe}\n  {app_script}\n\n"
            "This launcher must stay next to its 'runtime' and 'app' folders."
        )
        return 1

    result = subprocess.run([python_exe, app_script], cwd=app_dir)
    return result.returncode


def _show_error(message):
    try:
        import tkinter
        from tkinter import messagebox
        root = tkinter.Tk()
        root.withdraw()
        messagebox.showerror("Simulation Suite", message)
        root.destroy()
    except Exception:
        pass


if __name__ == "__main__":
    sys.exit(main())

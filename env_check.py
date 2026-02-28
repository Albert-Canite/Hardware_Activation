import os
import platform
import sys


def ensure_runtime_compatibility() -> None:
    """Fail-fast checks to avoid common Windows 0xC0000005 crashes.

    Most frequent issue in this project setup is PyTorch 2.2.x + NumPy 2.x ABI mismatch.
    """
    # Keep thread usage conservative on Windows to reduce native runtime instability.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    try:
        import numpy as np
    except Exception as exc:
        print(f"[ENV CHECK] NumPy import failed: {exc}")
        print("Please reinstall NumPy in the active environment.")
        raise SystemExit(1)

    np_ver = tuple(int(p) for p in np.__version__.split("+")[0].split(".")[:2])
    if np_ver >= (2, 0):
        print("[ENV CHECK] Detected NumPy >= 2.0 in current environment.")
        print("[ENV CHECK] With PyTorch 2.2.x on Windows, this can trigger 0xC0000005 (access violation).")
        print("[ENV CHECK] Recommended fix in your conda env:")
        print("  pip uninstall -y numpy")
        print('  pip install "numpy<2"')
        print("Then rerun the script.")
        raise SystemExit(1)

    if platform.system().lower() == "windows":
        # Extra hint for common runtime mismatch scenarios.
        print(f"[ENV CHECK] Windows detected | Python: {sys.version.split()[0]} | NumPy: {np.__version__}")

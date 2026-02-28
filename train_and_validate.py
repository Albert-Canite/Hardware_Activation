import os
import subprocess
import sys


def run_preflight() -> int:
    print("[BOOT] Starting preflight import probe...")
    sys.stdout.flush()
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import torch, numpy; from openpyxl import load_workbook; "
                "print('IMPORT_PROBE_OK'); "
                "print('torch_version=', torch.__version__); "
                "print('torch_cuda_build=', torch.version.cuda); "
                "print('cuda_available=', torch.cuda.is_available())"
            ),
        ],
        capture_output=True,
        text=True,
        timeout=40,
    )

    if probe.returncode in (3221225477, -1073741819):
        print("[FATAL] Preflight import probe crashed with 0xC0000005.")
        print("[FATAL] This is a native runtime crash before training starts.")
        print("[FATAL] Common causes: NVIDIA driver/runtime mismatch, broken CUDA DLL path, OpenMP DLL conflicts.")
        return 1

    if probe.returncode != 0:
        print("[FATAL] Preflight import probe failed:")
        print(probe.stdout)
        print(probe.stderr)
        return probe.returncode

    print(probe.stdout.strip())
    print("[BOOT] Preflight import probe passed.")
    return 0


def launch_worker() -> int:
    worker_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_worker.py")
    cmd = [sys.executable, worker_script] + sys.argv[1:]
    print(f"[BOOT] Launching worker: {cmd[0]} {os.path.basename(worker_script)} ...")
    sys.stdout.flush()

    proc = subprocess.Popen(cmd)
    proc.wait()
    code = proc.returncode

    if code in (3221225477, -1073741819):
        print("\n[FATAL] Worker crashed with 0xC0000005 (native access violation).")
        print("[FATAL] Your Python script reached torch runtime and crashed in native layer.")
        print("[FATAL] Check these first on your machine:")
        print("  1) torch build CUDA version (from preflight torch_cuda_build) must align with installed NVIDIA driver.")
        print("  2) Avoid mixed CUDA DLLs in PATH (conda cudatoolkit + system CUDA conflict).")
        print("  3) Update GPU driver to a version supporting CUDA 12.1 runtime.")
        print("  4) If using IDE in OneDrive path, move project to a local non-synced folder and retry.")
        return 1

    return code


def main() -> None:
    code = run_preflight()
    if code != 0:
        raise SystemExit(code)

    code = launch_worker()
    raise SystemExit(code)


if __name__ == "__main__":
    main()

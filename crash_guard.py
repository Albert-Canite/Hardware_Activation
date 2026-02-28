import subprocess
import sys
from pathlib import Path
from typing import Iterable

WINDOWS_ACCESS_VIOLATION_CODES = {-1073741819, 3221225477, -1}


def _probe_import(cmd: list[str]) -> tuple[int, str]:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    out = (p.stdout or "").strip()
    return p.returncode, out


def print_runtime_probe() -> None:
    print("[CRASH GUARD] Running runtime probe...")
    probes = [
        [sys.executable, "-c", "import torch; print('torch', torch.__version__)"],
        [sys.executable, "-c", "import torchvision; print('torchvision', torchvision.__version__)"],
        [sys.executable, "-c", "from torchvision import models; m=models.vgg11(weights=None); print('vgg11 ok')"],
    ]
    for cmd in probes:
        code, out = _probe_import(cmd)
        print(f"[CRASH GUARD] probe rc={code} | cmd={' '.join(cmd[1:3])} ...")
        if out:
            print(out.splitlines()[-1])


def run_worker_with_guard(script_path: Path, cli_args: Iterable[str], hint_cuda: bool = True) -> int:
    cmd = [sys.executable, str(script_path.resolve()), *cli_args, "--_worker"]
    ret = subprocess.run(cmd).returncode

    if ret in WINDOWS_ACCESS_VIOLATION_CODES:
        print("\n[CRASH GUARD] Worker exited with code indicating Windows access violation (0xC0000005).")
        print(f"[CRASH GUARD] Raw worker return code: {ret}")
        print("[CRASH GUARD] This usually means native runtime conflict (PyTorch/CUDA/MKL DLL), not Python logic.")
        if hint_cuda:
            print("[CRASH GUARD] Please try in your PyONN env:")
            print("  1) conda install -y cpuonly pytorch torchvision pytorch-cuda=none -c pytorch")
            print("  2) or reinstall a matching CUDA stack for torch 2.2.2 + cu121")
            print("  3) set IDE Run configuration env: CUDA_VISIBLE_DEVICES=-1")
        print_runtime_probe()
        return 1

    if ret != 0:
        print(f"[CRASH GUARD] Worker failed with non-zero exit code: {ret}")
        print("[CRASH GUARD] If this is on Windows IDE, check interpreter path and try running from terminal once.")
    return ret

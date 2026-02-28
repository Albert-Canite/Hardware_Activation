import os


def apply_torch_stability_mode(torch_module) -> None:
    """Apply conservative torch runtime settings for Windows CPU stability."""
    torch = torch_module

    # Threading: reduce OpenMP/MKL contention-related native crashes.
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except Exception:
        pass

    # Disable MKLDNN on CPU if present; some environments crash in native kernels.
    try:
        torch.backends.mkldnn.enabled = False
    except Exception:
        pass

    # Ensure CPU quant backend is deterministic in old envs.
    try:
        if hasattr(torch.backends, "quantized"):
            torch.backends.quantized.engine = "fbgemm"
    except Exception:
        pass

    # Keep env hints aligned with single-thread mode.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")


def tiny_torch_self_check(torch_module, device):
    """Run a minimal forward/backward to detect low-level kernel crashes early."""
    torch = torch_module
    x = torch.randn(2, 3, 32, 32, device=device)
    w = torch.randn(8, 3, 3, 3, device=device, requires_grad=True)
    y = torch.nn.functional.conv2d(x, w, padding=1)
    loss = y.mean()
    loss.backward()

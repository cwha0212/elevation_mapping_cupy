#
# Array backend selection: cupy on CUDA machines, numpy everywhere else.
#
# Everything that used to say `import cupy as cp` now says
# `from elevation_mapping_cupy.backend import xp as cp` and keeps working on
# either. The choice is made once, at import time:
#
#   ELEVATION_BACKEND=cupy   force cupy (ImportError if it is missing)
#   ELEVATION_BACKEND=numpy  force numpy even where cupy exists (A/B tests)
#   unset / auto             cupy when importable, numpy otherwise
#
# The numpy path exists for boards without a CUDA GPU (RK3588 and the like).
# Its kernels live in kernels/numpy_kernels.py and are plain numpy with an
# optional numba JIT for the two per-point ray walks.
#
import os

import numpy as np

_choice = os.environ.get("ELEVATION_BACKEND", "auto").strip().lower()

cp = None
if _choice in ("auto", "cupy"):
    try:
        import cupy as cp  # noqa: F401
        import cupyx  # noqa: F401
    except Exception:  # ImportError, or a CUDA driver that fails to initialise
        cp = None
        if _choice == "cupy":
            raise

USE_CUPY = cp is not None
BACKEND = "cupy" if USE_CUPY else "numpy"

if USE_CUPY:
    import cupy as xp
    import cupyx
    from cupyx.scipy import ndimage
else:
    import numpy as xp
    from scipy import ndimage

    cupyx = None


def asnumpy(a):
    """Host copy of a backend array (a no-op view for numpy)."""
    if USE_CUPY:
        return xp.asnumpy(a)
    return np.asarray(a)


def zeros_pinned(shape, dtype=np.float32):
    """Pinned host buffer on cupy, an ordinary array on numpy."""
    if USE_CUPY:
        return cupyx.zeros_pinned(shape, dtype=dtype)
    return np.zeros(shape, dtype=dtype)


def copy_to_host(host_out, device_view):
    """host_out[...] = device_view, as one contiguous transfer."""
    if USE_CUPY:
        tmp = xp.ascontiguousarray(device_view)
        tmp.get(out=host_out)
    else:
        np.copyto(host_out, device_view)


def to_backend(a, dtype=None):
    return xp.asarray(a, dtype=dtype) if dtype is not None else xp.asarray(a)


__all__ = [
    "xp",
    "ndimage",
    "USE_CUPY",
    "BACKEND",
    "asnumpy",
    "zeros_pinned",
    "copy_to_host",
    "to_backend",
]

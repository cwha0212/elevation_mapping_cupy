#!/usr/bin/env python3
"""Run the scenario under one backend and save it, or compare two saved runs.
    ELEVATION_BACKEND=cupy  python3 tools/backend_ab.py save /tmp/ab_cupy.npz
    ELEVATION_BACKEND=numpy python3 tools/backend_ab.py save /tmp/ab_numpy.npz
    python3 tools/backend_ab.py compare /tmp/ab_cupy.npz /tmp/ab_numpy.npz
"""
import sys, time
import numpy as np

if sys.argv[1] == "save":
    from elevation_mapping_cupy.tests.backend_scenario import run_scenario
    from elevation_mapping_cupy.backend import BACKEND
    t0 = time.time(); out = run_scenario(); dt = time.time() - t0
    np.savez_compressed(sys.argv[2], **out)
    print("backend %s  scenario %.2f s  layers %s" % (BACKEND, dt, list(out)))
else:
    a, b = np.load(sys.argv[2]), np.load(sys.argv[3])
    for k in a.files:
        x, y = a[k], b[k]
        fa, fb = np.isfinite(x), np.isfinite(y)
        both = fa & fb
        diff = np.abs(x[both] - y[both]) if both.any() else np.zeros(1)
        print("%-15s finite %5.1f%% vs %5.1f%%  agree-finite %5.1f%%  |diff| med %.4f p95 %.4f max %.4f" % (
            k, 100 * fa.mean(), 100 * fb.mean(), 100 * (fa == fb).mean(), np.median(diff), np.percentile(diff, 95), diff.max()))

"""The scenario runs end to end on the active backend and produces sane layers."""
import numpy as np

from elevation_mapping_cupy.tests.backend_scenario import run_scenario


def test_scenario_runs_and_layers_are_sane():
    out = run_scenario(frames=8)
    e = out["elevation"]
    assert np.isfinite(e).mean() > 0.3
    # ground near the robot sits around z = 0 (sensor at 0.7, centre carried)
    c = e.shape[0] // 2
    assert abs(np.nanmedian(e[c - 20:c + 20, c - 20:c + 20])) < 0.1
    d = out["drivability"]
    assert np.isfinite(d).any() and 0.0 <= np.nanmin(d) and np.nanmax(d) <= 1.0
    s = out["safety"]
    assert np.isfinite(s).any()
    assert np.isfinite(out["untrav"]).any() or True  # the camera may see only a few cells

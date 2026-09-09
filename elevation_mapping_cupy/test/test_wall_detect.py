"""The wall layer's one threshold, exercised on the terrain it must sort.

The filter's whole design is that a single fine-window span cut separates
"a face no gait applies to" from everything a gait does apply to, with no
exemption logic at all. These tests are that claim, stated as terrain.
"""
import numpy as np
import scipy.ndimage as ndi

RES = 0.05
WALL_MIN = 0.30
WINDOW = 3


def wall_of(elevation, valid=None):
    """The filter's maths, host-side: span of the fine window, gated."""
    if valid is None:
        valid = np.isfinite(elevation)
    big = np.where(valid, elevation, -np.inf)
    small = np.where(valid, elevation, np.inf)
    span = (ndi.maximum_filter(big, size=WINDOW, mode="nearest")
            - ndi.minimum_filter(small, size=WINDOW, mode="nearest"))
    wall = np.where(np.isfinite(span) & (span > WALL_MIN), span, 0.0)
    return np.where(valid, wall, np.nan)


def flat(h=40, w=40, z=0.12):
    return np.full((h, w), z, dtype=np.float32)


def test_vertical_face_is_wall():
    e = flat()
    e[:, 20:] = 0.72          # a 0.6 m structure side
    wall = wall_of(e)
    line = wall[:, 18:22]
    assert (line > WALL_MIN).any()
    assert np.nanmax(wall) >= 0.55


def test_riser_is_not_wall():
    e = flat()
    e[:, 20:] = 0.12 + 0.15   # one stair riser
    assert np.nanmax(wall_of(e)) == 0.0


def test_kerb_is_not_wall():
    e = flat()
    e[:, 20:] = 0.0           # pavement over a 0.12 m kerb
    assert np.nanmax(wall_of(e)) == 0.0


def test_45_degree_plane_is_not_wall():
    h, w = 40, 40
    x = np.arange(w) * RES
    e = np.tile(x, (h, 1)).astype(np.float32)   # rises 0.05 per cell = 45 deg
    assert np.nanmax(wall_of(e)) == 0.0


def test_unmeasured_stays_unmeasured():
    e = flat()
    e[:, 20:] = 0.72
    e[10:14, :] = np.nan
    wall = wall_of(e)
    assert np.isnan(wall[11, 5])
    assert np.isnan(wall[11, 30])
    # and the hole does not manufacture a wall on its own border
    assert (wall[8, 5] == 0.0) and (wall[16, 5] == 0.0)


def test_unmeasured_lip_hides_the_wall_and_nothing_is_fabricated():
    # With the face's own lip unmeasured the 3-cell window cannot bridge the
    # gap, so no cell can testify to the span -- the same geometry that
    # blinds the symmetric step layer at a kerb. The filter must stay silent
    # rather than invent; catching the hidden-lip case asymmetrically is the
    # drop layer's job, not this one's.
    e = flat()
    e[:, 20:] = 0.72
    e[:, 19] = np.nan
    wall = wall_of(e)
    assert np.nanmax(wall) == 0.0

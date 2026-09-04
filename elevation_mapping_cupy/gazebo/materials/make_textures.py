#!/usr/bin/env python3
"""Generate the road and sidewalk textures for the semantic demo world.

The Cityscapes model was trained on photographs of streets, and untextured
grey planes give it nothing to separate: it called both surfaces road with
0.6+ confidence and never scored sidewalk above 0.36, so the robot's own
footing came out as an obstacle.

These are crude on purpose -- the point is to supply the cues the network
actually keys on, not to look good. Asphalt: dark, noisy, with a dashed lane
marking down the middle. Sidewalk: pale, with a regular slab grid and darker
grout. Both tile many times within the image, since Gazebo maps a primitive's
UVs 0..1 across the whole face and a single motif stretched over 40 m would
read as a smear.

Committed alongside the PNGs so the textures can be regenerated rather than
merely inherited.
"""

import pathlib

import numpy as np
from PIL import Image

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "textures"
SIZE = 2048
rng = np.random.default_rng(7)


def _noise(scale: float, amount: float) -> np.ndarray:
    """Blocky value noise, upsampled. Cheap and adequate for surface grain."""
    n = max(2, int(SIZE / scale))
    small = rng.random((n, n))
    return np.array(
        Image.fromarray((small * 255).astype(np.uint8)).resize(
            (SIZE, SIZE), Image.BILINEAR
        ),
        dtype=np.float32,
    ) / 255.0 * amount


def asphalt() -> Image.Image:
    base = 0.18 + _noise(6, 0.05) + _noise(40, 0.04)
    img = np.repeat(base[:, :, None], 3, axis=2)
    # A scatter of paler aggregate, so the surface is not flat grey.
    grit = rng.random((SIZE, SIZE)) > 0.9985
    img[grit] += 0.35

    # Dashed centre line, and solid edge lines: the strongest single cue that
    # a surface is a carriageway rather than a path.
    lane = slice(SIZE // 2 - 10, SIZE // 2 + 10)
    for start in range(0, SIZE, 320):
        img[start:start + 190, lane] = 0.82
    for edge in (int(SIZE * 0.06), int(SIZE * 0.94)):
        img[:, edge - 7:edge + 7] = 0.72

    return Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))


def sidewalk() -> Image.Image:
    base = 0.62 + _noise(8, 0.06) + _noise(50, 0.05)
    img = np.repeat(base[:, :, None], 3, axis=2)
    img[:, :, 2] *= 0.96  # faintly warm, the way concrete paving reads

    # Slab grid. 16 across the tile, so a 3 m wide walk shows slabs about
    # 0.2 m across once the texture is mapped over it.
    step = SIZE // 16
    for i in range(0, SIZE, step):
        img[i:i + 6, :] = 0.42
        img[:, i:i + 6] = 0.42
    # Every other row offset by half a slab: a running bond, not a chequerboard.
    for row in range(step, SIZE, 2 * step):
        img[row:row + step, :] = np.roll(img[row:row + step, :], step // 2, axis=1)

    return Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))


def facade() -> Image.Image:
    """A plain building front. Cityscapes leans on scene layout, and a street
    with nothing standing beside it is not a street it has seen."""
    base = 0.55 + _noise(10, 0.05)
    img = np.repeat(base[:, :, None], 3, axis=2)
    img[:, :, 0] *= 1.04
    win_w, win_h = SIZE // 10, SIZE // 8
    for row in range(SIZE // 12, SIZE - win_h, win_h * 2):
        for col in range(SIZE // 12, SIZE - win_w, win_w * 2):
            img[row:row + win_h, col:col + win_w] = 0.16
            img[row:row + 8, col:col + win_w] = 0.30
    return Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, maker in (("asphalt", asphalt), ("sidewalk", sidewalk), ("facade", facade)):
        path = OUT / f"{name}.png"
        maker().save(path, optimize=True)
        print(f"{path.relative_to(HERE.parent.parent)}  {path.stat().st_size // 1024} KB")


if __name__ == "__main__":
    main()

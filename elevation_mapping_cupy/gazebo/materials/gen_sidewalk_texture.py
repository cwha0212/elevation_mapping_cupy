#!/usr/bin/env python3
"""Regenerate sidewalk.png at the scale the geometry actually stretches it to.

The sidewalk visual is a single 40 m x 3 m box face and SDF materials carry no
UV repeat, so the square texture is stretched once across the whole face. The
first texture drew ~14 brick columns, which the stretch turned into 2.8 m
bricks with 5-10 cm grout gashes every 2.5 m -- and SAM-TP, reasonably, read a
dark line that wide across a walkway as a gap (measured +4 logits on flat
cells, the phantom obstacle lines in /projected_map).

So the grid here is anisotropic on purpose: 200 columns x 15 rows on a square
image comes out as 0.2 m x 0.2 m pavers on the ground, with ~2 cm grout, and
the grout contrast is kept mild. Run from this directory:

    python3 gen_sidewalk_texture.py
"""
import numpy as np

SIZE = 2048
FACE_X, FACE_Y = 40.0, 5.0     # sidewalk box face, meters
BRICK = 0.2                     # paver size on the ground, meters
FIELD = np.array([168, 166, 158], np.float32)
GROUT = np.array([150, 148, 141], np.float32)

cols = int(round(FACE_X / BRICK))          # 200
rows = int(round(FACE_Y / BRICK))          # 15
gw_x = max(1, int(round(0.02 / (FACE_X / SIZE))))   # 2 cm grout in px, x
gw_y = max(1, int(round(0.02 / (FACE_Y / SIZE))))   # and y

rng = np.random.default_rng(7)
img = np.empty((SIZE, SIZE, 3), np.float32)
img[:] = FIELD

# per-brick brightness jitter, rows offset half a brick like running bond
cw, ch = SIZE / cols, SIZE / rows
yy, xx = np.mgrid[0:SIZE, 0:SIZE]
row_i = (yy / ch).astype(int)
col_f = xx / cw + (row_i % 2) * 0.5
col_i = col_f.astype(int)
jitter = rng.uniform(-7, 7, (rows + 1, cols + 2))
img += jitter[row_i, col_i][..., None]

# grout lines
gx = ((col_f % 1.0) * cw < gw_x)
gy = ((yy % ch) < gw_y)
img[gx | gy] = GROUT + rng.uniform(-3, 3)

# fine speckle so the surface is not synthetic-flat
img += rng.normal(0, 4, (SIZE, SIZE, 1)).astype(np.float32)
img = np.clip(img, 0, 255).astype(np.uint8)

try:
    import cv2
    cv2.imwrite("textures/sidewalk.png", img[..., ::-1])
except ImportError:
    from PIL import Image
    Image.fromarray(img).save("textures/sidewalk.png")
print("wrote textures/sidewalk.png (%d cols x %d rows, grout %dx%d px)"
      % (cols, rows, gw_x, gw_y))

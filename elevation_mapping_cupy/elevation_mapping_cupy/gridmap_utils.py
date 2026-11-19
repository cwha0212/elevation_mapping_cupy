from array import array as float_array

import numpy as np
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import MultiArrayLayout as MAL
from std_msgs.msg import MultiArrayDimension as MAD


def encode_rot180_as_gridmap_column(host_buf: np.ndarray) -> Float32MultiArray:
    """Wrap an already-rotated, C-contiguous float32 (rows, cols) buffer.

    Equivalent to encode_layer_to_multiarray(rot180(m).T, "gridmap_column")
    for a square layer, with one host copy (the frombytes) instead of three:
    the transpose the column layout implies is exactly undone by the
    transpose the map convention needs, so the bytes are rot180(m) in C order.
    """
    rows, cols = host_buf.shape
    msg = Float32MultiArray()
    msg.layout = MAL()
    msg.layout.dim.append(MAD(label="column_index", size=cols, stride=rows * cols))
    msg.layout.dim.append(MAD(label="row_index", size=rows, stride=rows))
    msg.data = float_array("f")
    msg.data.frombytes(memoryview(host_buf).cast("B"))
    return msg



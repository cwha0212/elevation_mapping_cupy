from elevation_mapping_cupy.backend import USE_CUPY

if USE_CUPY:
    from .custom_image_kernels import *  # noqa: F401,F403
    from .custom_kernels import *  # noqa: F401,F403
else:
    from .numpy_kernels import (  # noqa: F401
        add_points_kernel,
        error_counting_kernel,
        finalize_map_kernel,
        dilation_filter_kernel,
        normal_filter_kernel,
        image_to_map_correspondence_kernel,
    )

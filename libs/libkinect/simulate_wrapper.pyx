
# Importing the function declarations from the .pxd file and cimporting malloc and free
cimport simulate_pxd
from libc.stdlib cimport malloc, free
from libc.string cimport memcpy
import numpy as np
cimport numpy as np


def python_simulate(np.ndarray[float, ndim=2, mode="c"] vertices,
                    int num_verts,
                    np.ndarray[int, ndim=2, mode="c"] faces,
                    int num_faces,
                    int width,
                    int height,
                    float fx,
                    float fy,
                    float cx,
                    float cy,
                    float z_near,
                    float z_far,
                    float baseline,
                    int noise_type,
                    str dot_pattern_path,
                    bint debug):
    # Allocate memory for the output depth image
    cdef float * out_depth = <float *> malloc(width * height * sizeof(float))

    # Encode the string to bytes here if the C++ function expects a byte string
    dot_pattern_bytes = dot_pattern_path.encode('utf-8')  # Create a persistent bytes object
    cdef char * c_dot_pattern_path = dot_pattern_bytes

    # Call the C++ simulate function
    result = simulate_pxd.simulate(&vertices[0, 0], num_verts, &faces[0, 0], num_faces, out_depth, width, height,
                                   fx, fy, cx, cy, z_near, z_far, baseline, noise_type, c_dot_pattern_path, debug)

    # Create a NumPy array from the C array without copying data
    cdef np.ndarray depth_image = np.empty((height, width), dtype=np.float32)
    memcpy(depth_image.data, out_depth, width * height * sizeof(float))

    # Free the allocated memory
    free(out_depth)

    return result, depth_image

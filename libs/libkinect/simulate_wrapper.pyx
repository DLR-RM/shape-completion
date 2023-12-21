
# Importing the function declarations from the .pxd file and cimporting malloc and free
cimport simulate_pxd
from libc.stdlib cimport malloc, free
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
                    str dot_pattern_path, 
                    bint debug):
    # Allocate memory for the output depth image
    cdef float* out_depth = <float*> malloc(width * height * sizeof(float))
    
    # Call the C++ simulate function
    result = simulate_pxd.simulate(&vertices[0, 0], num_verts, &faces[0, 0], num_faces, out_depth, width, height, fx, fy, cx, cy, dot_pattern_path.encode(), debug)
    
    # Convert the output depth image to a Python list
    depth_image = [out_depth[i] for i in range(width * height)]
    
    # Free the allocated memory
    free(out_depth)
    
    return result, depth_image

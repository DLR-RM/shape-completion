
# distutils: language = c++
# distutils: sources = simulate.cpp

cdef extern from "simulate.cpp":
    void segfault_handler(int signal)
    int simulate(float *vertices,
                 int num_verts,
                 int *faces,
                 int num_faces,
                 float *out_depth,
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
                 const char *dot_pattern_path,
                 bint debug)

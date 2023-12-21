
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
                 const char *dot_pattern_path,
                 bint debug)

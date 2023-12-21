# Signed Distance Fields

This library implements computes signed distance fields (SDFs) for 3D triangle meshes using the Vega library.

## Installation

1. Download [Vega](https://viterbi-web.usc.edu/~jbarbic/vega/download.html) and extract it.
2. Download [Thread Building Blocks](https://github.com/oneapi-src/oneTBB/releases/tag/2019_U5) and extract it.
3. Fix TBB: In `tbb2018_20180618oss_lin/tbb2018_20180618oss/include/tbb/task.h` change line 225 to `tbb::task* next_offloaded;`
4. Fix Vega: In `VegaFEM-v4.0/libraries/objMesh/objMesh.h` add `#include <limits>`. In `VegaFEM-v4.0/libraries/mesh/edgeKey.h` add `#include <cstdint>`.
5.   Configure Vega: In `Makefile-headers/Makefile-header-linux` uncomment line 7 (`TBBFLAG=-DUSE_TBB`). Change line 28 to `TBB_INCLUDE=-I/path/to/tbb2018_20180618oss_lin/tbb2018_20180618oss/include
`. Change line 29 to `TBB_LIB=-ltbb -ltbb_preview -L/path/to/tbb2018_20180618oss_lin/tbb2018_20180618oss/lib/intel64/gcc4.7`
6. Build Vega:

    ```bash
    cd VegaFEM-v4.0
    ./build
    cd utilities/computeDistanceField
    make
    cd ../..
    cd utilities/isosurfaceMesher
    make
    ```

7. Copy the `computeDistanceField` and `isosurfaceMesher` executables to `/libs/vega`).
8. Copy `libtbb.so.2` and `libtbb_preview.so.2` from `tbb2018_20180618oss_lin/tbb2018_20180618oss/lib/intel64/gcc4.7` to `/libs/vega`.
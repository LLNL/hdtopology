# hdtopology
Topological data analysis library for NDDAV (n-dimensional data analysis and visualization) system (https://github.com/LLNL/NDDAV.git)

#### Requirements:
python3, swig3, numpy, cmake (3.12 or up), ANN (Approximate Nearest Neighbor, original repo: https://www.cs.umd.edu/~mount/ANN/, for compiling with CMake: https://github.com/dials/annlib)

The hdtopology library also incorporated the core functionality of the NGL library (developed by Carlos Correa) for empty region graph computation, the full NGL code base can be access at: http://www.ngraph.org/


#### Install
1. Install (e.g., ```apt-get install libann-dev``` on linux, or ```port install ann``` on mac) or compile ANN library from source.

2. Compile the hdtopology library and python wrapper:
```console
mkdir build
cd build
cmake .. -DENABLE_PYTHON=ON -DANN_INCLUDE_DIR=path/to/ann/include -DANN_LIBRARY=path/to/ann/libANN.a
make install
```

Released under LLNL-CODE-772013

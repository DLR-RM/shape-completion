# Intersect - Triangle Hash Computation

Intersect is a C++ and Cython extension library for efficient triangle hash computation. It provides a fast and
optimized method to handle triangle intersections and hashing.

## Installation

To install the library, simply run the following command in the directory containing the `setup.py` file:

```bash
pip install .
```

Ensure you have NumPy and Cython installed in your environment, as they are required for the compilation.

## Usage

You can import and use the `TriangleHash` class as follows:

```python
from intersect.triangle_hash import TriangleHash as _TriangleHash
triangle_hash = _TriangleHash()
```

## Features

- Efficient triangle hash computation
- C++ optimization for speed
- Easy integration with Python projects

## Requirements

- Python 3.6 or higher
- NumPy
- Cython

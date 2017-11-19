# BPR Optimized SLIM in C++
## Requirements
- CMake
- [Eigen](http://eigen.tuxfamily.org/)
- [Cython](http://cython.org/) (_optional_)

## Compile C++ sources
```sh
$ mkdir build && cd build
$ cmake ..
$ make -j4
```

## Compile Cython wrapper
```
$ cd cython
$ cythonize -a -i BPyRSlim.pyx
```

Then import it in your Python code
```py
from BPyRSlim import BPyRSlim
```
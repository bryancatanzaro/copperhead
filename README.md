Copperhead: Data Parallel Python
================================

Copperhead is a data parallel language embedded in Python, along with
a compiler which creates highly efficient parallel code.  Currently
our compiler and runtime targets CUDA-enabled GPUs, as well as
multi-core CPUs using either Threading Building Blocks (TBB) or
OpenMP.

Copperhead is hosted at
    http://github.com/copperhead

Dependencies
============
    
Necessary for all backends
--------------------------

  - [boost::python](http://boost.org).  Tested with version 1.48.
  **Important note**: it is imperative that your boost::python library
  was built with the same compiler as your Python interpreter. For
  Linux machines using package managers, this is usually the case if
  you install boost::python through your system package manager.
  
  - [Codepy](http://mathema.tician.de/software/codepy).  We require
    version 2012.1.2 or greater.  This will be installed automatically
    if you do not have it.

  - [Thrust](http://github.com/thrust).  We require version 1.6 or
  better. Thrust is a header only library, so it is easy to install.

Backend-specific
----------------

  1. [CUDA](http://nvidia.com/cuda).  If you would like to use the CUDA
    backend, you must have CUDA installed, version 4.1 or better.
  
  2. [OpenMP](http://openmp.org).  Support for OpenMP is built-in to
    g++, so no installation is necessary.
  
  3. [Threading Building Blocks
    (TBB)](http://threadingbuildingblocks.org).  We have tested with
    TBB version 4.0.

Copperhead runs on OS X and Linux.  It currently does not run on
Windows, although we plan on supporting Windows in the future.

Tools
-----

  - g++ version 4.5 or greater.  If you want to build support for the
CUDA backend, you must use a version of g++ supported by nvcc.  As of
version 4.1, nvcc does not support versions of g++ newer than 4.5.

  - Scons.

Installation
============

To install Copperhead, simply run the included `setup.py`:
```python setup.py install```

The setup script will examine your system and try to build support for
all possible backends. Depending on your configuration, you may need
to include some additional configuration information.  The setup
script will alert you if configuration does not succeed, and will
create a file named `siteconf.py` with comments explaining what
configuration information you can provide.

You can verify your installation by running tests:
```python setup.py test```

Contributors
============

The primary contributors to Copperhead are [Bryan
Catanzaro](http://catanzaro.name) and [Michael
Garland](http://mgarland.org).
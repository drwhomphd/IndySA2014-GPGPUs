IndySA2014-GPGPUs
=================

The source code and presentation for my IndySA talk on GPGPU programming.

## Requirements
 1. Cuda 5.5+
 2. CMake 2.8
 3. GCC or possibly Visual Studio (untested)

## Compilation
 1. mkdir build
 2. cd build
 3. cmake ..
 4. make
 5. Look in sub directories
 
Compilation tested on Red Hat Enterprise Linux. Should run on any modern OS with GCC or a compiler that supports OpenMP. This means OSX users will most likely need to configure the use of GCC over LLVM with CMake. While I love LLVM, Apple really should have waited for it to have feature parity with GCC.

## A few experiments you can do with the examples:

  * warpdivergence: The warp divergence examples is supposed to show the performance hit from putting if statements inside a kernel. Obviously the numbers show the (hypothetically) divergent kernel to be MUCH faster. Figure out what the turning point is that would cause divergent branches to slow down. 
  * optcpugpu: Determine what performance changes occur if we switch from integers to floats to doubles

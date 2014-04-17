IndySA2014-GPGPUs
=================

The source code and presentation for my IndySA talk on GPGPU programming

A few experiments you can do with the examples:

  * warpdivergence: The warp divergence examples is supposed to show the performance hit from putting if statements inside a kernel. Obviously the numbers show the (hypothetically) divergent kernel to be MUCH faster. Figure out what the turning point is that would cause divergent branches to slow down. 
  * optcpugpu: Determine what performance changes occur if we switch from integers to floats to doubles

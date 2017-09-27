# electron-optimization

This is the source code for calculating electron trajectories in biconic cusp magnetic fields.

The physics aspect of the code is implemented in C in the part1.cl file. Two separate integration methods are present - Euler's and RK4. Euler's runs faster but RK4 can be performed with fewer time steps. 

The OpeCL code supports nth data reporting, allowing millions of steps to be performed with very little memory overhead. 

The potential\_optimizer.py code has a "all" object which contains handles for the OpenCL kernel deployment and a few simple methods of running basic simulations. 

It is possible to just ./potential\_optimizer.py and get a graph of a single electron trajectory, but all the necessary packages must be installed.

numpy
scipy
seaborn
pyopencl
matplotlib

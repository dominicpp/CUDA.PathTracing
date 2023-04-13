# Path Tracing with CUDA

Master's Thesis Project - Investigation of different parallelization techniques on GPUs with NVIDIA's API CUDA using a Path Tracer.

The goal of this project was to find out which parallelization techniques have an impact of parallelization when rendering a Path Tracer using CUDA. For this purpose, a parallel Path Tracer with CUDA was implemented on a GPU and a sequential Path Tracer was implemented on a CPU.

The main branch contains the sequential C/C++ implementation and the cuda_impl branch contains the parallel CUDA implementation.

This Master's Thesis Project was developed from scratch within a few weeks. It began as a sequential Path Tracer using C/C++, which was later converted into a parallel Path Tracer using CUDA.

Numerous important aspects of computer graphics were implemented, such as a virtual camera system, ray generation and simulated light rays, super-sampling including stratified sampling, Monte Carlo integration for diffuse virtual objects, surface reflections with various materials, recursive algorithm for light propagation, and more.

<p align="center">
    <img src="./doc/final_path_tracing_image.png"  width="99%" height="99%">
</p>

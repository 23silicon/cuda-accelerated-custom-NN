# Custom Feedforward Neural Network
Build a deep feedforward neural network from scratch, acccelerated with CUDA
 - Built FNN from the ground up (no machine learning libraries), supports ReLU, tanh, sigmoid activations.
 - **Achieves 97.65% on MNIST and 87.86% on Fashion-MNIST**, built a custom optimizer based on RMS-prop.

# GPU acceleration portion description:
This project uses a matrix multiplication CUDA kernel I wrote to significantly accelerate both inference and backpropagation by parallelizing on the GPU.
Uses Pybind11 to call the kernel (written in C++) from my network in python.

**On CPU:                      Training completes in 40m 4.68s**

**On GPU (with matmul kernel): Training completes in 3m 49.59s**

**Average Speedup: 10.47x (best result: 10.57x)**

GPU: NVIDIA Tesla T4
Cuda matmul kernels launched in lines 103, 177, 179 in the neural network training cell.


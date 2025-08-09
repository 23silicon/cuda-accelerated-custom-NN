# deep-learning
Build a deep feedforward neural network from scratch, acccelerated with CUDA
 Built FNN from the ground up (no machine learning libraries); supports ReLU, tanh, sigmoid activations.
 **Achieves 97.65% on MNIST and 87.86% on Fashion-MNIST**; built a custom optimizer based on RMS-prop.

# GPU acceleration portion description:
This project uses a matrix multiplication CUDA kernel I wrote to significantly accelerate both inference and backpropagation by parallelizing on the GPU.
Uses Pybind11 to call the kernel (written in C++) from my network in python.

**On CPU:                      Training completes in 40m 4.68s
On GPU (with matmul kernel): Training completes in 16m 37.23s

Speedup: 2.411x**

GPU: NVIDIA Tesla T4
Cuda matmul kernels launched in lines 93, 169, 171 in the neural network training cell.
TO DO for much faster training: (Once openml finally works again)
  - Implement vector addition kernel for adding result of matmul to bias vectors.
  - Replace numpy with torch tensors so copy between device and host is not needed.



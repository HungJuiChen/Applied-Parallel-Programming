Here's a README file for your **Applied Parallel Programming** project based on your report:

---

# Applied Parallel Programming

This repository contains the implementation and optimization of convolution operations using CUDA in an **Applied Parallel Programming** project. The project focuses on optimizing parallel execution using various CUDA techniques to enhance the performance of convolutional neural networks (CNNs).

## 📌 Project Overview

This project is part of the **ECE 408 / CS 483 - Applied Parallel Programming** coursework at the **University of Illinois Urbana-Champaign (UIUC)**. The goal is to improve the efficiency of CNN convolution operations on GPUs by applying several CUDA optimization techniques.

### 🔹 Implemented Optimizations:
1. **CUDA Streams**: Overlapping computation and memory transfer to improve GPU utilization.
2. **Shared Memory and Register Tiling**: Reducing global memory access latency by leveraging shared memory and register tiling.
3. **Kernel Fusion**: Merging multiple kernels to reduce memory operations and kernel launch overhead.
4. **Constant Memory for Weights**: Storing frequently used convolution weights in constant memory for faster access.
5. **Restrict Qualifier (`__restrict__`)**: Optimizing memory access patterns by reducing pointer aliasing.

## 🚀 Performance Improvements

The optimizations led to significant speedups in convolution execution time:

| Batch Size | Baseline Time | Optimized Time | Speedup |
|------------|--------------|---------------|---------|
| 100        | 4.016s       | 1.338s        | ~3x     |
| 1,000      | 43.717s      | 9.271s        | ~4.7x   |
| 10,000     | 363.963s     | 89.895s       | ~4x     |

## 🛠️ Implementation Details

- **CUDA Streams:** Enabled concurrent execution of memory operations and computation.
- **Matrix Multiplication with Shared Memory:** Improved data locality and minimized memory bottlenecks.
- **Kernel Fusion:** Combined multiple operations into a single kernel to reduce global memory traffic.
- **Loop Unrolling and Memory Access Optimization:** Applied `__restrict__` and constant memory techniques to optimize memory throughput.

## 🔧 Setup and Compilation

### Prerequisites
- **NVIDIA GPU with CUDA support**
- **CUDA Toolkit** (Ensure `nvcc` is installed)
- **Make** (for compiling the project)

### Compilation
To compile the project, run:

```bash
make
```

To clean previous builds:

```bash
make clean
```

## 🏃 Running the Code

To execute the convolution forward pass:

```bash
./conv_forward
```

To run performance profiling:

```bash
nvprof ./conv_forward
```

## 📊 Profiling and Results

The optimizations were evaluated using CUDA profiling tools (`nvprof`, `nsight`). Results show reduced execution time due to efficient memory management and concurrency. See the **results/** directory for detailed profiling reports.

## 📖 References

- **Programming Massively Parallel Processors** – Kirk & Hwu
- **NVIDIA CUDA Programming Guide**
- **UIUC ECE 408 Course Material**

## 🏆 Acknowledgments

This project was developed as part of the **ECE 408: Applied Parallel Programming** coursework at UIUC. Special thanks to course instructors and TAs for their guidance.

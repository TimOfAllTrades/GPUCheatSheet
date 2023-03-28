# 01_CUDA_add

The purpose of this program is to do a basic matrix addition using the GPU.

To compile this program in the command prompt, type:\n
nvcc 01_CUDA_add.cu -o 01_CUDA_add.exe\n
You can replace the input and output filenames as you wish\n

Background

The GPU creates numerous threads that are characterized by its location within a block and a each block is characterized by its location within a grid.

Thread Index             | Thread number (threadIdx.x) | Block number (blockIdx.x)
------------            | -------------  | -------------
0 |0    |0
1 |1    |0
2 |2    |0
3 |3    |0
4 |0    |1
5 |1    |1
6 |2    |1
7 |3    |1
8 |0    |2
9 |1    |2
10|2    |2
11|3    |2

The above table shows the index corresponding to a grid consisting of 3 blocks and 4 threads per block for a total of 12 threads in the grid.  The thread index can be returned by using the equation:

thread number + (block number * Threads per block)

or in terms of the system variables

threadIdx.x + blockIdx.x * blockDim.x  

To maximize parallelization, each thread of the grid should be used to perform a calculation for each index of the array.

The first section defines the GPU function
```C
__device__ int AddOne(int A)
{
    A += 1; 
    return A;
}

__global__ void SetIndex(int *A)                                       
{
    int index = threadIdx.x + blockIdx.x*blockDim.x;
    A[index] += index;     
    A[index] = AddOne(A[index]); 
}

```
The function with the \_\_global\_\_ declaration specifier is the GPU kernel function that gets called in the main function.  Using the system variables, threadIdx and blockIdx, we can determine the corresponding index of the kernel grid and match it to the corresponding index of the array.

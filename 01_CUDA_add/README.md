# 01_CUDA_add

The purpose of this program is to do a basic matrix addition using the GPU.

To compile this program in the command prompt, type:
nvcc 01_CUDA_add.cu -o 01_CUDA_add.exe
You can replace the input and output filenames as you wish

Background:
The GPU creates numerous threads that are characterized by its location within a block and a block is characterized by its location within a grid.

Index             | Thread number | Block number
------------            | -------------  | -------------
1 |1     | 1
2 |2     | 1
3 |3     | 1
4 |4     | 1
5 |1     | 2
6 |2     | 2
7 |3     | 2
8 |4     | 2
9 |1     | 3
10 |2     | 3
11 |3     | 3
12 |4     | 3

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

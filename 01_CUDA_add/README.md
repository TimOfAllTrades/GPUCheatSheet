# 01_CUDA_add

The purpose of this program is to do a basic matrix addition using the GPU.

To compile this program in the command prompt, type:
nvcc 01_CUDA_add.cu -o 01_CUDA_add.exe
You can replace the input and output filenames as you wish

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
The function with the \_\_global\_\_ declaration specifier is the 

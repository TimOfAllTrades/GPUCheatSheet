# 01_CUDA_add

The purpose of this program is to do a basic matrix addition using the GPU.

To compile this program in the command prompt, type:
nvcc 01_CUDA_add.cu -o 01_CUDA_add.exe
You can replace the input and output filenames as you wish

__Background__

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

__The Code__

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
The function with the \_\_global\_\_ declaration specifier is the GPU kernel function that gets called in the main function.  The \_\_device\_\_ declaration specifier indicates that the function is used for a GPU kernel.  Using the system variables, threadIdx and blockIdx, we can determine the corresponding index of the kernel grid and match it to the corresponding index of the array.  The above code indicates that increment the matrix value by its current index.  Then we apply another function called AddOne to each array element.

```C
int main(void)
{
    
    int N = 20;
    
    int *dev_A, *A;

    A = (int*) malloc(N*sizeof(int));

    for (int n = 0; n < N ; n++)
    {
        A[n] = 1;
    }

```
The first section defines N and declares two pointers, one used for system memory and the other to be used for video memory.  The system array pointer A is allocated memory equal to N elements.  The values are then all initialized to 1.


```C
    cudaMalloc((void**) &dev_A, N*sizeof(int));

    cudaMemset(dev_A, 0, N*sizeof(int)); 

    cudaMemcpy(dev_A, A, N*sizeof(int), cudaMemcpyHostToDevice);

    SetIndex<<<N/5,5>>>(dev_A);  

    cudaDeviceSynchronize(); 

    cudaMemcpy(A, dev_A, N*sizeof(int), cudaMemcpyDeviceToHost);
    
    Verify(A,N); 

    cudaFree(dev_A); 
}
```

The next section allocates space on the video memory for the GPU array pointer and cudaMemset resets all values to 0.  CudaMemcpy will copy the contents of the system memory onto the video memory.  SetIndex is the GPU kernel that will perform the GPU functions.  In this case we are specifying a grid of N/5 or 4 blocks, and each block having 5 threads.

cudaDeviceSynchronize waits for the device to finish calculating all threads in the Grid before moving on.  cudaMemcpy then copies all the results back onto the system memory where it can be regularly operated on with non GPU code, such as displaying it in the consol via cout or printf().
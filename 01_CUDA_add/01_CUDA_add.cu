#include <iostream>

//Compile Command: nvcc [filename].cu -o [filename].exe

__global__ void SetIndex(int *A)
{
    int index = threadIdx.x + blockIdx.x*blockDim.x;
    A[index] = index;
}

int main(void)
{
    int N = 20;
    int *dev_A, *A;

    //Allocate Host memory size
    A = (int*) malloc(N*sizeof(int));

    //Allocate Device memory size
    cudaMalloc((void**) &dev_A, N*sizeof(int));

    //Set device memory to 0
    cudaMemset(dev_A, 0, N*sizeof(int));

    //Run Kernel, Using (N/5) blocks with 5 threads per block
    SetIndex<<<N/5,5>>>(dev_A);

    //Copy back results
    cudaMemcpy(A, dev_A, N*sizeof(int), cudaMemcpyDeviceToHost);

    //Verify
    for (int n = 0; n < N; n++)
    {
        std::cout<<A[n] << "\n";
    }

    //Erasing the CUDA array
    cudaFree(A)
}
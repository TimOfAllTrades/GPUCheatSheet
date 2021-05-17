#include <iostream>

//Compile Command: nvcc [filename].cu -o [filename].exe

__device__ int AddOne(int A)
{
    //Functions in CUDA or GPU code start with __device__
    A += 1; 
    return A;
}

__global__ void SetIndex(int *A)
{
    //GPU function that sets the array index value to index and then adds 1
    int index = threadIdx.x + blockIdx.x*blockDim.x;
    A[index] = index;
    A[index] = AddOne(A[index]);

}

int main(void)
{
    //Set array size to 20 
    int N = 20;

    //Declare host and device array pointer
    int *dev_A, *A;

    //Allocate Host memory size
    A = (int*) malloc(N*sizeof(int));

    //Allocate Device memory size
    cudaMalloc((void**) &dev_A, N*sizeof(int));

    //Set device memory to 0
    cudaMemset(dev_A, 0, N*sizeof(int));

    //Run Kernel, Using (N/5) blocks with 5 threads per block
    SetIndex<<<N/5,5>>>(dev_A);

    //Synchronizing is not necessary since cudaMemcpy step is next but it never hurts to do so
    cudaDeviceSynchronize();

    //Copy back results
    cudaMemcpy(A, dev_A, N*sizeof(int), cudaMemcpyDeviceToHost);

    //Verify
    for (int n = 0; n < N; n++)
    {
        std::cout<<A[n] << "\n";
    }

    //Erasing the CUDA array
    cudaFree(A);
}
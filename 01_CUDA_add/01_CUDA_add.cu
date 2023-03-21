#include <iostream>

//Compile Command: nvcc [filename].cu -o [filename].exe

__device__ int AddOne(int A)                                            //Functions in CUDA or GPU code start with __device__
{
    A += 1; 
    return A;
}

__global__ void SetIndex(int *A)                                        //GPU function that sets adds array index value to index and then adds 1
{
    int index = threadIdx.x + blockIdx.x*blockDim.x;                    //Calculating the index of thread in the grid
    A[index] += index;                                                  //Incrementing the current array value by the index value
    A[index] = AddOne(A[index]);                                        //Also adding an additional 1
}

void Verify(int *A,int N)                                               //A typical C function with no output
{
    for (int n=0;n < N; n++)
    {
        std::cout<< A[n] << "\n";                                       //Iterating over all indices and outputting the result.
    }
}

int main(void)
{
    
    int N = 20;                                                         //Set array size to 20
    
    int *dev_A, *A;                                                     //Declare host and device array pointer

    A = (int*) malloc(N*sizeof(int));                                   //Allocate Host memory size

    for (int n = 0; n < N ; n++)                                        //Set some initial values to host array
    {
        A[n] = 1;                                                       //Set all values to 1
    }

    
    cudaMalloc((void**) &dev_A, N*sizeof(int));                         //Allocate video memory for calculation

    cudaMemset(dev_A, 0, N*sizeof(int));                                //Set device memory to 0

    cudaMemcpy(dev_A, A, N*sizeof(int), cudaMemcpyHostToDevice);        //Copy System memory to device memory.

    SetIndex<<<N/5,5>>>(dev_A);                                         //Run Kernel, Using (N/5) blocks with 5 threads per block

    cudaDeviceSynchronize();                                            //Synchronizing is not necessary since cudaMemcpy step is next but it never hurts to do so

    cudaMemcpy(A, dev_A, N*sizeof(int), cudaMemcpyDeviceToHost);        //Copy back results

    
    Verify(A,N);                                                        //Once the results are copied back, we will output it to see if the results match

    cudaFree(dev_A);                                                    //Erasing the CUDA array.  dev_A must be reallocated if to be used again.
}
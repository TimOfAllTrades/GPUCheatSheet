#include <iostream>


    __global__ void AddStuff(int *dev_N_Array)
    {
        int index = threadIdx.x + blockIdx.x*blockDim.x;
        dev_N_Array[index] +=5;

    }
    
    
    
int main(void) 
{
    int N = 1024*1024;
    int *dev_N_Array;

    cudaMalloc((void**) &dev_N_Array, N*sizeof(int));
    cudaMemset(dev_N_Array,0,N*sizeof(int));

    float elapsedTime;    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start,0);

    AddStuff<<<1024*1024/32, 32>>>(dev_N_Array);

    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    
    std::cout << elapsedTime ;

}

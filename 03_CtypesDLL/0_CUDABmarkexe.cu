#include <iostream>

//Compile code:nvcc 0_CUDABmarkexe.cu -o cuda.exe

/*
This program runs the code as a standalone exe program.
The purpose is to demonstrate that running GPU code from a DLL
has no performance penalty compared to a standalone *.exe file.

This program will create a 1024 x 1024 array, add 5 to all the
elements and measure the time it took to do so.

*/

__global__ void AddStuff(int *dev_N_Array)
{
    int index = threadIdx.x + blockIdx.x*blockDim.x;
    dev_N_Array[index] +=5;

}

    
//The main function    
int main(void) 
{

    //Creating a 1024x1024 array.
    int N = 1024*1024;
    int *dev_N_Array;

    //Allocating the memory size
    cudaMalloc((void**) &dev_N_Array, N*sizeof(int));
    cudaMemset(dev_N_Array,0,N*sizeof(int));

    //Creating the timing events
    float elapsedTime;    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start,0);

    //Run the GPU kernal
    AddStuff<<<1024*1024/32, 32>>>(dev_N_Array);

    //Stopping the timing events
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    
    //Output the measurement time
    std::cout <<"Calculation time: " << elapsedTime ;

}

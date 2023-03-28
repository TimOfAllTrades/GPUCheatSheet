#include <iostream>
#include <stdio.h>

//Compile Command: nvcc -o GPUkernal.dll --shared GPUkernel.cu

/*
This program demonstrates how to import the a numpy array from python
and perform GPU functions on it.  Despite the GPU grid having multi-
dimension capability, memory arrays do not.  It is necessary to flatten
them in python prior to being imported into the program.

This program attempts to add the same index value in GPU code that was
previously assigned in python.  The correct result is when all elements
are added with their respective index values a second time.
*/

extern "C" {

    __global__ void AddIndex(int *array)
    {
        int x = threadIdx.x + blockIdx.x*blockDim.x;
        int y = threadIdx.y + blockIdx.y*blockDim.y;
        int z = threadIdx.z + blockIdx.z*blockDim.z;
        int index = x + y*blockDim.x*gridDim.x + z*(blockDim.x*gridDim.x)*(blockDim.y*gridDim.y);

        array[index] += index;
    }

    
    

    __declspec(dllexport) void GPUAddIntIndex(int *indata, int xdim, int ydim, int zdim)
    {
        //Defining N, the total number of elements
        int N = xdim * ydim * zdim;
        
        //CUDA 3D grid uses the dim3 data type
        dim3 grids(1,1,1);
        dim3 threads(xdim,ydim,zdim);

        //Declaring dev pointers
        int *dev_indata;

        //Allocating memory space
        cudaMalloc((void**) &dev_indata , N*sizeof(int));
        
        //Copying memory
        cudaMemcpy(dev_indata, indata, N*sizeof(int), cudaMemcpyHostToDevice);
        
        //Run GPU kernel
        AddIndex<<<grids,threads>>>(dev_indata);

        cudaDeviceSynchronize();

        //Copying memory
        cudaMemcpy(indata, dev_indata, N*sizeof(int), cudaMemcpyDeviceToHost);
    }

}
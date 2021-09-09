#include <iostream>
#include <stdio.h>

//Compile Command: nvcc [filename].cu -o [filename].exe

extern "C" {

    __global__ void Addall(int *indata, int *xadd, int *yadd, int *zadd)
    {
        int x = threadIdx.x + blockIdx.x*blockDim.x;
        int y = threadIdx.y + blockIdx.y*blockDim.y;
        int z = threadIdx.z + blockIdx.z*blockDim.z;
        int index = x + y*blockDim.x*gridDim.x + z*(blockDim.x*gridDim.x)*(blockDim.y*gridDim.y);
        int xwidth = blockDim.x*gridDim.x;
        int ywidth = blockDim.y*gridDim.y;
        int zwidth = blockDim.z*gridDim.z;

        indata[index] = xadd[index] + yadd[index] + zadd[index];
    }

    __declspec(dllexport) void Matadd(int *indata, int *xadd, int *yadd, int *zadd)
    {
        int N = 27; 

        //CUDA 3D grid uses the dim3 data type
        dim3 grids(1,1,1);
        dim3 threads(3,3,3);

        //Declaring dev pointers
        int *dev_indata, *dev_xadd, *dev_yadd, *dev_zadd;

        //Allocating memory space
        cudaMalloc((void**) &dev_indata , N*sizeof(int));
        cudaMalloc((void**) &dev_xadd , N*sizeof(int));
        cudaMalloc((void**) &dev_yadd , N*sizeof(int));
        cudaMalloc((void**) &dev_zadd , N*sizeof(int));

        //Copying memory
        cudaMemcpy( dev_indata, indata, N*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy( dev_xadd  , xadd  , N*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy( dev_yadd  , yadd  , N*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy( dev_zadd  , zadd  , N*sizeof(int), cudaMemcpyHostToDevice);

        //Run GPU kernel
        Addall<<<grids,threads>>>(dev_indata, dev_xadd, dev_yadd, dev_zadd);

        cudaDeviceSynchronize();

        //Copying memory
        cudaMemcpy( indata, dev_indata, N*sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy( xadd  , dev_xadd  , N*sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy( yadd  , dev_yadd  , N*sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy( zadd  , dev_zadd  , N*sizeof(int), cudaMemcpyDeviceToHost);

        cudaDeviceSynchronize();
    }

}
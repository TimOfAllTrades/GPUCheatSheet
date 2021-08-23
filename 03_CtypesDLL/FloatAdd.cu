#include <iostream>

//Compile code nvcc -o [kernalname].dll --shared [kernalcode].cu
extern "C" {
    __global__ void GPUfunction(float *A, float a)
    {
        int index = threadIdx.x + blockIdx.x*blockDim.x;
        A[index] += index*a;
    }



    __declspec(dllexport) int Fsum(float *infdata, float *outfdata, float a)
    {
        //The input array just so happens to be 20, but for dynamic cases, an input would be needed
        int N = 20;

        float *dev_infdata;

        cudaMalloc((void**) &dev_infdata, N*sizeof(float));

        cudaMemcpy(dev_infdata, infdata, N*sizeof(float), cudaMemcpyHostToDevice);

        GPUfunction<<<N/5,5>>>(dev_infdata, a);

        cudaDeviceSynchronize();

        cudaMemcpy(outfdata, dev_infdata, N*sizeof(float), cudaMemcpyDeviceToHost);

        return a;
    }
}
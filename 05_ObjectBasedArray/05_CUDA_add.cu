#include <iostream>
#include <fstream>

//Compile Command: nvcc [filename].cu -o [filename].exe


class ArrayInt 
{
    public:
    int N;
    int *Host;
    int *Device;
    int xwidth;
    int ywidth;
    int zwidth;
    const char *Description;

    //Constructor
    public:
    ArrayInt(int x, int y, int z, const char *name)
    {
        N = x*y*z;
        Host = (int*) malloc(N*sizeof(int));
        cudaMalloc((void**) &Device, N*sizeof(int));
        Description = name;
        xwidth = x;
        ywidth = y;
        zwidth = z;
    }

    //Functions
    void CopyD2H ()
    {
        cudaMemcpy(Host, Device, N*sizeof(int), cudaMemcpyDeviceToHost);
    }

    void DeviceClear()
    {
        cudaFree(Device);
    }

    void DeviceIni()
    {
        cudaMalloc((void**) &Device, N*sizeof(int));
    }
    
    void DeviceZero()
    {
        cudaMemset(Device, 0, N*sizeof(int));
    }

    void CopyH2D ()
    {
        cudaMemcpy( Device, Host, N*sizeof(int), cudaMemcpyHostToDevice);
    }

    void HostClear()
    {
        free(Host);
    }

    void HostIni()
    {
        Host = (int*) malloc(N*sizeof(int));
    }

    void HostZero()
    {
        memset(Host, 0, N*sizeof(int));
    }

    

    void DumpTxt()
    {
        int* temparray;
        temparray = (int*) malloc(xwidth*ywidth*zwidth*sizeof(int));
        cudaMemcpy(temparray, Device, xwidth*ywidth*zwidth*sizeof(int), cudaMemcpyDeviceToHost);
        
        std::ofstream parafile;
        parafile.open(Description) ;
        for (int z = 0; z < zwidth; z++)
        {
            parafile << z << "\n";
            for (int y = 0; y < ywidth; y++)
            {
                for (int x = 0; x < xwidth; x++)
                {
                    parafile << Host[x + y * xwidth + z *xwidth*ywidth] << ",";
                }
                parafile << "," <<  y << ",,";
                for (int x = 0; x < xwidth; x++)
                {
                    parafile << temparray[x + y * xwidth + z *xwidth*ywidth] << ",";
                }
                parafile << "," <<  y << ",,";
                for (int x = 0; x < xwidth; x++)
                {
                    parafile << Host[x + y * xwidth + z *xwidth*ywidth]-temparray[x + y * xwidth + z *xwidth*ywidth] << ",";
                }
                parafile << "," <<  y << ",,";
                
                for (int x = 0; x < xwidth; x++)
                {
                    parafile << x + y * xwidth + z *xwidth*ywidth  << ",";
                }
                parafile << "," <<  y << ",,";
                parafile << "\n";

            }
            parafile << z << "\n";
            for (int x = 0; x < xwidth; x++)
                {
                    parafile << x << ",";
                }
            parafile << ","  << ",,";
            for (int x = 0; x < xwidth; x++)
                {
                    parafile << x << ",";
                }
            parafile << ","  << ",,";
            for (int x = 0; x < xwidth; x++)
                {
                    parafile << x << ",";
                }
            parafile << ","  << ",,";
            for (int x = 0; x < xwidth; x++)
                {
                    parafile << x << ",";
                }

                parafile << "\n";
                
        }
        free(temparray);
    }

};

__device__ int AddOne(int A)
{
    //Functions in CUDA or GPU code start with __device__
    A += 1; 
    return A;
}

__global__ void SetIndex(int *A)
{
    //GPU function that sets adds array index value to index and then adds 1
    int index = threadIdx.x + blockIdx.x*blockDim.x;
    A[index] += index;
    A[index] = AddOne(A[index]);

}
void Verify(int *A,int N)
{
    for (int n=0;n < N; n++)
    {
        std::cout<< A[n] << "\n";
    }
}

int main(void)
{
    //Set array size to 20
    int N = 20;

    //Declare host and device array pointer
    ArrayInt A(20,1,1,"A.csv");

    //Set some initial values to host array
    for (int n = 0; n < N ; n++)
    {
        A.Host[n] = 1;
    }


    //Copy host memory to device memory
    A.CopyH2D();

    //Run Kernel, Using (N/5) blocks with 5 threads per block
    SetIndex<<<N/5,5>>>(A.Device);

    //Synchronizing is not necessary since cudaMemcpy step is next but it never hurts to do so
    cudaDeviceSynchronize();

    //Copy back results
    A.CopyD2H();

    //Verify
    Verify(A.Host,N);

}
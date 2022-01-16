#include <iostream>
#include <fstream>

//Compile Command: nvcc [filename].cu -o [filename].exe
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

class GlobalParameters
{
    public:
    int Blocks;
    int Threads;
    int xwidth;
    int ywidth;
    int zwidth;
    
};

class ArrayInt: public GlobalParameters
{
    public:
    int N;
    int *Host;
    int *Device;
    //int xwidth;
    //int ywidth;
    //int zwidth;
    //const char *Description;


    //Functions

    void SetModelSpec(GlobalParameters ModelSpec)
    {
        N = ModelSpec.xwidth * ModelSpec.ywidth * ModelSpec.zwidth;
        Host = (int*) malloc(N*sizeof(int));
        cudaMalloc((void**) &Device, N*sizeof(int));
        xwidth = ModelSpec.xwidth;
        ywidth = ModelSpec.ywidth;
        zwidth = ModelSpec.zwidth;
    }

    void CopyGPU2CPU ()
    {
        cudaMemcpy(Host, Device, N*sizeof(int), cudaMemcpyDeviceToHost);
    }

    void CopyCPU2GPU ()
    {
        cudaMemcpy( Device, Host, N*sizeof(int), cudaMemcpyHostToDevice);
    }

    void GPUClear()
    {
        cudaFree(Device);
    }

    void GPUInitialize()
    {
        cudaMalloc((void**) &Device, N*sizeof(int));
    }
    
    void GPUZero()
    {
        cudaMemset(Device, 0, N*sizeof(int));
    }

    void CPUClear()
    {
        free(Host);
    }

    void CPUInitialize()
    {
        Host = (int*) malloc(N*sizeof(int));
    }

    void CPUZero()
    {
        memset(Host, 0, N*sizeof(int));
    }


    void DumpTxt(const char *Description)
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

class GPUFunction: public GlobalParameters
{
    public:
    int N;
    ArrayInt AddArray;
    

    void SetIndexP1(GlobalParameters ModelSpec)
    {
        N = AddArray.N;
        AddArray.CopyCPU2GPU();
        SetIndex<<<ModelSpec.Blocks,ModelSpec.Threads>>>(AddArray.Device);
        cudaDeviceSynchronize();
        AddArray.CopyGPU2CPU();
    }

};


int main(void)
{
    //Set global model size and GPU kernel parameters
    GlobalParameters ModelSpec;
    ModelSpec.Blocks =5;
    ModelSpec.Threads= 4;
    ModelSpec.xwidth =20;
    ModelSpec.ywidth =1;
    ModelSpec.zwidth = 1;

    //Create CPU/GPU Array object
    ArrayInt A;
    A.SetModelSpec(ModelSpec);

    //Set some initial values to the CPU array
    for (int n = 0; n < 20 ; n++)
    {
        A.Host[n] = 1;
    }

    //Create GPU function object
    GPUFunction AddGPU;

    //Set GPU function input Array
    AddGPU.AddArray = A;

    //Run the GPU method
    AddGPU.SetIndexP1(ModelSpec);

    //Copy back the GPU attribute
    A = AddGPU.AddArray;
    
    //Verify
    Verify(A.Host,20);

}
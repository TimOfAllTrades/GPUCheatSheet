# 01_CUDA_add

The purpose of this program is to do a basic matrix addition using the GPU.

```C
__device__ int AddOne(int A)
{
    A += 1; 
    return A;
}
```

![alt text](https://github.com/TimOfAllTrades/GPUCheatSheet/blob/master/DesktopDev.png?raw=true)

It is also necessary to set an environment variable before GPU code can be compiled.  For windows 10 if you search view advanced system properties, the below menu should come up.

![alt text](https://github.com/TimOfAllTrades/GPUCheatSheet/blob/master/SysVar.png?raw=true)

The "Path" variable should contain a directory that leads to "cl.exe" found in the Microsoft Visual Studio Community folder.  Once this is complete, it should be possible to compile *.cu files from the command prompt.


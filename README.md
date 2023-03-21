# GPUCheatSheet

Hello

This repository contains some basic codes and functions of CUDA GPU programming and instructions on how to compile into a dll and linking it via C or python.
Contains no proprietary or confidential code of any kind.  All information here is for educational purposes and provided as is.

Module/File             | Description
------------            | -------------
01          | A simple GPU function that does a simple matrix addition.  Contains examples on memory/pointer allocation, copying, launching kernels, defining device and host functions and releasing memory resources.
02 | A simple example showing how to do static and dynamic linking for C++.  (Not so important)
03 | An example on how to dynamically link compiled C++/GPU code to python via ctypes.  Explains how to do pass numpy array pointers, numbers to the C/GPU functions with the proper data type.  Also shows how to use the float type.
04 | An example on how to flatten, pass and then reshape numpy arrays into a compiled DLL for processing and also shows an example of using the 3D grid in CUDA.  The example assumes the numpy array created in python is Fortran ordering, i.e. matrix indicies are such matrix[x][y][z] not matrix[z][y][x] (C ordering).
05 | Several examples on how to use objects with C++

Getting started

System requirements
OS: Windows 10
CPU: Intel or AMD x86/x64 architecture
GPU: Nvidia brand CUDA compatible GPU (pretty much any video card made after 2010)

To make GPU programs, it is necessary to acquire the Nvidia GPU drivers, the CUDA toolit and Microsoft Visual Studio.

GPU drivers can be obtained
https://www.nvidia.com/download/index.aspx

The Nvidia CUDA toolkit can be obtained
https://developer.nvidia.com/cuda-toolkit

Microsoft Visual Studio Community can be obtained
https://visualstudio.microsoft.com/

The order of installation should be the GPU drivers, Microsoft Visual Studio Community and then the Nvidia CUDA toolkit.  In the Visual Studio Community setup, it is important to check "Desktop Development with C++" as that package contains the necessary compiler (it may contain a lot of uncessary stuff too, but I couldn't figure out which packages exactly are needed for compiling CUDA code).

![alt text](https://github.com/TimOfAllTrades/GPUCheatSheet/blob/master/DesktopDev.png?raw=true)

It is also necessary to set an environment variable before GPU code can be compiled.  For windows 10 if you search view advanced system properties, the below menu should come up.

![alt text](https://github.com/TimOfAllTrades/GPUCheatSheet/blob/master/SysVar.png?raw=true)

The "Path" variable should contain a directory that leads to "cl.exe" found in the Microsoft Visual Studio Community folder.  Once this is complete, it should be possible to compile *.cu files from the command prompt.


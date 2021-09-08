# GPUCheatSheet

Hello

This repository contains some basic codes and functions of CUDA GPU programming and instructions on how to compile into a dll and linking it via C or python.
Contains no proprietary or confidential code of any kind.  All information here is for educational purposes and provided as is.

Module/File             | Description
------------            | -------------
01          | A simple GPU function that does a simple matrix addition.  Contains examples on memory/pointer allocation, copying, launching kernels, defining device and host functions and releasing memory resources.
02 | A simple example showing how to do static and dynamic linking for C++.  Not so important
03 | An example on how to dynamically link compiled C++/GPU code to python via ctypes.  Covers how to do pass numpy array pointers, numbers to the C/GPU functions with the proper data type.
#ifndef KERNEL_H
#define KERNEL_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

//This is the header file for the Core CUDA program

#ifdef __cplusplus
extern "C" {
#endif

void __declspec(dllexport) hello(const char *s);

void __declspec(dllexport) world(const char *s);

#ifdef __cplusplus
}
#endif

#endif

/*Instructions 
First compile the kernel.h with the command found in that file
Then compile the main.cu with its respective command
Main.cu then access the library to perform the function
*/
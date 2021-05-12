#include <stdio.h>
#include "kernel.h"

//This is the main CUDA program
//Compile DLL code: nvcc -o [Kernelname].dll --shared [CUDAcodefilename].cu

void hello(const char *s) {
        printf("Hello and %s\n", s);
}

void world(const char *s) {
        printf("World and %s\n" , s);
}


/*Instructions 
First compile the kernel.h with the command found in that file
Then compile the main.cu with its respective command
Main.cu then access the library to perform the function
*/
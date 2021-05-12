#include <stdio.h>

//Compile code: nvcc -o main.exe main.cu -I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.1\include" -L. -lkernel
//Main program to access the dll

#ifdef __cplusplus
extern "C" {
#endif

void __declspec ( dllimport ) hello(const char *s);

#ifdef __cplusplus
}
#endif

int main(void) {
        hello("World");
        return 0;
}


/*Instructions 
First compile the kernel.h with the command found in that file
Then compile the main.cu with its respective command
Main.cu then access the library to perform the function
*/
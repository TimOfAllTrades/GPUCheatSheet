import numpy as np
import ctypes

# This script will show how to properly flatten multidimension (usually 3D) arrays for GPU processing since GPUs handle 1D arrays most efficiently

# Creating a 3D array
indata = np.zeros((3, 3, 3), dtype=int)

# Initialize values
for k in range(0, 3):
    for j in range(0, 3):
        for i in range(0, 3):
            indata[i][j][k] = i + 3*j + 9*k



print(indata[0][2][0], indata[1][2][0], indata[2][2][0],"\t", indata[0][2][1], indata[1][2][1], indata[2][2][1],"\t", indata[0][2][2], indata[1][2][2], indata[2][2][2])
print(indata[0][1][0], indata[1][1][0], indata[2][1][0],"\t", indata[0][1][1], indata[1][1][1], indata[2][1][1],"\t", indata[0][1][2], indata[1][1][2], indata[2][1][2])
print(indata[0][0][0], indata[1][0][0], indata[2][0][0],"\t", indata[0][0][1], indata[1][0][1], indata[2][0][1],"\t", indata[0][0][2], indata[1][0][2], indata[2][0][2])


# flattening all 3D arrays for GPU dll processing
indata = indata.flatten('F').astype(int)

c_int_p = ctypes.POINTER(ctypes.c_int)

testlib = ctypes.CDLL("./GPUkernal.dll")

testlib.GPUAddIntIndex(indata.ctypes.data_as(c_int_p), ctypes.c_int(3), ctypes.c_int(3),ctypes.c_int(3))

indata = np.reshape(indata, (3, 3, 3), order='F')

print(indata[0][2][0], indata[1][2][0], indata[2][2][0],"\t", indata[0][2][1], indata[1][2][1], indata[2][2][1],"\t", indata[0][2][2], indata[1][2][2], indata[2][2][2])
print(indata[0][1][0], indata[1][1][0], indata[2][1][0],"\t", indata[0][1][1], indata[1][1][1], indata[2][1][1],"\t", indata[0][1][2], indata[1][1][2], indata[2][1][2])
print(indata[0][0][0], indata[1][0][0], indata[2][0][0],"\t", indata[0][0][1], indata[1][0][1], indata[2][0][1],"\t", indata[0][0][2], indata[1][0][2], indata[2][0][2])



import numpy as np
import ctypes

#This script will show how to properly flatten multidimension (usually 3D) arrays for GPU processing since GPUs handle 1D arrays most efficiently

indata = np.zeros((3,3,3), dtype = np.int)
xadd = np.zeros((3,3,3), dtype = np.int)
yadd = np.zeros((3,3,3), dtype = np.int)
zadd = np.zeros((3,3,3), dtype = np.int)

#Initialize values
for k in range(0,3):
    for j in range(0,3):
        for i in range(0,3):
            if i == 0:
                xadd[i][j][k] = 100
            if j == 0:
                yadd[i][j][k] = 10
            if k == 0:
                zadd[i][j][k] = 1

#flattening all 3D arrays for GPU dll processing
indata = indata.flatten('F').astype(np.int)
xadd = xadd.flatten('F').astype(np.int)
yadd = yadd.flatten('F').astype(np.int)
zadd = zadd.flatten('F').astype(np.int)

c_int_p = ctypes.POINTER(ctypes.c_int)

testlib = ctypes.CDLL("./GPUkernal.dll")

#Running the dll
testlib.Matadd(indata.ctypes.data_as(c_int_p), xadd.ctypes.data_as(c_int_p), yadd.ctypes.data_as(c_int_p), zadd.ctypes.data_as(c_int_p))

#reshaping back to a 3D array
indata = np.reshape(indata,(3,3,3),order='F' )

#read out data
for k in range(0,3):
    for j in range(0,3):
        print(indata[0][j][k], " ", indata[1][j][k], " ", indata[2][j][k], "\n")
import ctypes

# 1_CUDABmarkDLL.cu must be compiled into a dll file named
# cudadll.dll in the same directory as this script

# DLL Path here, sometimes a full path may be necessary.
testlib = ctypes.CDLL("./cudadll.dll")

# This will call the "sum" function inside 1_CUDABmarkDLL.cu
testlib.sum(ctypes.c_int(3), ctypes.c_int(4))

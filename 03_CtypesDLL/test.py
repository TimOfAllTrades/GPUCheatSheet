import numpy as np
import ctypes



#testlib2 = ctypes.LibraryLoader("kernel.dll") #DLL Path here

indata = np.ones((3), dtype = np.int)
outdata = np.zeros((3), dtype = np.int)
#if float numpy is used, set dtype to np.float32

#Setting a pointer
c_int_p = ctypes.POINTER(ctypes.c_int)
#If float pointer is needed, use ctypes.c_float

#Load DLL
testlib = ctypes.CDLL("./kernel.dll") #DLL Path here, sometimes a full path may be necessary.

#print(testlib.sum(ctypes.c_void_p(indata.ctypes.data), ctypes.c_void_p(outdata.ctypes.data),3,4))
testlib.sum(indata.ctypes.data_as(c_int_p), outdata.ctypes.data_as(c_int_p), ctypes.c_int(3), ctypes.c_int(4))

print(indata)
print(outdata)
print("\n\nNow doing floating numbers\n\n")

infdata = np.ones((20), dtype = np.float32)
outfdata = np.zeros((20), dtype = np.float32)

c_float_p = ctypes.POINTER(ctypes.c_float)

floatlib = ctypes.CDLL("./floatadd.dll")

print(floatlib.Fsum(infdata.ctypes.data_as(c_float_p), outfdata.ctypes.data_as(c_float_p), ctypes.c_float(2.1)))

print(outfdata)
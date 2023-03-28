import numpy as np
import ctypes

# Creating numpy arrays
infdata = np.ones((20), dtype=np.float32)
outfdata = np.zeros((20), dtype=np.float32)

# Declaring pointer type
c_float_p = ctypes.POINTER(ctypes.c_float)

# Loading the DLL
floatlib = ctypes.CDLL("./floatadd.dll")

# Running the DLL with the float type numpy pointers passed through
print(floatlib.Fsum(infdata.ctypes.data_as(c_float_p),
      outfdata.ctypes.data_as(c_float_p), ctypes.c_float(2.1)))

# print out the result
print(outfdata)

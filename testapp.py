import cpp2np
import numpy as np
import ctypes

print(cpp2np.hello())

print("create c array")
c_arr = ((ctypes.c_int*2)*2)()
print(type(c_arr))
print("[["+str(c_arr[0][0]) + " " + str(c_arr[0][1]) + "], [" + str(c_arr[1][0]) + " " + str(c_arr[1][1]) + "]]")

print("create numpy wrapper from same memory buffer")
wrapper = cpp2np.wrap(c_arr, owndata=False, dtype=np.dtype("int32"))
print(type(wrapper))
print(wrapper)

print("change value in numpy array")
wrapper[1, 1] = 4
print(wrapper)

print("observe change in original c array")
print("[["+str(c_arr[0][0]) + " " + str(c_arr[0][1]) + "], [" + str(c_arr[1][0]) + " " + str(c_arr[1][1]) + "]]")

print("deleting wrapper. memory of original c array persists")
del wrapper
print("[["+str(c_arr[0][0]) + " " + str(c_arr[0][1]) + "], [" + str(c_arr[1][0]) + " " + str(c_arr[1][1]) + "]]")

# Code below makes script crash
# print("create numpy wrapper who takes over ownership of data")
# wrapper2 = cpp2np.wrap(c_arr, owndata=True, dtype=np.dtype("int32"))
# del wrapper2
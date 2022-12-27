import cpp2np as c2n
import numpy as np

# -----------------------
print("get pointer on 2x2 std::array of type int allocated by c++:")
pointer, shape = c2n.c_arr_i4()
print(pointer)
print(shape)

# -----------------------
print("\nwrap pointer in numpy array:")
wrapper = c2n.wrap(pointer, shape, dtype=np.dtype("int32"))
print(wrapper)
print("confirm type of wrapper:")
print(type(wrapper))

# -----------------------
print("\nchange value in numpy array:")
wrapper[0,0] = 255
print(wrapper)

# -----------------------
print("\ndelete numpy array and create new wrapper from same pointer:")
del wrapper
wrapper2 = c2n.wrap(pointer, shape, dtype=np.dtype("int32"))
print(wrapper2)
print("(we observe the change of value in first wrapper was done on the original memory buffer,\n" + 
        "as it also shows up in the new wrapper. Also deleting the wrapper did not delete the buffer)")

# -----------------------
print("\nGet information of underlying data of the wrapper:")
print(c2n.descr(wrapper2))

# -----------------------
print("\nTo check if data is contiguous we can look into (ndarray.flags):")
print("C contiguous? " + str(wrapper2.flags['C_CONTIGUOUS']))
print("F contiguous? " + str(wrapper2.flags['F_CONTIGUOUS']))
print("flags overview: \n" + str(wrapper2.flags))

# -----------------------
print("\nNow we explicitly free the memory of the c++ array:")
c2n.free(pointer)
print(wrapper2)
print("We observe that the numpy array is pointing nowhere as the original buffer was freed on the c++ side")
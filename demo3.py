import numpy as np
import cpp2np as c2n

# test print method with a different numpy array
print("\nTest if roundtrip works: create new numpy in python and print it from C++:\n")
new_arr = np.ones((8,8), dtype="int16")

# retrieve pointer 
new_arr_ptr = c2n.descr(new_arr)['data']
print("pointer in python: " + str(new_arr_ptr))

print("\nprint in c++:")
c2n.print_arr(new_arr_ptr)

print("\n print in python:")
print(new_arr)
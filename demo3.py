import numpy as np
import cpp2np as c2n

print("\nTest numpy memory can be accessed and used from C++:\n")
new_arr = np.ones((8,8), dtype="uint8")

# retrieve pointer 
ptr = c2n.descr(new_arr)['data']
print("pointer in python: " + str(ptr))
print("owndata: " + str(new_arr.flags['OWNDATA']))

# disable OWNDATA flag in numpy array to prevent it from deleting the memory
print("-> disable flag")
c2n.owndata(new_arr, False)
print("owndata: " + str(new_arr.flags['OWNDATA']))

# now delete numpy array
del new_arr

# do some stuff in memory to see if data area would get overriden
a = np.zeros((10,10), dtype="double")
b = np.zeros((5,5))

# now print memory the pointer is referencing
print("\nprint numpy data from c++:")
c2n.print_testarr(ptr)

# manually delete data from c++
print("\nnow free python allocated data from c++ and try to print again:")
c2n.py_free(ptr)

# memory stuff happening again
a = np.zeros((10,10), dtype="double")
b = np.zeros((5,5))

c2n.print_testarr(ptr)
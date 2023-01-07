import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import scipy.fftpack as sfft
import cpp2np as c2n

def to_uint8(A):
    A_uint8 = A.astype('int16')  # same size as int16 but unsigned
    for i in range(len(A)):
        for j in range(len(A)):
            if A_uint8[i,j] < 0:
                A_uint8[i, j] += 255
    return A_uint8
                
def quant1(V8,p):
    V8 = V8 - 128
    A = sfft.dctn(V8,norm='ortho')
    Q = p*np.array([[8,16,24,32,40,48,56,64],[16,24,32,40,48,56,64,72],[24,32,40,48,56,64,72,80],[32,40,48,56,64,72,80,88],[40,48,56,64,72,80,88,96],[48,56,64,72,80,88,96,104],[56,64,72,80,88,96,104,112],[64,72,80,88,96,104,112,120]],dtype='uint8')
    AQ = np.rint(A/Q)
    VQ = AQ*Q
    Alow = sfft.idctn(VQ,norm='ortho')
    Alow = Alow + 128
    return Alow

print("-----------------------------------------------------------------------")
print("This demo script reads in a matrix of pixel from C++ and applies image\n" +
      "compression using cosine transformations and low pass filter.")
print("The end result shows the transformed matrix with the same memory address\nas the original.")
print("-----------------------------------------------------------------------")

# read in c++ matrix
print("\n1) Read in pixel from the cpp2np module. The original array is 8x8 array of int16_t:\n")
ptr, shape = c2n.c_arr_pixel()
pixel = c2n.wrap(ptr, shape, dtype=np.dtype('int16'))
print('Pixel array (int16):')
print(type(pixel))
print(c2n.descr(pixel))
print(pixel.flags)
print(pixel)

# create working copy
print("\n2) We create a working copy A and transform it to unsigned int8:\n")
A = to_uint8(pixel)
print('A (uint8):')
print(c2n.descr(A))
print(A)

img = Image.fromarray(A).convert('LA')
plt.figure(1)
plt.imshow(img)
plt.show()


# apply cosine transformation for compression
A = quant1(A, 1)
img_compressed = Image.fromarray(A.astype('uint8')).convert('LA')

print("\n\n3) Cosine transformation:\n")
print(A)
plt.figure(2)
plt.imshow(img_compressed)
plt.show()

# assign changes back to original memory area in c++
print("\n\n4) Transform working copy back to int16 and assign changes to original memory area:\n")

np.place(pixel, np.ones(shape), A.astype('int16').tolist())
print('pixel matrix (int16):')
c2n.print_arr(ptr)

# free memory
res = c2n.free(ptr)

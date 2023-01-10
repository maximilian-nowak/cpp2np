import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import scipy.fftpack as sfft
import cpp2np as c2n

                
def quant1(V8,p):
    V8 = V8.astype('int16') - 128
    A = sfft.dctn(V8, norm='ortho')
    Q = p*np.array([[8,16,24,32,40,48,56,64],[16,24,32,40,48,56,64,72],[24,32,40,48,56,64,72,80],[32,40,48,56,64,72,80,88],[40,48,56,64,72,80,88,96],[48,56,64,72,80,88,96,104],[56,64,72,80,88,96,104,112],[64,72,80,88,96,104,112,120]],dtype='uint8')
    AQ = np.rint(A/Q)
    VQ = AQ*Q
    Alow = sfft.idctn(VQ,norm='ortho')
    Alow = Alow + 128
    return Alow

print("-----------------------------------------------------------------------")
print("This demo script reads in a matrix of pixel from C++ and applies image\n" +
      "compression using cosine transformations.")
print("The end result shows the transformed matrix with the same memory address\nas the original.")
print("-----------------------------------------------------------------------")

# read in pixel from c++ array
print("\n1) Read in pixel from the cpp2np module. The original array is 8x8 array of int16_t:\n")

ptr, shape = c2n.c_arr_pixel()
pixel = c2n.wrap(ptr, shape, dtype=np.dtype('uint8'))

print('Pixel array (uint8):')
print(type(pixel))
print(c2n.descr(pixel))
print(pixel.flags)
print(pixel)

img = Image.fromarray(pixel).convert('LA')
plt.figure(1)
plt.imshow(img)
plt.show()


# apply cosine transformation for compression
print("\n\n3) Compress image with 2D cosine transformation:\n")

A = quant1(pixel, 1)
print(c2n.descr(A))
print(A)

img_compressed = Image.fromarray(A.astype('int8')).convert('LA')
plt.figure(2)
plt.imshow(img_compressed)
plt.show()

# assign changes back to original memory area in c++
print("\n\n4) Transform result back to uint8 and assign changes to original memory area:\n")

np.place(pixel, np.ones(shape), A.astype('uint8').tolist())
print('pixel matrix (uint8):')
c2n.print_testarr(ptr)

# free memory
res = c2n.free(ptr)

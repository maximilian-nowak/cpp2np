import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import scipy.fftpack as sfft
import cpp2np as c2n

def uint8(A):
    A_uint8 = A.astype('int16')
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
            
ptr, shape = c2n.c_arr_pixel()
pixel = c2n.wrap(ptr, shape, dtype=np.dtype('int16'))
print('A (int16):')
print(c2n.descr(pixel))
print(pixel.flags)
print(pixel)

img = Image.fromarray(pixel.astype('uint8')).convert('LA')
plt.figure(1)
plt.imshow(img)

# create working copy
A = uint8(pixel)
print('V8 (uint8):')
print(c2n.descr(A))
print(A)
plt.show()

# apply cosinus transformation for compression
A = quant1(A, 1)
img_compressed = Image.fromarray(A.astype('uint8')).convert('LA')

print("Cosinus Transformation:")
print(A)
plt.figure(2)
plt.imshow(img_compressed)
plt.show()

np.place(pixel, np.ones(shape), A.astype('int16').tolist())
print('\npixel after (int16):')
print(c2n.descr(pixel))
c2n.print_arr(ptr)

res = c2n.free(ptr)

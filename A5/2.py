import cv2
import numpy as np
import matplotlib.image as mpimg
import math as m

def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.144])

def DFT_slow(x):
    x = np.asarray(x, dtype=complex)
    N = x.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.exp(-2j * np.pi * k * n / N)
    return np.dot(M, x)

def FFT_cooley_tukey(x):
    N = x.shape[0]
    if N & (N-1) != 0:
        np.append(x, [0 for i in range(int(m.pow(2,m.ceil(m.log2(N)))) - N)])

    x = np.asarray(x, dtype=complex)
    N = x.shape[0]

    if N <= 4:
        return DFT_slow(x)
    else:
        X_even = FFT_cooley_tukey(x[::2])
        X_odd = FFT_cooley_tukey(x[1::2])
        factor = np.exp(-2j * np.pi * np.arange(N) / N)
        return np.concatenate([X_even + factor[:N // 2] * X_odd,
                               X_even + factor[N // 2:] * X_odd])

img = cv2.imread('lena.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.resize(img, (64, 64), interpolation = cv2.INTER_AREA)
print(img.shape)

fftOut = []

for x in img:
    fftOut.append(FFT_cooley_tukey(x))
fftOut = np.asarray(fftOut, dtype=complex)

for i in range(len(fftOut[0])):
    fftOut[:, i] = FFT_cooley_tukey(fftOut[:, i])


print(fftOut)
print(np.fft.fft2(img))
print(np.allclose(np.fft.fft2(img),fftOut))



#
# cv2.imshow("img", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

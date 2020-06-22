import cv2
import numpy as np
import math as m
import matplotlib.pyplot as plt

def DFT_slow(x):
    x = np.asarray(x, dtype=complex)
    N = x.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.exp(-2j * np.pi * k * n / N)
    out = np.dot(M, x)
    return out/len(out)

def IDFT_slow(x):
    x = np.asarray(x, dtype=complex)
    N = x.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.exp(2j * np.pi * k * n / N)
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
        X_odd /= len(X_odd)
        X_even /= len(X_even)
        factor = np.exp(-2j * np.pi * np.arange(N) / N)
        out = np.concatenate([X_even + factor[:N // 2] * X_odd,
                               X_even + factor[N // 2:] * X_odd])
    return out

def IFFT_cooley_tukey(x):
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
        factor = np.exp(2j * np.pi * np.arange(N) / N)
        return np.concatenate([X_even + factor[:N // 2] * X_odd,
                               X_even + factor[N // 2:] * X_odd])

def FFT2(img):
    img = cv2.imread(img,0)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (256, 256), interpolation = cv2.INTER_AREA)

    fftOut = []
    for x in img:
        fftOut.append(FFT_cooley_tukey(x))
    fftOut = np.asarray(fftOut, dtype=complex)

    for i in range(len(fftOut[0])):
        fftOut[:, i] = FFT_cooley_tukey(fftOut[:, i])

    out=[]
    out.append(fftOut)
    out.append(img)
    return out

fftOutLena = FFT2('lena.png')
fftOutDog = FFT2('dog.png')

lenaPhase = np.angle(fftOutLena[0])
lenaMag = np.log(np.abs(fftOutLena[0]))

lenaPhaseBuiltIn = np.angle(np.fft.fft2(fftOutLena[1]))
# lenaMagBuiltIn = np.log(np.abs(np.fft.fft2(fftOutLena[1])))

dogPhase = np.angle(fftOutDog[0])
dogMag = np.log(np.abs(fftOutDog[0]))

dogPhaseBuiltIn = np.angle(np.fft.fft2(fftOutDog[1]))
# dogMagBuiltIn = np.log(np.abs(np.fft.fft2(fftOutDog[1])))


plt.imshow(lenaMag,cmap='gray')
plt.show()


################################################# Q4
def IFFT(x):
    F=[]
    N = x.shape[0]
    for U in range(0,N):
        sum=0
        for y in range(0,N):
            sum=sum+(x[y]*(np.exp(2j * np.pi * U * y / N)))
        F.append(sum/N)
    return F


#swapping phase and magnitude of lena and dog resp.
swapped = np.multiply(dogMag, np.exp(1j * lenaPhase))
swapped1 = np.multiply(lenaMag, np.exp(1j * dogPhase))

ifftOut = []
ifftOut1 = []
for i in range(len(swapped)):
    ifftOut.append(IFFT(swapped[i]))
    ifftOut1.append(IFFT(swapped1[i]))

ifftOut = np.asarray(ifftOut, dtype=complex)
ifftOut1 = np.asarray(ifftOut1, dtype=complex)

for i in range(len(ifftOut[0])):
    ifftOut[:, i] = IFFT(ifftOut[:, i])
    ifftOut1[:, i] = IFFT(ifftOut1[:, i])

print(ifftOut)
print(ifftOut1)

IFFTcombined = np.asarray(ifftOut)
IFFTcombined1 = np.asarray(ifftOut1)

plt.title('Lena phase + Dog Mag')
plt.imshow( np.real(IFFTcombined),cmap='gray')
plt.show()
plt.title('Lena phase + Dog Mag builtIn')
plt.imshow(np.real(np.fft.ifft2(swapped)),cmap='gray')
plt.show()

plt.title('Dog phase + Lena Mag')
plt.imshow( np.real(IFFTcombined1),cmap='gray')
plt.show()
plt.title('Dog phase + Lena Mag builtIn')
plt.imshow(np.real(np.fft.ifft2(swapped1)),cmap='gray')
plt.show()

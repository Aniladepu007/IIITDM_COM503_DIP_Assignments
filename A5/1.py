import numpy as np
import math as m

def DFT_slow(x):
    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.exp(-2j * np.pi * k * n / N)
    return np.dot(M, x)

def FFT_cooley_tukey(x):
    N = x.shape[0]
    if N & (N-1) != 0:
        np.append(x, [0 for i in range(int(m.pow(2,m.ceil(m.log2(N)))) - N)])

    x = np.asarray(x, dtype=float)
    N = x.shape[0]

    if N <= 4:
        return DFT_slow(x)
    else:
        X_even = FFT_cooley_tukey(x[::2])
        X_odd = FFT_cooley_tukey(x[1::2])
        factor = np.exp(-2j * np.pi * np.arange(N) / N)
        return np.concatenate([X_even + factor[:N // 2] * X_odd,
                               X_even + factor[N // 2:] * X_odd])

x = np.random.random(12)
temp = FFT_cooley_tukey(x)
print(temp)
print(np.allclose(temp, np.fft.fft(x)))

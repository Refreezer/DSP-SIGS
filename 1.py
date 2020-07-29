import time

import numpy as np
import matplotlib.pyplot as plot

def DFT_slow(x):
    
    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.exp(-2j * np.pi * k * n / N)
    return np.dot(M, x)


def FFT(x):
    
    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    
    if N % 2 > 0:
        raise ValueError("size of x must be a power of 2")
    elif N <= 32:  
        return DFT_slow(x)
    else:
        X_even = FFT(x[::2])
        X_odd = FFT(x[1::2])
        factor = np.exp(-2j * np.pi * np.arange(N) / N)
        return np.concatenate([X_even + factor[:N // 2] * X_odd, X_even + factor[N // 2:] * X_odd])


x = np.random.random(1024)

sft_start = time.time()
for i in range(10):
    sft = DFT_slow(x)
sft_finish = time.time()


fft_start = time.time()
for i in range(10000):
    fft = np.fft.fft(x)
fft_finish = time.time()


print(np.allclose(sft, fft))
print('10 sft time ' , str(sft_finish - sft_start))
print('10000 fft time ' , str(fft_finish - fft_start))
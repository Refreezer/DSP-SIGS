import numpy as np
import matplotlib.pyplot as plt

def plotSpectrum(y,Fs):
	n = len(y) # length of the signal
	k = np.arange(n)
	T = n/Fs
	frq = k/T # two sides frequency range
	frq = frq[range(n//2)] # one side frequency range

	Y = np.fft.fft(y)/n # fft computing and normalization
	Y = Y[range(n//2)]

	plt.plot(frq,np.abs(Y),'r') # plotting the spectrum
	plt.xlabel('Freq (Hz)')
	plt.ylabel('|Y(freq)|')


time = np.arange(0, 2 * np.pi, 2 * np.pi /1024);
# f1 = 10
# f2 = 20
# fd = 1024
# w1 = 2*np.pi *f1/fd
# print(w1)
# w2 = 2*np.pi * f2/fd
# print(w2)

period = np.arange(0, 2 * np.pi / 10, 2 * np.pi/ 10 /1024);
amplitude   = np.sin(10 * time) + np.sin(20 * time)
a_period = np.sin(10 * period) + np.sin(20 * period)
fig = plt.figure(figsize=(9, 9))
fig.add_axes([0, 0, 1, 1])

plt.subplot(2, 1, 1)
plt.plot(period, a_period, color = 'blue')
plt.title('A')
plt.xlabel('T')
plt.ylabel('Amplitude = sin(10t) + sin(20t)')

plt.grid(True, which='both')
plt.axhline(y=0, color='k')

plt.subplot(2, 1, 2)
axes = plt.gca()
axes.set_xlim([0, 30]) 
plotSpectrum(amplitude, 1024)

plt.show()

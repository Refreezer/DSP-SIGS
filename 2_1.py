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

Fd = 1024

time = np.arange(0, 2 * np.pi, 2 * np.pi /Fd);
period = np.arange(0, 2 * np.pi, 2 * np.pi/Fd);
amplitude   = np.cos(time) + np.cos(40 * time)
a_period = np.cos(period) + np.cos(40 * period)
fig = plt.figure(figsize=(9, 9))

fig.add_axes([0, 0, 1, 1])

plt.subplot(2, 1, 1)
plt.plot(period, a_period, color = 'blue')
plt.title('A')
plt.xlabel('T')
plt.ylabel('Amplitude = cos(t) + cos(40t)')

plt.grid(True, which='both')
plt.axhline(y=0, color='k')

plt.subplot(2, 1, 2)
axes = plt.gca()
axes.set_xlim([0, 60]) 
plotSpectrum(amplitude, Fd)

plt.show()

import numpy as np
from scipy.fftpack import fft, ifft, fftshift
from scipy.io.wavfile import read
from math import ceil, pi
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def stft(signal, window_size=256):
	window_size = int(window_size)
	length = int(ceil(signal.shape[0] / window_size) * window_size)
	new_signal = np.zeros((length, signal.shape[1]))
	new_signal[0:signal.shape[0]] = signal
	output = np.zeros((int(length/window_size), window_size, signal.shape[1]), dtype='complex')
	assert((length/window_size)%1.0 == 0)
	for i in range(0, int(length/window_size)):
		sub_signal = new_signal[window_size*i:window_size*(i+1)]
		output[i] = fft(sub_signal)[0:window_size]
	return output

def istft(signal):
	length = signal.shape[1] * signal.shape[0]
	output = np.zeros((length, signal.shape[2]))
	for i in range(0, signal.shape[0]):
		for ch in range(signal.shape[2]):
			output[i*signal.shape[1]:(i+1)*signal.shape[1]] = ifft(signal[i]).real
	return output

sr, signal = read('440hz.wav')
if signal.dtype == np.int16:
	signal = np.asarray(signal, dtype=np.float) / pow(2, 15)

signal = np.asarray(signal, dtype=np.float)
res = stft(signal, window_size=sr/50) # 44100 / 50 = 882 frequency bins

back = istft(res)

left = res[:,:,0]


def drawFFT(signal, sr=44100):

	frequencies = fft(signal)

	phase_information = frequencies.imag
	magnitude_information = np.abs(frequencies)
	freq = np.argmax(magnitude_information)

	t = sr / signal.shape[0]
	print(freq*t)
	print(phase_information[freq])
	print(freq*t+phase_information[freq])
	exit()

	fig, ax = plt.subplots(1, 1)
	length = fft_sig.shape[0]
	fft_sig = fft_sig[:length//2]

	phase_information = fft_sig.imag
	magnitude_information = np.abs(fft_sig)
	
	freq = np.argmax(magnitude_information)
	print(fft_sig.shape)
	print(freq)
	ax.plot(magnitude_information)
	ax.set_xscale('log')
	plt.show()


test_sine = np.sin(326*2*pi*np.arange(0,2.0,1.0/44100))

print()

'''freq = np.argmax(ft)
print(freq)

test_signal = signal[0:44100,0]
plt.plot(test_signal[0:800])
plt.plot(test_sine[0:800])
plt.show()
'''
# 10 ms window
drawFFT(test_sine[0:441])
import numpy as np
import matplotlib.pyplot as plt
from math import pow, ceil
from misc import resize, gaussian
from scipy.io.wavfile import write
from scipy.signal import stft

size = (10,20) # 10 x 20 meters
freq = 10000   # max frequency we want to simulate
c = 340        # sound travels at 340 m/s
wallLocation = 10
wallWidth = 1
wallHeight = 8

source = (2, 3)
dest = (2, 17)

''' step sizes '''

k = 1 / (freq*2)
h = c / freq
rhsq = 1.0 / pow(h, 2)

''' actual domain sizes '''

shape = (int(ceil(size[0] / h)), int(ceil(size[1] / h)))
sourceTransformed = (int(shape[0] * source[0] / size[0]), int(shape[1] * source[1] / size[1]))
destTransformed = (int(shape[0] * dest[0] / size[0]), int(shape[1] * dest[1] / size[1]))

''' data '''

position = np.zeros(shape)
velocity = np.zeros(shape)
forces = np.zeros(shape)

forces = gaussian(sourceTransformed, 1, shape) * 3e9

walls = np.ones(shape)
wallStart = int(shape[1] * (wallLocation - wallWidth / 2) / size[1])
wallEnd = int(shape[1] * (wallLocation + wallWidth / 2) / size[1])
wallHeightT = int(shape[0] * wallHeight / size[0])
walls[:wallHeightT,wallStart:wallEnd] = 0.1

def laplacian(pos, r):

	Ux = np.zeros_like(pos)
	UTopRightToBottomLeft = np.zeros_like(pos)
	UTopLeftToBottomRight = np.zeros_like(pos)

	shape = pos.shape
	pos = np.pad(pos, 1, mode='constant')

	
	for y in range(0, shape[1]):
		for x in range(0, shape[0]):
			Ux[x,y] = pos[x][y+1] + pos[x+2, y+1] - 2 * pos[x+1, y+1]
			if y % 2 == 0: # even
				UTopLeftToBottomRight[x,y] = pos[x+1, y] + pos[x+2, y+2] - 2 * pos[x+1, y+1]
				UTopRightToBottomLeft[x,y] = pos[x+2, y] + pos[x+1, y+2] - 2 * pos[x+1, y+1]
			else: # odd
				UTopLeftToBottomRight[x,y] = pos[x, y] + pos[x+1, y+2] - 2 * pos[x+1, y+1]
				UTopRightToBottomLeft[x,y] = pos[x+1, y] + pos[x, y+2] - 2 * pos[x+1, y+1]
	res = Ux / r + UTopLeftToBottomRight / r + UTopRightToBottomLeft / r
	return res

velocity = velocity + 0.5 * k * (forces + c * c * laplacian(position, rhsq))

input = []
output = []

''' pyplot '''

fig, ax = plt.subplots()
plt.ion()
plt.show()

from datetime import datetime
start = datetime.now()

I = int(freq)
for i in range(0,I):

	position = position + k * velocity * walls

	#f = forces * np.sin(10 * 2 * np.pi * float(i) / 10000)
	velocity = velocity + k * (c * c * laplacian(position, rhsq))

	if i % 100 == 0:
		ax.imshow(position, cmap='magma', vmin=-1, vmax=1)
		ax.scatter(np.array([sourceTransformed[1], destTransformed[1]]), np.array([sourceTransformed[0], destTransformed[0]]))
		plt.draw()
		plt.pause(0.01)
		plt.cla()
		
		elapsed = datetime.now() - start
		precentageComplete = i / I
		if precentageComplete > 0:
			eta = elapsed.seconds * (1 - precentageComplete) / precentageComplete
			if eta > 60:
				print('eta: {} minutes'.format(int(eta/60)))
			else:
				print('eta: {} seconds'.format(int(eta)))

	input.append(position[sourceTransformed[0], sourceTransformed[1]])
	output.append(position[destTransformed[0], destTransformed[1]])

plt.close()
plt.ioff();

input = np.array(input)
output = np.array(output)

write("input.wav", freq * 2, input)
write("output.wav", freq * 2, output)

f, testc, Zxx = stft(output, fs = 16)

NFFT = 200     # the length of the windowing segments
Fs = freq  # the sampling rate

# plot signal and spectrogram

fig, ax = plt.subplots(2,2)

ax[0,0].plot(input)   # for this one has to either undersample or zoom in 
ax[0,1].plot(output)   # for this one has to either undersample or zoom in 

PxxI, freqsI, binsI, imI = ax[1,0].specgram(input, NFFT=NFFT,   Fs=Fs,noverlap=100, cmap=plt.cm.gist_heat, scale='dB')
PxxO, freqsO, binsO, imO = ax[1,1].specgram(output, NFFT=NFFT,   Fs=Fs,noverlap=100, cmap=plt.cm.gist_heat, scale='dB')

plt.show()


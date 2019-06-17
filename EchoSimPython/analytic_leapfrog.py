import numpy as np
import matplotlib.pyplot as plt
from transformtest import dct2, idct2
from math import pi, cos
from scipy.misc import imresize
from misc import resize, gaussian

''' Simulation parameters '''

dimensionSize = (10, 10) # 10 * 10 meters
maxFrequency = 1378
speedOfSound = 340

''' Initialization '''

#fig, ax = plt.subplots()
#plt.ion()
#plt.show()

stepSize = float(speedOfSound) / maxFrequency
shape = (int(dimensionSize[0] / stepSize), int(dimensionSize[1] / stepSize))
position = np.zeros(shape)
velocity = np.zeros(shape)
forces = np.zeros(dimensionSize)
forces[2:4,2:4] = 1e3
k = 1.0 / (1378 * 2)


plt.ion()
fig, ax = plt.subplots()

''' Convert to frequency domain '''

fPosition = np.zeros(shape)
fVelocity = np.zeros(shape)
fForces = gaussian((shape[0]//2,shape[1]//2), 10, shape)

xs = np.arange(0, shape[0], 1)
ys = np.arange(0, shape[1], 1)

g = pi * np.sqrt(np.array([pow(xs[ix],2) + pow(ys[iy],2) for ix,iy in np.ndindex(shape)]).reshape(shape))
w = speedOfSound * g
multipliers = np.cos(w*k)

fVelocity = fVelocity + 0.5 * k * (dct2(fForces) - np.power(w, 2) * fPosition)

for i in range(0, 10000):

	fPosition = fPosition + k * fVelocity
	fVelocity = fVelocity + k * (fForces - np.power(w, 2) * fPosition)

	ax.imshow(idct2(fPosition), cmap='magma', vmin=-1, vmax=1)
	plt.draw()
	plt.pause(0.1)
	plt.cla()

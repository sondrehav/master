import numpy as np
import matplotlib.pyplot as plt
import math
from misc import InteractivePlot, gaussian

def hlaplacian(pos):

	
	conv = np.array([2, -27, 270, -490, 270, -27, 2]) / 180
	#conv = np.array([1, -2, 1])

	padValue = len(conv)//2
	pos = np.pad(pos, padValue, mode='constant')

	Ux = np.zeros_like(pos)
	UTopRightToBottomLeft = np.zeros_like(pos)
	UTopLeftToBottomRight = np.zeros_like(pos)

	shape = pos.shape
	
	for y in range(padValue, shape[1] - padValue):
		Ux[:,y] = np.convolve(pos[:,y], conv, mode='same')
		for x in range(padValue, shape[0] - padValue):
			xs1 = 0.0
			xs2 = 0.0
			for z in range(-padValue, padValue+1):
				xm1 = 0
				xm2 = 0
				if y % 2 == 1: # even 
					xm1 = math.ceil(z / 2.0);
					xm2 = math.ceil(-z / 2.0);
				else: # odd
					xm1 = math.floor(z / 2.0);
					xm2 = math.floor(-z / 2.0);
				xs1 += pos[x+xm1,y+z] * conv[z + padValue]
				xs2 += pos[x+xm2,y+z] * conv[z + padValue]
			UTopLeftToBottomRight[x, y] = xs1
			UTopRightToBottomLeft[x, y] = xs2
	#d = math.sqrt(1.25)
	d = 1.0
	res = Ux + UTopLeftToBottomRight / d + UTopRightToBottomLeft / d
	return res[padValue:-padValue,padValue:-padValue]

def laplacian(pos):

	Ux = np.zeros_like(pos)
	Uy = np.zeros_like(pos)

	conv = np.array([2, -27, 270, -490, 270, -27, 2]) / 180
	
	for x in range(0, shape[0]):
		Ux[x,:] = np.convolve(pos[x,:], conv, 'same')
	for y in range(0, shape[1]):
		Uy[:,y] = np.convolve(pos[:,y], conv, 'same')
	res = Ux + Uy
	return res

hshape = (128, 148)
shape = (128, 128)

timeStep = 1.0/44100.0
soundVelocity = 340.0
spaceStep = 3.4e-2

hpressure = np.zeros(hshape)
hvelocity = np.zeros(hshape)
hforce = np.zeros(hshape)
hforce[hshape[0]//2,hshape[1]//2] = 1e9 * 1.4657

hvelocity = hvelocity + timeStep * 0.5 * (hforce + soundVelocity * soundVelocity * hlaplacian(hpressure) / spaceStep)

pressure = np.zeros(shape)
velocity = np.zeros(shape)
force = np.zeros(shape)
force[shape[0]//2,shape[1]//2] = 1e9

velocity = velocity + timeStep * 0.5 * (force + soundVelocity * soundVelocity * laplacian(pressure) / spaceStep)

fig, ax = plt.subplots(1, 2)

for i in range(0, 800):

	pressure = pressure + timeStep * velocity
	velocity = velocity + timeStep * soundVelocity * soundVelocity * laplacian(pressure) / spaceStep

	hpressure = hpressure + timeStep * hvelocity
	hvelocity = hvelocity + timeStep * soundVelocity * soundVelocity * hlaplacian(hpressure) / spaceStep
	
	print('{}, {}, {}'.format(pressure.min(), pressure.max(), i))

ax[0].imshow(hpressure, cmap='magma', vmin=-1, vmax=1)
ax[1].imshow(pressure, cmap='magma', vmin=-1, vmax=1)
plt.show()
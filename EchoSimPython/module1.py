import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct
from math import sqrt, exp, pi

dctType = 2

def normDist(shape, pos, size):
	return np.array([exp(-(pow((x - pos[0]), 2) + pow((y - pos[1]), 2))/(2*size*size)) / (size * sqrt(2 * pi)) for y, x in np.ndindex(shape)]).reshape(shape)

def dct2(array):
	return dct(dct(array, axis=0, type=dctType) / (2 * array.shape[0]), axis=1, type=dctType) / (2 * array.shape[1])

def idct2(array):
	return idct(idct(array, axis=1, type=dctType), axis=0, type=dctType)

def gr_right(a, i):
    rightHand = a[i:][::-1]
    g = np.zeros_like(a)
    g[i-len(rightHand):i] = rightHand
    mask = np.ones_like(g)
    mask[i:] = 0
    return mask * (g + a)

def residual_right(a, i):
	return a - gr_right(a, i)

def gr_left(a, i):
    leftHand = a[:i][::-1]
    g = np.zeros_like(a)
    g[i:i+len(leftHand)] = leftHand
    mask = np.ones_like(g)
    mask[:i] = 0
    return mask * (g + a)

def residual_left(a, i):
	return a - gr_left(a, i)

SPEED_OF_SOUND = 340

class Solver:

	def __init__(self, shape, timeStep, forces):

		self.ki = np.sqrt(np.array([pow(pi * (x + 1) / shape[0], 2) + pow(pi * (y + 1) / shape[1], 2) for x, y in np.ndindex(shape)]).reshape(shape))
		self.wi = SPEED_OF_SOUND * self.ki
		self.coswidt = np.cos(self.wi * timeStep)

		self.forces = forces

	def __iter__(self):
		
		position = np.zeros(shape)
		lastPosition = np.zeros(shape)

		forcingTerm = 2 * dct2(self.forces) * (1.0 - self.coswidt) / np.power(self.wi, 2)
		newPosition = 2 * position * self.coswidt - lastPosition + forcingTerm
		lastPosition = position
		position = newPosition

		while True:
			forcingTerm = 2 * dct2(self.forces) * (1.0 - self.coswidt) / np.power(self.wi, 2)
			newPosition = 2 * position * self.coswidt - lastPosition + forcingTerm
			lastPosition = position
			position = newPosition
			yield position 

	def setForces(self, forces):
		self.forces = forces


fig, ax = plt.subplots()
plt.ion()
plt.show()

shape = (438, 128)
forces = normDist(shape, (60, 60), 10) * 1e11
forces2 = normDist(shape, (128-60, 438-60), 10) * 1e11

i = 0
timeStep = 0.001

left = Solver(shape, timeStep, forces)
right = Solver(shape, timeStep, forces2)

for l, r in zip(left, right):
	
	
	output = np.hstack((idct2(l), idct2(r)))  / (2 * shape[0] * shape[1])

	leftForces = np.zeros(shape)
	rightForces = np.zeros(shape)
	
	conv = np.array([2, -27, 270, -490, 270, -27, 2]) / 180.0
	
	for u in range(0, 3):
		residual = residual_right(conv, u + 4)
		for y in range(0, shape[0]):
			value = np.convolve(residual, output[y,shape[1]-3-u:shape[1]+4-u], mode='valid')[0]
			leftForces[y, -u-1] = value * SPEED_OF_SOUND * SPEED_OF_SOUND / pow(shape[0], 2)
			
	for u in range(0, 3):
		residual = residual_left(conv, 3 - u)
		for y in range(0, shape[0]):
			value = np.convolve(residual, output[y,shape[1]-3-u:shape[1]+4-u], mode='valid')[0]
			rightForces[y, u] = value * SPEED_OF_SOUND * SPEED_OF_SOUND / pow(shape[0], 2)
	
	left.setForces(leftForces)
	right.setForces(rightForces)
	
	if i % 100 == 0:
		ax.imshow(output, cmap='magma', vmin=-1, vmax=1)
		plt.draw()
		plt.pause(0.01)
		plt.cla()
		print(i)
	i = i+1
	if i > 10000:
		break
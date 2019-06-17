import numpy as np
from scipy.ndimage import imread
import matplotlib.pyplot as plt
import random
from collections import deque
from scipy.misc import imresize
from collections import deque
from math import pi, cos
from transformtest import dct2, idct2

def shift(a, n):
	b = np.roll(a, n)
	if n > 0:
		b[:n] = 0
	else:
		b[n:] = 0
	return b

def findStart(array):
	shape = array.shape
	indices = []
	random.seed(8734)
	for x in range(0, shape[0]):
		for y in range(0, shape[1]):
			if array[x, y]:
				indices.append((x, y))
	random.shuffle(indices)
	q = deque(indices)
	
	def expand(rectangle, direction):
		((x0, y0),(x1, y1)) = rectangle
		if direction == 0: # up
			if np.all(array[x0:x1, y0-1:y0]):
				array[x0:x1, y0-1:y0] = False
				return ((x0, y0-1),(x1, y1)), False
		if direction == 1: # right
			if np.all(array[x1:x1+1, y0:y1]):
				array[x1:x1+1, y0:y1] = False
				return ((x0, y0),(x1+1, y1)), False
		if direction == 2: # down
			if np.all(array[x0:x1, y1:y1+1]):
				array[x0:x1, y1:y1+1] = False
				return ((x0, y0),(x1, y1+1)), False
		if direction == 3: # left
			if np.all(array[x0-1:x0, y0:y1]):
				array[x0-1:x0, y0:y1] = False
				return ((x0-1, y0),(x1, y1)), False
		return rectangle, True

	while q:
		index = q.popleft()
		if not array[index[0], index[1]]:
			continue
		rectangle = ((index[0], index[1]), (index[0] + 1, index[1] + 1))
		array[index[0], index[1]] = True
		direction = 0
		failureCount = 0
		while True:
			rectangle, failed = expand(rectangle, direction)
			if failed:
				failureCount += 1
				if failureCount >= 4:
					# done
					yield rectangle
					break
			else:
				failureCount = 0
			direction = (direction + 1) % 4


def loadDomain(path):
	return imread(path).astype('bool')

def getPartitions(array, solver):
	p = []
	shape = array.shape
	array2 = np.zeros((shape[0], shape[1]), dtype='int')
	for rect in findStart(array):
		((x0, y0), (x1, y1)) = rect
		partition = Partition((x0, y0), (x1, y1), solver)
		array2[x0:x1,y0:y1] = partition.__hash__()
		p.append(partition)
	return p

def visualize(partitions, shape):
	array2 = np.zeros((shape[0], shape[1], 3), dtype='int')
	color = np.random.uniform(0, 255, 3)
	for r in partitions:
		array2[r.lowerBoundary[0]:r.upperBoundary[0], r.lowerBoundary[1]:r.upperBoundary[1]] = color
		color = np.random.uniform(0, 255, 3)
	plt.imshow(array2)
	plt.show()

class Partition:
	def __init__(self, lowerBoundary, upperBoundary, parent):

		self.parent = parent
		
		self.lowerBoundary = lowerBoundary
		self.upperBoundary = upperBoundary
		
		self.shape = (self.upperBoundary[0] - self.lowerBoundary[0], self.upperBoundary[1] - self.lowerBoundary[1])

		self.topNeighbours = set()
		self.rightNeighbours = set()
		self.bottomNeighbours = set()
		self.leftNeighbours = set()
		
		xs = np.arange(0, self.shape[0], 1) + 1
		ys = np.arange(0, self.shape[1], 1) + 1
		scaling_x = float(self.shape[1]) / max(self.shape[0], self.shape[1])
		scaling_y = float(self.shape[0]) / max(self.shape[0], self.shape[1])
		self.kValues = pi * np.sqrt(np.array([pow(xs[ix] * scaling_x,2) + pow(ys[iy] * scaling_y,2) for ix,iy in np.ndindex(self.shape)]).reshape(self.shape))
		
		self.fValues = np.zeros(self.shape)
		self.fLastValues = np.zeros(self.shape)

	def __hash__(self):
		# todo: replace 100000 with domain width...
		return self.lowerBoundary[0] * 100000 + self.lowerBoundary[1]

	def getShape(self):
		return self.shape

	def getValues(self):
		return idct2(self.fValues)

	
	def neighbours(self, positions, rhsq):
		conv = np.array([2, -27, 270, -490, 270, -27, 2]) / 180
		U = np.zeros(self.shape)

		for y in range(self.lowerBoundary[1], self.upperBoundary[1]):
			
			y_t = y - self.lowerBoundary[1]

			e1 = (shift(conv, 1) + conv) * np.array([1,1,1,1,0,0,0])
			r1 = conv - e1
			c1 = np.convolve(r1, positions[-4+self.upperBoundary[0]:3+self.upperBoundary[0],y], 'valid')
			U[-1,y_t] = c1[0]
		
			e2 = (shift(conv, 2) + conv) * np.array([1,1,1,0,0,0,0])
			r2 = conv - e2
			c2 = np.convolve(r2, positions[-5+self.upperBoundary[0]:2+self.upperBoundary[0],y], 'valid')
			U[-2,y_t] = c2[0]

			e3 = (shift(conv, 3) + conv) * np.array([1,1,0,0,0,0,0])
			r3 = conv - e3
			c3 = np.convolve(r3, positions[-6+self.upperBoundary[0]:1+self.upperBoundary[0],y], 'valid')
			U[-3,y_t] = c3[0]

		U = U / rhsq

		return U



	def step(self, externalForces, positions, delta_t = 1.0 / 1378.0, c = 340):
		w = c * self.kValues
		mult = np.cos(w*delta_t)
		force = externalForces[self.lowerBoundary[0]:self.upperBoundary[0],\
			self.lowerBoundary[1]:self.upperBoundary[1]]
		neighbouringForces = self.neighbours(positions, pow(1.0 / 64, 2))
		
		forceTerm = 2.0 * dct2(force + neighbouringForces * c * c) * (1.0 - mult) / (np.power(w, 2))
		fNew = 2 * self.fValues * mult - self.fLastValues + forceTerm
		self.fLastValues = self.fValues
		self.fValues = fNew

class Solver:
	def __init__(self, file):
		domain = loadDomain(file).T
		self.partitions = getPartitions(domain, self)
		self.shape = domain.shape
		self.neighbourForces = np.zeros(self.shape)
		self.externalForces = np.zeros(self.shape)
		self.speedOfSound = 340
		self.deltaT = 1.0 / 44100.0

	def step(self):
		values = self.getValues()
		for part in self.partitions:
			part.step(self.externalForces, values, self.deltaT, self.speedOfSound)

	def getValues(self):
		output = np.zeros(self.shape)
		for part in self.partitions:
			output[part.lowerBoundary[0]:part.upperBoundary[0],\
				part.lowerBoundary[1]:part.upperBoundary[1]] = part.getValues()
		return output

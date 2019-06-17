import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct
from math import pi

SPEED_OF_SOUND = 340.0

fig, ax = plt.subplots()
plt.ion()
plt.show()

timings = {}

def timing(f):
    def wrap(*args):
        time1 = time.time()
        ret = f(*args)
        time2 = time.time()
        name = type(args[0]).__name__
        if name not in timings:
            timings[name] = time2 - time1
        timings[type(args[0]).__name__] = timings[type(args[0]).__name__] * 0.98 + 0.02 * (time2 - time1)
        return ret
    return wrap

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



class FDTD:

	def __init__(self, lower, upper, hx):
		self.length = upper - lower
		self.lower = lower
		self.upper = upper
		self.hx = hx
		self.arrayLength = int(self.length/hx)
		self.arrayLower = int(self.lower/hx)
		self.arrayUpper = int(self.upper/hx)
		self.positions = np.zeros(self.arrayLength)
		self.velocities = np.zeros(self.arrayLength)
		self.firstStep = False

	def setInitial(self, positions):
		self.positions = positions

	@timing
	def step(self, forces, timeStep):
		conv = np.array([2, -27, 270, -490, 270, -27, 2]) / 180
		if not self.firstStep:
			self.velocities = 0.5 * timeStep * (forces[self.arrayLower:self.arrayUpper] + SPEED_OF_SOUND * SPEED_OF_SOUND * np.convolve(conv, self.positions, mode='same') / pow(self.hx, 2))
			self.firstStep = True
		else:
			self.velocities += timeStep * (forces[self.arrayLower:self.arrayUpper] + SPEED_OF_SOUND * SPEED_OF_SOUND * np.convolve(conv, self.positions, mode='same') / pow(self.hx, 2))
		self.positions += timeStep * self.velocities

	def getValues(self):
		return self.positions

class Analytic:
	def __init__(self, lower, upper, hx):
		self.length = upper - lower
		self.lower = lower
		self.upper = upper
		self.hx = hx
		self.arrayLength = int(self.length/hx)
		self.arrayLower = int(self.lower/hx)
		self.arrayUpper = int(self.upper/hx)
		self.positions = np.zeros(self.arrayLength)
		self.lastPositions = np.zeros(self.arrayLength)
		self.ki = pi * (np.arange(0, self.arrayLength, 1) + 1) / self.arrayLength
		self.wi = SPEED_OF_SOUND * self.ki
		self.outPos = np.zeros_like(self.positions)

	def setInitial(self, positions):
		self.positions = dct(positions, type=1, norm='ortho') / self.length
		

	@timing
	def step(self, forces, timeStep):
		coswidt = np.cos(self.wi * timeStep)
		forcingTerm = 2 * dct(forces[self.arrayLower:self.arrayUpper], type=1, norm='ortho') * (1.0 - coswidt) / np.power(self.wi, 2)
		newPositions = 2 * self.positions * coswidt - self.lastPositions + forcingTerm
		self.lastPositions = self.positions
		self.positions = newPositions
		self.outPos = idct(self.positions, type=1, norm='ortho')
		
	def getValues(self):
		return self.outPos

hx = 0.1
length = 100

forces = np.zeros(2*int(length/hx))

fdtd = FDTD(0, length, hx)
ts = 1/44100

analyticLeft = Analytic(0, length, hx)

initialPosition = np.array([((x / (int(length/hx))) if (x < int(length/(2*hx))) else (1 - x / (int(length/hx)))) for x in range(0, int(length/hx))])
analyticLeft.setInitial(initialPosition)
fdtd.setInitial(initialPosition)


for i in range(0, 10000):

	fdtd.step(forces, ts)

	analyticLeft.step(forces, ts)
	forces *= 0.9

	
	if i % 10 == 0:
		ax.set_ylim([-1, 1])
		ax.plot(fdtd.getValues(), color='blue')
		ax.plot(analyticLeft.getValues(), color='red')
		plt.draw()
		plt.pause(0.01)
		plt.cla()
		
		for item in timings:
			print('{:s} function took {:.3f} ms'.format(item, timings[item]*1000.0))

import numpy as np
import matplotlib.pyplot as plt
from math import pi, cos
from transformtest import dct2, idct2
from misc import gaussian
from interface import Solver

solver = Solver('files/domain6.bmp')

force = gaussian((87, 87), 3, solver.shape) * 3e9
solver.externalForces = force

fig, ax = plt.subplots()
plt.ion()
plt.show()

for i in range(0, 1000):
	solver.step()
	output = solver.getValues()
	
	ax.imshow(output, cmap='magma', vmin=-1, vmax=1)
	plt.draw()
	plt.pause(0.01)
	plt.cla()

	if i == 0:
		solver.externalForces[:,:] = 0


	'''
	forcesLeft[:,:] = 0
	forcesRight[:,:] = 0
	
	posRight = (left.shape[0], 0)
	rightShape = right.shape
	
	posLeft = (0, 0)
	leftShape = left.shape
	
	forcesRight = force[posRight[0]:posRight[0]+rightShape[0],posRight[1]:posRight[1]+rightShape[1]]
	forcesLeft = force[posLeft[0]:posLeft[0]+leftShape[0],posLeft[1]:posLeft[1]+leftShape[1]]
	
	h = 1.0 / 64

	additionalForcesRight = - output[posRight[0],:] + output[posRight[0]-1,:]
	forcesRight[0,:] += pow(speedOfSound, 2) * additionalForcesRight / pow(h, 2)

	additionalForcesLeft = - output[posRight[0]-1,:] + output[posRight[0],:]
	forcesLeft[-1,:] += pow(speedOfSound, 2) * additionalForcesLeft / pow(h, 2)
	'''

print(force.shape)
print('k')

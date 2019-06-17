import numpy as np
import matplotlib.pyplot as plt
from math import pi, sin, cos
from transformtest import dct2, idct2, generateNoise
from solver import solver
from interface import loadDomain, getPartitions, visualize, Solver
from misc import gaussian




solver = Solver('files/domain3.bmp')

M = gaussian((32, 32), 1, solver.shape) * 10e9
solver.externalForces =  M


fig, ax = plt.subplots()
plt.ion()
plt.show()

wave = []

i = 0
for i in range(0,1000):
	solver.step()
	solver.externalForces *=  0.5
	if i % 100 == 0:
		ax.imshow(solver.getValues(), cmap='magma', vmin=-1, vmax=1)
		plt.draw()
		plt.pause(0.01)
		plt.cla()



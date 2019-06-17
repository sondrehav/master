import numpy as np
import matplotlib.pyplot as plt
from math import pi, sin
from scipy.io.wavfile import write


c = 1				# stiffness
k = 0.1				# timestep
XY = 198			# unknowns in x and y direction
S = 60				# size in xy

h = XY / (S + 1) 	# stepsize
r = c * k / h

D = np.zeros((XY + 2, XY + 2))
np.fill_diagonal(D[1:], np.ones(XY + 1))
np.fill_diagonal(D[:,1:], np.ones(XY + 1))
np.fill_diagonal(D, -2*np.ones(XY))

U = np.zeros((XY + 2, XY + 2))
U[(XY+2)//2,(XY+2)//2] = 10

Ul = U
from random import uniform


fig, ax = plt.subplots()
plt.ion()
plt.show()

for i in range(0, 10000):
	
	Un = pow(r,2) * (U.dot(D) + D.dot(U)) + 2 * U - Ul
	#plot ...

	if i % 100 == 0:
		ax.imshow(Un, vmin=-2, vmax=2, cmap='magma')
		plt.draw()
		plt.pause(0.0001)
		plt.cla()

	# end
	Ul = U
	U = Un

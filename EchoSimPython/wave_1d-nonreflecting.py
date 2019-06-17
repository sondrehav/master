import numpy as np
import matplotlib.pyplot as plt

xdim = 100
k = 1.0 / 1378.0
c = 340
h = 1.0 / xdim

U = np.zeros(xdim)
Ul = U
Un = np.zeros_like(U)

fig, ax = plt.subplots()
plt.ion()
plt.show()

for i in range(0, 1000):

	F = np.zeros_like(U)
	if i == 0:
		F[50] = 50
	elif i == 1:
		F[50] = -50
	velocity = U - Ul

	r = c * k

	
	conv = np.array([2, -27, 270, -490, 270, -27, 2]) / 180
	Un = (r*r*(np.convolve(U, conv, 'same') + F) + U + velocity)

	Un[0] = (2*r*r*U[1] + 2*(1-r*r)*U[0]-(1-r)*Ul[0]) / (1 + r)
	Un[-1] = (2*r*r*U[-2] + 2*(1-r*r)*U[-1]-(1-r)*Ul[-1]) / (1 + r)

	Ul = U
	U = Un


	if i % 10 == 0:
		ax.set_ylim([-10, 10])
		ax.plot(U)
		plt.draw()
		plt.pause(0.01)
		plt.cla()


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

F = np.zeros_like(U)
F[20:25] = -0.05
F[75:80] = 0.05

for i in range(0, 10000):
	
	velocity = U - Ul

	r = c * k
	Un = r*r*(np.convolve(U, np.array([1,-2,1]), 'same') + F) + U + velocity

	Ul = U
	U = Un

	if i % 10 == 0:
		ax.set_ylim([-10, 10])
		ax.plot(U)
		plt.draw()
		plt.pause(0.01)
		plt.cla()



import numpy as np
import matplotlib.pyplot as plt
from math import pi, sin, cos, sqrt, exp
from scipy.fftpack import dst

L = 101
phi = 7.0
alpha = 50
c = 1

f = lambda x: np.exp(-np.power(x-alpha,2)/(2*np.power(phi, 2))) / np.sqrt(2*pi*np.power(phi, 2))

A = np.fromfunction(f, (L,))
A[0] = A[-1] = 0

def fn(t, F):
	xs = np.arange(0, L, 1)
	fF = dst(F)
	for x in xs:
		xs[x] = 0
		for n in range(0, F.shape[0]):
			xs[x] += fF[x] * cos(c*pi*n*t) * sin(pi*n*x/L)
	return xs

fig, ax = plt.subplots(2)
ax[0].plot(A)
ax[1].plot(fn(0, A))
plt.show()
exit()

fig, ax = plt.subplots()
plt.ion()
plt.show()

for i in range(0,100):
	ax.plot(fn(i/10.0, A))
	ax.set_ylim(-10, 10)
	plt.draw()
	plt.pause(1)
	plt.cla()
	print(i)

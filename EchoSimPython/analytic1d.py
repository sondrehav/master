import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct
from math import pi

def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

size = 256
pressure = np.zeros((3, size))
force = np.array([gaussian(x, 5*size//8, 10) for x in range(0, size)]) * 1e2

#force = (2 * np.random.rand(size) - 1) * 1e2

ki = pi * np.arange(1, size + 1, 1) / size
wi = 343 * ki

sigma = np.zeros_like(force)
sigma[-1] = 1

absorber = 1 / (1 + 1j * dct(sigma) / wi)


ts = 1.0 / 4410.0

multiply = np.cos(wi * ts)
wi2 = np.power(wi, 2)

fig, ax = plt.subplots()
plt.ion()
plt.show()

for i in range(0, 10000):
	pressure[(i + 2) % 3] = 2 * pressure[(i + 1) % 3] * multiply - pressure[i % 3] + np.divide(2 * dct(force) * (1.0 - multiply), wi2)
	force = np.zeros_like(force)
	
	if i % 100 == 0:
		ax.plot(idct(pressure[(i + 2) % 3]))
		ax.set_ylim([-1, 1])
		plt.draw()
		plt.pause(0.1)
		plt.cla()
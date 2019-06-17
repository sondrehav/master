import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct
import math
from misc import gaussian, InteractivePlot

def dct2(block):
    return dct(dct(block.T, norm='ortho').T, norm='ortho')

def idct2(block):
    return idct(idct(block.T, norm='ortho').T, norm='ortho')

shape = (256, 256)

pressure = np.zeros((3, shape[0], shape[1]))
forces = gaussian((64, 92), 4, shape) * 1e3
freq = 22050
speedOfSound = 343.0
h = speedOfSound / (2 * freq)
k = 1.0 / (2 * freq)

x = np.power(np.arange(0, shape[1], 1), 2)
y = np.power(np.arange(0, shape[0], 1), 2)

xx, yy = np.meshgrid(x, y)
T = np.sqrt(xx + yy)

g = speedOfSound * math.pi * T
multipliers = np.cos(g*k)

plot = InteractivePlot()

for i in range(0, 10000):
	current = i % 3
	pressure[(current + 1) % 3] = 2 * pressure[current] * multipliers - pressure[(current - 1) % 3] + 2 * dct2(forces) * np.divide((1.0 - multipliers), np.power(T, 2), where=T!=0)
	if i % 10 == 0:
		plot.draw(idct2(pressure[(current + 1) % 3]))
	forces = np.zeros_like(forces)
plot.close()

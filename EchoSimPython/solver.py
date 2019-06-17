import numpy as np
from math import pi, cos
from transformtest import dct2

def solver(initial, prev = None, delta_t = 1.0 / 44100.0, c = 340.0):
	size = initial.shape
	fM = np.zeros(size)
	xs = np.arange(0, size[0], 1) + 1
	ys = np.arange(0, size[1], 1) + 1
	scaling_x = float(size[1]) / max(size[0], size[1])
	scaling_y = float(size[0]) / max(size[0], size[1])
	k = pi * np.sqrt(np.array([pow(xs[ix] * scaling_x,2) + pow(ys[iy] * scaling_y,2) for ix,iy in np.ndindex(fM.shape)]).reshape(fM.shape))
	w = c * k
	mult = np.cos(w*delta_t)
	fMTemp = fM

	F = dct2(initial)
	Fm = 2 * F * (1.0 - mult) / np.power(w, 2)

	if prev != None:
		fMTemp = dct2(prev)
	
	while True:
		new = 2 * fM * mult - fMTemp + Fm
		fMTemp = fM
		fM = new
		yield fM

		Fm = np.zeros(size)
		

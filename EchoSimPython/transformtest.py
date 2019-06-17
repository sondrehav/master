import numpy as np
from math import sqrt, log2
from scipy.fftpack import idct, dct
from scipy.ndimage import zoom

def dct2(block):
    return dct(dct(block.T, norm='ortho').T, norm='ortho')

def idct2(block):
    return idct(idct(block.T, norm='ortho').T, norm='ortho')

def generateNoise(size):
	array = np.zeros(size)
	octaves = int(log2(min(size[0], size[1])))
	for i in range(0, octaves):
		shape = (size[0] // pow(2, i), size[1] // pow(2, i))
		random = np.random.uniform(-1, 1, shape) * pow(2, i - octaves)
		random = zoom(random, pow(2, i), order=2)
		array += random
	return array
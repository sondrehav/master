import numpy as np
from scipy.fftpack import fft2
import matplotlib.pyplot as plt
from math import pow

a = 10
b = 20

f0 = lambda x, y: -pow(x-5,2)-pow(y-10,2) + 25

f = np.zeros((a, b))
for n in range(0, a):
	for m in range(0, b):
		f[n,m] = max(f0(n, m), 0)


plt.imshow(f)
plt.show()
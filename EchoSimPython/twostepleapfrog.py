import numpy as np
import matplotlib.pyplot as plt
from misc import InteractivePlot, gaussian, readwav, writewav24
import math
from scipy.linalg import solve
import sys

def createFirstOrderDifferentiationConvolution():
	xs = np.array([-1, 1])
	A = np.array([np.power(xs, n) for n in range(0, len(xs))])
	b = np.array([0, -1])
	return solve(A, b)

convolutionArray = createFirstOrderDifferentiationConvolution()

def derivativeX(vx):
	vx = np.pad(vx, ((1,1),(0,0)), mode='edge')
	return vx[1:,:] - vx[:-1,:]

def derivativeY(vy):
	return derivativeX(vy.T).T

def divergence(Vx, Vy):
	return derivativeX(Vx)[1:-1,:] + derivativeY(Vy)[:,1:-1]

def gradient(P):
	return (derivativeX(P), derivativeY(P))

shape = (64, 64)
numPML = 16

vx = np.zeros((shape[0]+1,shape[1]))
vy = np.zeros((shape[0],shape[1]+1))
p  = np.zeros(shape)
f  = gaussian((32, 32), 1, shape) * 1e2
#f  = np.zeros(shape)
#f[shape[0]//2,shape[1]//2] = 1e2
psi = np.zeros(shape)
phi = np.zeros(shape)
g = np.zeros(shape)

'''for x in range(32,48):
	for y in range(x,48):
		g[-x + 20, -y] = 0.9'''

soundVelocity = 343
timeStep = 1/44100
stepSize = soundVelocity * timeStep
pmlMax = 1500

vx = np.pad(vx, numPML, mode='constant')
vy = np.pad(vy, numPML, mode='constant')
p  = np.pad(p, numPML, mode='constant')
f  = np.pad(f, numPML, mode='linear_ramp')
g  = np.pad(g, numPML, mode='linear_ramp')

gvx, gvy = gradient(g)

psi = np.pad(psi, numPML, mode='constant')
phi = np.pad(phi, numPML, mode='constant')

px = np.array([1.0 - min(x/numPML, (shape[0] + 2 * numPML - x - 1) / numPML, 1) for x in range(0, p.shape[0])])
py = np.array([1.0 - min(y/numPML, (shape[1] + 2 * numPML - y - 1) / numPML, 1) for y in range(0, p.shape[1])])

px, py = np.meshgrid(py, px)

plt.imshow(px)
plt.show()

pml_vx = np.pad(px, ((1, 1), (0,0)), mode='edge')
pml_vx = (pml_vx[1:,:] + pml_vx[:-1,:]) / 2.0
pml_vy = np.pad(py, ((0,0),(1, 1)), mode='edge')
pml_vy = (pml_vy[:,1:] + pml_vy[:,:-1]) / 2.0

pml = (px + py) * pmlMax
gvx, gvy = gradient(g)

grad_x, grad_y = gradient(p + f)

vx = vx + 0.5 * timeStep * (soundVelocity * (1 - gvx) * grad_x / stepSize - pml_vx * vx)
vy = vy + 0.5 * timeStep * (soundVelocity * (1 - gvy) * grad_y / stepSize - pml_vy * vy)

psi = psi + 0.5 * timeStep * (soundVelocity * (1 - g) * px * derivativeY(vy)[:,1:-1] / stepSize - psi*py)
phi = phi + 0.5 * timeStep * py * ( soundVelocity * (1 - g) * derivativeX(vx)[1:-1,:] / stepSize + psi - p * px)

output = []
plot = InteractivePlot()

for i in range(0, 2500):

	_d = divergence(vx, vy)
	p = p + timeStep * (soundVelocity * (1 - g) * _d - pml * p + psi)
	
	#grad_x, grad_y = gradient(p + f)
	grad_x, grad_y = gradient(p + f)
	#grad_x, grad_y = gradient(p)
	f *= 0
	
	vx = vx + timeStep * (soundVelocity * (1 - gvx) * grad_x / stepSize - pml_vx * vx)
	vy = vy + timeStep * (soundVelocity * (1 - gvy) * grad_y / stepSize - pml_vy * vy)

	psi = psi + timeStep * (soundVelocity * (1 - g) * px * derivativeY(vy)[:,1:-1] / stepSize - psi*py)
	phi = phi + timeStep * py * ( soundVelocity * (1 - g) * derivativeX(vx)[1:-1,:] / stepSize + psi - p * px)

	if i % 100 == 0:
		plot.draw(p)

	output.append(p[numPML:-numPML,numPML:-numPML][48,32])
	

input = np.zeros_like(output)
input[0] = 1
output = np.array(output)

plot.close()

plt.plot(np.array(input), label='Impulse')
plt.plot(output*0.7, label='Response')
plt.legend()
plt.show()

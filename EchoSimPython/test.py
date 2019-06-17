import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from misc import InteractivePlot, gaussian

# divergence #
def staggeredDivergence(Vx, Vy):

	x = Vx[1:,:] - Vx[:-1,:]
	y = Vy[:,1:] - Vy[:,:-1]

	return x + y

def staggeredGradient(P):
	
	Px = np.pad(P, ((1, 1), (0, 0)), 'edge')
	Py = np.pad(P, ((0, 0), (1, 1)), 'edge')

	x = Px[1:,:] - Px[:-1,:]
	y = Py[:,1:] - Py[:,:-1]
	
	return (x, y)

def smooth(array):
	array = np.pad(array, 1, 'edge')
	array = 2 * array[1:-1,1:-1] + array[:-2,:-2] + array[2:,:-2] + array[2:,2:] + array[:-2,2:]
	return array / 6

def circle(centre, rad, shape):
	array = np.zeros(shape)
	for x in range(shape[1]):
		for y in range(shape[0]):
			if pow(y - centre[0], 2) + pow(x - centre[1], 2) <= rad * rad:
				array[y,x] = 1
	return array

def line(shape):
	array = np.zeros(shape)
	for x in range(shape[0]):
		for y in range(shape[1]):
			if -x + shape[1] < y:
				array[x, y] = 1
	return array

def geometryToStaggeredVelocity(geom):
	geometry_vx_for_y = np.pad(vx, ((0, 0), (1, 1)), 'edge')
	geometry_vx_for_y = (geometry_vx_for_y[:-1,:-1] + geometry_vx_for_y[1:,:-1] + geometry_vx_for_y[1:,1:] + geometry_vx_for_y[-1:,:1]) / 4.0
	geometry_vy_for_x = np.pad(vy, ((1, 1), (0, 0)), 'edge')
	geometry_vy_for_x = (geometry_vy_for_x[:-1,:-1] + geometry_vy_for_x[1:,:-1] + geometry_vy_for_x[1:,1:] + geometry_vy_for_x[-1:,:1]) / 4.0

shape = (256, 256)

#p = gaussian((shape[0]//2, shape[1]//4), 2, shape) * 5
p = np.zeros(shape)
p[128-4:128+4,128-4:128+4] = 1
vx = np.zeros((shape[0]+1, shape[1]))
vy = np.zeros((shape[0], shape[1]+1))

#geometry = smooth(circle((shape[0]//2, 3*shape[1]//4), int(np.sqrt(shape[0]*shape[1])) // 4, shape))
#geometry = line(shape)
#geometry = np.zeros(shape)
#geometry[20:-20,150:160] = 1

pressure = 1.2
soundVelocity = 340
timeStep = 7.81e-6
stepSize = 3.83e-4

pmlLayers = 0
pmlMax = 0#1e-1 * 0.5 / timeStep

#padding as according to pml
p = np.pad(p, pmlLayers, 'edge')
vx = np.pad(vx, pmlLayers, 'edge')
vy = np.pad(vy, pmlLayers, 'edge')

#geometry = np.pad(geometry, pmlLayers, 'edge')
#
#geometry_vx, geometry_vy = staggeredGradient(geometry)
#geometry_vx_for_y = np.pad(geometry_vx, ((0, 0), (1, 1)), 'edge')
#geometry_vx_for_y = (geometry_vx_for_y[:-1,:-1] + geometry_vx_for_y[1:,:-1] + geometry_vx_for_y[1:,1:] + geometry_vx_for_y[-1:,:1]) / 4.0
#geometry_vy_for_x = np.pad(geometry_vy, ((1, 1), (0, 0)), 'edge')
#geometry_vy_for_x = (geometry_vy_for_x[:-1,:-1] + geometry_vy_for_x[1:,:-1] + geometry_vy_for_x[1:,1:] + geometry_vy_for_x[-1:,:1]) / 4.0
#
#geom_dot_for_x = geometry_vx * geometry_vx  + geometry_vy_for_x * geometry_vy_for_x
#geom_dot_for_y = geometry_vx_for_y * geometry_vx_for_y  + geometry_vy * geometry_vy

def constructPML(shape, numLayers):
	pml = np.array([(x + 1) / numLayers for x in range(0, numLayers)])
	out = np.zeros(shape)
	for i in range(0, shape[0]):
		out[i,:numLayers] = 1 - pml + 1.0 / pmlLayers
		out[i,-numLayers:] = pml
	return out

#pmlX = constructPML(vx.shape, pmlLayers) * pmlMax
#pmlY = constructPML(vy.T.shape, pmlLayers).T * pmlMax
#pmlP = constructPML(p.shape, pmlLayers) * pmlMax
#pmlP += constructPML(p.T.shape, pmlLayers).T * pmlMax

plot = InteractivePlot();

grad_x, grad_y = staggeredGradient(p)
vx = vx + 0.5 * timeStep * ( - grad_x / pressure)
vy = vy + 0.5 * timeStep * ( - grad_y / pressure)

for i in range(0,1000):

	rhs = - pressure * pow(soundVelocity, 2) * staggeredDivergence(vx, vy) / stepSize
	#rhs -= pmlP * p
	p = p + timeStep * rhs

	grad_x, grad_y = staggeredGradient(p)
	vx = vx + timeStep * ( - grad_x / (pressure))
	vy = vy + timeStep * ( - grad_y / (pressure))

	#vyAverages = np.pad(vy, ((1, 1), (0, 0)), 'edge')
	#vyAverages = (vyAverages[:-1,:-1] + vyAverages[1:,:-1] + vyAverages[1:,1:] + vyAverages[-1:,:1]) / 4.0
	#
	#vxAverages = np.pad(vx, ((0, 0), (1, 1)), 'edge')
	#vxAverages = (vxAverages[:-1,:-1] + vxAverages[1:,:-1] + vxAverages[1:,1:] + vxAverages[-1:,:1]) / 4.0
	#
	#vx = vx - 0.5 * np.divide(geometry_vx * 2 * (vx * geometry_vx + vyAverages * geometry_vy_for_x), geom_dot_for_x, out=np.zeros_like(vx), where=geom_dot_for_x!=0)
	#vy = vy - 0.5 * np.divide(geometry_vy * 2 * (vxAverages * geometry_vx_for_y + vy * geometry_vy), geom_dot_for_y, out=np.zeros_like(vy), where=geom_dot_for_y!=0)

	#vy = vy - geometry_vy * 2 * (vxAverages * geometry_vx_for_y + vy * geometry_vy) / geom_dot_for_y

	if i % 10 == 0:
		
		#plot.draw(p[pmlLayers:-pmlLayers,pmlLayers:-pmlLayers])
		plot.draw(p)

plt.close()

#plt.imshow(p, cmap='magma', vmin = -1, vmax = 1)
#plt.show()

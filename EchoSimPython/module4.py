from pyassimp import load, release
import numpy as np
from collections import deque
import pickle
from math import ceil

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

vertices = []
faces = []
res = load('C:/Users/Sondre/Desktop/guitar2.3ds')

for c in res.rootnode.children:
	for mesh in c.meshes:
		faces.extend(mesh.faces + len(vertices))
		vertices.extend(mesh.vertices)
	print(str(c))

release(res)

def voxelize(vertices, faces, stepSize = np.array([.2,.2,.2])):

	minV = np.array([float("inf"), float("inf"), float("inf")])
	maxV = -minV
	for vertex in vertices:
		minV = np.minimum(vertex, minV)
		maxV = np.maximum(vertex, maxV)

	size = np.ceil((maxV - minV) / stepSize) + 2 * stepSize
	shape = (int(size[0]), int(size[1]), int(size[2]))
	voxels = np.zeros(shape)

	def toVoxelSpace(vertex):
		normalized = np.round((vertex - minV) / stepSize) + stepSize
		return normalized.astype(int)

	indexNum = 0
	for triangle in faces:
		num = 0
		vertexList = [toVoxelSpace(vertices[i]) for i in triangle]
		origin = vertexList[0]
		p1 = vertexList[1]
		p2 = vertexList[2]
		dist1 = np.linalg.norm(p1 - origin)
		dist2 = np.linalg.norm(p2 - origin)
		if dist1 == 0 or dist2 == 0:
			continue
		d1 = (p1 - origin) / dist1
		d2 = (p2 - origin) / dist2
		for j in range(0, max(ceil(dist1), ceil(dist2))):
			np1 = d1 * min(j, dist1)
			np2 = d2 * min(j, dist2)
			dir = np2 - np1
			dist = np.linalg.norm(dir)
			if dist == 0:
				continue
			dir /= dist
			for i in range(0, ceil(dist)):
				voxelPos = np.round(origin + np1 + i * dir).astype(int)
				if voxels[voxelPos[0], voxelPos[1], voxelPos[2]] == 0:
					voxels[voxelPos[0], voxelPos[1], voxelPos[2]] = 1
					num += 1
		indexNum += 1
		if indexNum % 50 == 0:
			print('progress: {} of {}'.format(indexNum, len(faces)))
	return voxels

def interiorVoxelize(voxels):
	inside = np.array(voxels)
	outside = np.zeros_like(voxels)
	
	for (x,y,z), value in np.ndenumerate(voxels):
		if value > 0.0: continue
		if outside[x, y, z] > 0: continue
		test = np.zeros_like(voxels)
		isOutside = False
		queue = deque([(x,y,z)])
		while queue:
			(x, y, z) = queue.popleft()
			if x <= 0 or y <= 0 or z <= 0 or x >= voxels.shape[0] - 1 or y >= voxels.shape[1] - 1 or z >= voxels.shape[2] - 1 or outside[x, y, z] > 0:
				outside = test + outside
				isOutside = True
				break
			elif inside[x,y,z] == 0:
				test[x,y,z] = 1
				queue.append([(x+1,y,z),(x-1,y,z),(x,y+1,z),(x,y-1,z),(x,y,z+1),(x,y,z-1)])
		if not isOutside:
			inside = inside + test
	return inside

def openVoxelsFromFile(file):
	return pickle.load(open(file, "rb" ))

def saveVoxelsToFile(voxels, file):
	pickle.dump(voxels, open(file, "wb"))

v = None
import os.path
if os.path.isfile("guitar.vox"):
	v = openVoxelsFromFile("guitar.vox")
else:
	v = voxelize(vertices, faces, stepSize = np.array([20,20,20]))
	v = interiorVoxelize(v)
	saveVoxelsToFile(v, "guitar.vox")


# and plot everything
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.voxels(v, facecolors='red', edgecolor='blue')

plt.show()

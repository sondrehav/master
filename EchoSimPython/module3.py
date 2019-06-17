from pyassimp import load, release
import numpy as np
from collections import deque
import pickle

THRESHOLD = 1e-5

'''

https://stackoverflow.com/questions/4293120/converting-a-3d-model-to-a-voxel-field

Here's a brief overview of how we solved this problem when converting regular quad/triangle models to a cubical array of 
indices of refraction/epsilon values for a photonics/EM simulation.

 1) Create a BSP tree of your scene. (Yes, really)

 2) Iterate at regular intervals over X,Y,Z across your model or solution space. (The interval in each axis should be equal 
    to the desired voxel dimensions.)

 3) At each point in your x/y/z loop, check the point against the BSP tree. If it's inside an entity, create a voxel at that 
    point, and set it's attributes (color, texture coordinates, etc) based on the source model (as referenced from your BSP 
    node). (Optimization Hint: If your inner-most loop is along the Y axis (vertical axis) and you're creating terrain or an 
    XZ-oriented surface, you can exit the Y loop whenever you create a voxel.)

 4) Save

 5) Profit!

'''

vertices = []
faces = []
res = load('C:/Users/Sondre/Desktop/guitar2.3ds')

for c in res.rootnode.children:
	for mesh in c.meshes:
		faces.extend(mesh.faces + len(vertices))
		vertices.extend(mesh.vertices)
	print(str(c))

release(res)




'''

https://en.wikipedia.org/wiki/Binary_space_partitioning

The canonical use of a BSP tree is for rendering polygons (that are double-sided, that is, without back-face culling) with the 
painter's algorithm. Each polygon is designated with a front side and a back side which could be chosen arbitrarily and only 
affects the structure of the tree but not the required result.[2] Such a tree is constructed from an unsorted list of all the 
polygons in a scene. The recursive algorithm for construction of a BSP tree from that list of polygons is:[2]

 1) Choose a polygon P from the list.
 
 2) Make a node N in the BSP tree, and add P to the list of polygons at that node.
 
 3) For each other polygon in the list:
        If that polygon is wholly in front of the plane containing P, move that polygon to the list of nodes in front of P.
        If that polygon is wholly behind the plane containing P, move that polygon to the list of nodes behind P.
        If that polygon is intersected by the plane containing P, split it into two polygons and move them to the respective 
		lists of polygons behind and in front of P.
        If that polygon lies in the plane containing P, add it to the list of polygons at node N.
 
 4) Apply this algorithm to the list of polygons in front of P.
 
 5) Apply this algorithm to the list of polygons behind P.

[2]: http://www.cs.unc.edu/~fuchs/publications/VisSurfaceGeneration80.pdf
'''

# http://geomalgorithms.com/a06-_intersect-2.html
def rayTriangleIntersection(planeNormal, planeOrigin, rayOrigin, rayDestination):
	
	d0 = np.dot(planeNormal, (rayOrigin - planeOrigin))
	dD = np.dot(planeNormal, (rayDestination - planeOrigin))
	assert((d0 > 0 and dD <= 0) or (d0 <= 0 and dD > 0))
	D = abs(d0) + abs(dD)
	alpha = d0 / D
	I = alpha * (rayDestination - rayOrigin) + rayOrigin

	dist = np.dot(planeNormal, (I - planeOrigin))

	return I

def checkValidPolygon(polygon, threshold = THRESHOLD):
	v = np.array([vertices[p] for p in polygon]).reshape((3,3))
	rolled = np.roll(v, 1, axis=0)
	t = np.array([ np.linalg.norm(v1) for v1 in (rolled - v)])
	v = np.all(t > threshold)
	return v





class BSPNode:
	def __init__(self):
		self.front = None
		self.back = None
		self.polygons = []


def buildBSP(triangle, rest, maxDepth = 100):
	node = BSPNode()

	node.polygons.append(triangle)

	queue = deque(rest)
	front = []
	back = []
		
	p = [vertices[v] for v in triangle]
	planeOrigin = p[0]
	normal = np.cross(p[1] - p[0], p[2] - p[0])
	dist = np.linalg.norm(normal)
	if dist == 0:
		return None

	normal /= np.linalg.norm(normal)
		
	while queue:
		polygon = queue.popleft()
		if not checkValidPolygon(polygon):
			continue
			
		frontFacingPoints = [i for i in polygon if np.dot(normal, (vertices[i] - p[0])) > THRESHOLD]
		backFacingPoints = [i for i in polygon if np.dot(normal, (vertices[i] - p[0])) < -THRESHOLD]
		onPlanePoints = [i for i in polygon if abs(np.dot(normal, (vertices[i] - p[0]))) <= THRESHOLD]

		assert(len(frontFacingPoints) + len(backFacingPoints) + len(onPlanePoints) == 3)
			
		if len(onPlanePoints) == 3:
			node.polygons.append(polygon)
		elif len(frontFacingPoints) == 3 or (len(frontFacingPoints) == 2 and len(onPlanePoints) == 1):
			front.append(polygon)
		elif len(backFacingPoints) == 3 or (len(backFacingPoints) == 2 and len(onPlanePoints) == 1):
			back.append(polygon)
		elif len(onPlanePoints) == 2:
			if len(backFacingPoints) > 0:
				assert(len(backFacingPoints) == 1)
				back.append(polygon)
			else:
				assert(len(frontFacingPoints) == 1)
				front.append(polygon)
		else:
			# Ok. We now know we have to split a polygon. :(
			if len(onPlanePoints) == 1:
				rayTarget1 = vertices[backFacingPoints[0]]
				rayTarget2 = vertices[frontFacingPoints[0]]
				newPoint = rayTriangleIntersection(normal, planeOrigin, rayTarget1, rayTarget2)
				vertices.extend([newPoint])
				newFrontFace = [frontFacingPoints[0], onPlanePoints[0], len(vertices) - 1]
				newBackFace = [backFacingPoints[0], onPlanePoints[0], len(vertices) - 1]
				front.append(newFrontFace)
				back.append(newBackFace)
			elif len(backFacingPoints) == 1:
				rayOrigin = vertices[backFacingPoints[0]]
				rayTarget1 = vertices[frontFacingPoints[0]]
				rayTarget2 = vertices[frontFacingPoints[1]]
				newPoint1 = rayTriangleIntersection(normal, planeOrigin, rayOrigin, rayTarget1)
				newPoint2 = rayTriangleIntersection(normal, planeOrigin, rayOrigin, rayTarget2)
				vertices.extend([newPoint1, newPoint2])
				newBackFace = [backFacingPoints[0], len(vertices) - 2, len(vertices) - 1]
				newFrontFace1 = [frontFacingPoints[0], len(vertices) - 2, len(vertices) - 1]
				newFrontFace2 = [frontFacingPoints[0], frontFacingPoints[1], len(vertices) - 1]
				front.append(newFrontFace1)
				front.append(newFrontFace2)
				back.append(newBackFace)
			elif len(frontFacingPoints) == 1:
				rayOrigin = vertices[frontFacingPoints[0]]
				rayTarget1 = vertices[backFacingPoints[0]]
				rayTarget2 = vertices[backFacingPoints[1]]
				newPoint1 = rayTriangleIntersection(normal, planeOrigin, rayOrigin, rayTarget1)
				newPoint2 = rayTriangleIntersection(normal, planeOrigin, rayOrigin, rayTarget2)
				vertices.extend([newPoint1, newPoint2])
				newFrontFace = [frontFacingPoints[0], len(vertices) - 2, len(vertices) - 1]
				newBackFace1 = [backFacingPoints[0], len(vertices) - 2, len(vertices) - 1]
				newBackFace2 = [backFacingPoints[0], backFacingPoints[1], len(vertices) - 1]
				back.append(newBackFace1)
				back.append(newBackFace2)
				front.append(newFrontFace)
			else:
				assert(False)
		
	if len(front) > 0 and maxDepth > 0:
		while node.front == None and len(front) > 0:
			node.front = buildBSP(front[0], front[1:], maxDepth - 1)
			# bad polygon, skip this...
			front = front[1:]
	if len(back) > 0 and maxDepth > 0:
		while node.back == None and len(back) > 0:
			node.back = buildBSP(back[0], back[1:], maxDepth - 1)
			# bad polygon, skip this...
			back = back[1:]
	return node

root = buildBSP(faces[0], faces[1:])
print('done!')







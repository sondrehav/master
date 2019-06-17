import numpy as np
from collada import Collada

def load(file):
	mesh = Collada(file)
	index_count = 0
	all_indices = []
	all_vertices = []
	for geometry in mesh.geometries:
		for primitive in geometry.primitives:
			indices = []
			for tri in primitive.indices:
				indices.extend(tri[:,0] + index_count)
			index_count += len(primitive.vertex)
			vertices = primitive.vertex
			all_vertices.extend(vertices)
			all_indices.extend(indices)
	return np.array(all_vertices).reshape(len(all_vertices), 3), np.array(all_indices)

def project(vertices, indices, direction = np.array([1, 1, 0])):
	n = direction / np.linalg.norm(direction)
	proj = np.empty_like(vertices)
	for index, vertex in enumerate(vertices):
		p = vertex - np.dot(vertex, n) * n
		proj[index] = p
	return proj


vertices, indices = load('files/box.dae')
print(project(vertices, indices))
# A Python3 program for
# Prim's Minimum Spanning Tree (MST) algorithm.
# The program is for adjacency matrix
# representation of the graph
# Library for INT_MAX
import sys
import numpy as np
class Graph():
	def __init__(self, vertices):
		self.V = vertices
		self.graph = np.zeros((vertices,vertices),dtype=np.float32)
	def printMST(self, parent):
	#  A utility function to print
	#  the constructed MST stored in parent[]
	    # print("Edge \tWeight")
		for i in range(1, self.V):
			print(parent[i], "-", i, "\t", self.graph[i][parent[i]])
	def minKey(self, key, mstSet):
	#  A utility function to find the vertex with
	#  minimum distance value, from the set of vertices
	#  not yet included in shortest path tree
		# Initialize min value
		min = sys.maxsize
		for v in range(self.V):
			if key[v] < min and mstSet[v] == False:
				min = key[v]
				min_index = v
		return min_index
	def primMST(self):
	# Function to construct and print MST for a graph
	# represented using adjacency matrix representation
		# Key values used to pick minimum weight edge in cut
		key = [sys.maxsize] * self.V
		parent = [None] * self.V # Array to store constructed MST
		# Make key 0 so that this vertex is picked as first vertex
		key[0] = 0
		mstSet = [False] * self.V
		parent[0] = -1 # First node is always the root of
		for cout in range(self.V):
			# Pick the minimum distance vertex from
			# the set of vertices not yet processed.
			# u is always equal to src in first iteration
			u = self.minKey(key, mstSet)
			# Put the minimum distance vertex in
			# the shortest path tree
			mstSet[u] = True
			# Update dist value of the adjacent vertices
			# of the picked vertex only if the current
			# distance is greater than new distance and
			# the vertex in not in the shortest path tree
			for v in range(self.V):
				# graph[u][v] is non zero only for adjacent vertices of m
				# mstSet[v] is false for vertices not yet included in MST
				# Update the key only if graph[u][v] is smaller than key[v]
				if self.graph[u][v] > 0 and mstSet[v] == False \
				and key[v] > self.graph[u][v]:
					key[v] = self.graph[u][v]
					parent[v] = u
		self.printMST(parent)

# Driver's code
if __name__ == '__main__':
	g = Graph(5)
	g.graph = np.array([[0., 0.02713484, 0.01794069, 0.09402313, 0.15748233],
 [0.02713484, 0., 0.02065755, 0.09902701,0.18016563],
 [0.01794069, 0.02065755, 0., 0.10312883, 0.1882463],
 [0.09402313, 0.09902701, 0.10312883, 0., 0.1032144],
 [0.15748233, 0.18016563, 0.1882463, 0.1032144, 0.]])
	g.primMST()

# Contributed by Divyanshu Mehta

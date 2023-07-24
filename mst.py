# A Python3 program for
# Prim's Minimum Spanning Tree (MST) algorithm.
# The program is for adjacency matrix
# representation of the graph
# Library for INT_MAX
import sys
import numpy as np
import glob
import cv2 as cv
import testset_build

def getMatchNum(matches,ratio):

    matchesMask=[[0,0] for i in range(len(matches))]
    matchNum=0
    for i,(m,n) in enumerate(matches):
        if m.distance<ratio*n.distance:
            matchesMask[i]=[1,0]
            matchNum+=1
    return (matchNum,matchesMask)

def CalcSimilaritySIFT(imgList):
	sift = cv.SIFT_create()
	FLANN_INDEX_KDTREE=0
	indexParams=dict(algorithm=FLANN_INDEX_KDTREE,trees=5)
	searchParams=dict(checks=50)
	flann=cv.FlannBasedMatcher(indexParams,searchParams)

	length = len(imgList[0])
	Similarity = np.zeros((length,length),dtype=np.float32)

	for i in range(0,length):
		sampleImage = cv.imread(imgList[0][i],0)
		kp1, des1 = sift.detectAndCompute(sampleImage, None)
		for j in range(0,length):
			queryImage = cv.imread(imgList[0][j],0)
			kp2, des2 = sift.detectAndCompute(queryImage, None)
			matches=flann.knnMatch(des1,des2,k=2)
			(matchNum,matchesMask)=getMatchNum(matches,0.9)
			matchRatio=matchNum/len(matches)
			Similarity[i][j] = matchRatio
			print(Similarity[i][j])
	for i in range(0,length):
		for j in range(0,length):
			if i == j:
				pass
			else:
				Similarity[i][j] = max([Similarity[i][j],Similarity[j][i]])
	for i in range(0,length):
		for j in range(0,length):
			Similarity[i][j] = 1 - Similarity[i][j]
	return Similarity

def TwoImgSimilarity(img1,img2):
    IMG1_SIZE = img1.shape[0:2] #assume BGR colored image
    IMG2_SIZE = img2.shape[0:2] #assume BGR colored image
    NEW_SIZE = tuple([max(size1, size2) for size1, size2 in zip(IMG1_SIZE,IMG2_SIZE)]) #new size is the maximum of 2 dimensions 
    img1re = cv.resize(img1, NEW_SIZE, interpolation=cv.INTER_LANCZOS4)
    img2re = cv.resize(img2, NEW_SIZE, interpolation=cv.INTER_LANCZOS4)
    grey2 = cv.cvtColor(img2re, cv.COLOR_BGR2GRAY)
    grey1 = cv.cvtColor(img1re, cv.COLOR_BGR2GRAY)
    hist1 = cv.calcHist(grey1, [0], None, [256], [0.0, 255.0])
    hist2 = cv.calcHist(grey2, [0], None, [256], [0.0, 255.0])
    Similarity = cv.compareHist(hist1, hist2, cv.HISTCMP_BHATTACHARYYA)
    return Similarity

def CalcSimilarityHist(imgList):
    length = len(imgList)
    if len(imgList)==0:
        print("in CalcSimilarityHist img is empty")
        sys.exit()
        return -1
    elif len(imgList)== 1:
        print("in CalcSimilarityHist img has only one image")
        sys.exit()
        return -1
    Similarity = np.zeros((length,length),dtype=np.float32)
    for i in range(0,length):
        for j in range(0,length):
            Similarity[i][j] = TwoImgSimilarity(imgList[i],imgList[j])
            print(Similarity[i][j])
    return Similarity

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
		parent[0] = -1 # First node is always the root
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
		return parent

# Driver's code
if __name__ == '__main__':
	methodcode = input('Please input the MST method code \n 1 for SIFT\n 2 for Histgram\n')
	if ord(methodcode) < 49 or ord(methodcode) >50:
		raise ValueError
	setcode = input('Please input the testset code \n 1 for cropped_img\n 2 for rotated_img\n 3 for zoomed_img \n 4 for set1 \n 5 for set2\n')
	if ord(setcode) < 49 or ord(setcode) >53:
		raise ValueError
	testset = {'1':'cropped_img', '2':'rotated_img', '3':'zoomed_img', '4':'testset1_img', '5':'testset2_img'}
	imgList = [glob.glob("./data/"+testset[setcode]+"[0-9].jpeg")]
	setSize = len(imgList)
	imgList = []
	for i in range(setSize):
		originalImg = cv.imread("./data/"+testset[setcode]+str(i)+".jpeg")
		imgList.append(originalImg)
	g = Graph(len(imgList))
	if methodcode == '1': 
		g.graph = CalcSimilaritySIFT(imgList)
	elif methodcode == '2':
		g.graph = CalcSimilarityHist(imgList)
	print(g.graph)
	parent = g.primMST()

# Contributed by Divyanshu Mehta

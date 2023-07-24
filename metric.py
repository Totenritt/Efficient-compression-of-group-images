from math import log10, sqrt
import cv2 as cv
import numpy as np
import glob
import matplotlib.pyplot as plt

def PSNR(original, compressed):
	mse = np.mean((original - compressed) ** 2)
	if(mse == 0): # MSE is zero means no noise is present in the signal .
				# Therefore PSNR have no importance.
		return 100
	max_pixel = 255.0
	psnr = 20 * log10(max_pixel / sqrt(mse))
	return psnr

def psnrFormula(mse):
    return 20 * log10(255.0 / sqrt(mse))

def groupPsnr(originalImgList, compressedImgList):
	'''
	calculate psnr for a group of images
	'''
	overallMse = [[] for i in range(20)] # storage for all mse data
	# calculate mse 
	for (i, originalImg) in enumerate(originalImgList[1:]): # again assume the first element is the mother image
		for j,compressedImg in enumerate(compressedImgList[i]):
			mse = np.mean((originalImg - compressedImg) ** 2)
			overallMse[j].append(mse)
	# calculate psnr 
	mseArr = np.array(overallMse)
	psnrVec = np.vectorize(psnrFormula)
	psnr = psnrVec(np.mean(mseArr, axis= 1))
	return psnr 

def extractImg(setName):
	'''
	extract both original and compressed images, store original image in originalImgList, and store correspond compressed image in 
	compressedImgList, each list within compressedImgList contains 20 final images'''
	originalImgList = [cv.imread(file) for file in glob.glob("./data/"+setName+'[0-9]'+".jpeg")]
	setSize = len(originalImgList)
	originalImgList = []
	for i in range(setSize):
		originalImg = cv.imread("./data/"+ setName + str(i+1) +".jpeg")
		originalImgList.append(originalImg)	
	compressedImgList = [[] for i in range(len(originalImgList)-1)] # since 1st img is assumed mother, we don't have compressed img for mother, this leaves setSize - 1 list within compressedImgList
	for i in range(len(originalImgList) - 1): # assuming img1 is the mother image
		for j in range(0,20):
			compressedImg = cv.imread("./data/residual/rotated_img"+str(i+2)+"finalImg"+str(j+1)+".ppm")
			compressedImgList[i].append(compressedImg)
	return originalImgList,compressedImgList

def test():
	originalImgList,compressedImgList = extractImg('rotated_img')
	value = groupPsnr(originalImgList, compressedImgList)
	print(value)
	return None

def main():
	test()

if __name__ == "__main__":
	main()

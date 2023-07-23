from math import log10, sqrt
import cv2 as cv
import numpy as np
import glob
def PSNR(original, compressed):
	mse = np.mean((original - compressed) ** 2)
	if(mse == 0): # MSE is zero means no noise is present in the signal .
				# Therefore PSNR have no importance.
		return 100
	max_pixel = 255.0
	psnr = 20 * log10(max_pixel / sqrt(mse))
	return psnr

def groupPsnr(setName, mst):
	'''
	calculate psnr for a group of images
	'''
	originalImgList = [cv.imread(file) for file in glob.glob("./data/"+setName+'[0-9]'+".jpeg")] 
	compressedImgList = []
	# get compressed image list by search given directory
	for i in range(originalImgList):
		compressedImgList.append([cv.imread(file) for file in glob.glob("./data/residual/"+setName+str(i+1)+"finalImg{,[1-9][0-9]}.jpeg", glob.GLOB_BRACE)])
	mseList =[[]*len(originalImgList)]
	# calculate mse for each original image and 20 correspond compressed img, store result in seperate lists
	for i in range(len(originalImgList)):
		for j in range(len(compressedImgList[0])):
			mseList[i].append([np.mean((originalImgList[i] - compressedImgList[i][j]))])
	# for each seperate list calculate the mean
	mseMean = np.mean(mseList, axis=1)
	for mean in mseMean:
		if(mseMean == 0): # MSE is zero means no noise is present in the signal .
					# Therefore PSNR have no importance.
			return 100
		max_pixel = 255.0
		psnr = 20 * log10(max_pixel / sqrt(mseMean))
	return psnr
	
def groupPsnr(originalImg,compressedImgList)
	mseList = []
	# calculate mse for each original image and 20 correspond compressed img, store result in seperate lists
	for i in range(len(compressedImgList)):
		mseList.append([np.mean((originalImg - compressedImgList[i]) ** 2)])
	mseMean = np.mean(mseList)
	if(mseMean == 0): # MSE is zero means no noise is present in the signal .
				# Therefore PSNR have no importance.
		return 100
	max_pixel = 255.0
	psnr = 20 * log10(max_pixel / sqrt(mseMean))
	return psnr

def main():
	original = cv.imread("./data/cropped_img2.jpeg")
	compressed = cv.imread("./data/residual/cropped_img2finalImg20BryBlur.ppm")
	value = PSNR(original, compressed)
	print(f"PSNR value is {value} dB")
def psnrForGroup	
if __name__ == "__main__":
	main()

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
	overallMse = [[] for i in range(21)] # storage for all mse data
	# calculate mse 
	for (i, originalImg) in enumerate(originalImgList[1:]): # again assume the first element is the mother image
		for j,compressedImg in enumerate(compressedImgList[i]):
			mse = np.mean((originalImg - compressedImg) ** 2)
			overallMse[j].append(mse)
	# calculate psnr 
	mseArr = np.array(overallMse)
	psnrVec = np.vectorize(psnrFormula) # psnrVec apply psnr formula to every elements in the array
	psnr = psnrVec(np.mean(mseArr, axis= 1)) # 
	return psnr 

def CMPRgroupPsnr(originalImgList, comparisonImgList):
	'''
	calculate psnr for a group of images
	'''
	overallMse = [[] for i in range(20)] # storage for all mse data
	# calculate mse 
	for (i, originalImg) in enumerate(originalImgList[1:]): # again assume the first element is the mother image
		for j, comparisonImg in enumerate(comparisonImgList[i]):
			mse = np.mean((originalImg - comparisonImg) ** 2)
			overallMse[j].append(mse)
	# calculate psnr 
	mseArr = np.array(overallMse)
	psnrVec = np.vectorize(psnrFormula) # psnrVec apply psnr formula to every elements in the array
	CMPRpsnr = psnrVec(np.mean(mseArr, axis= 1)) # 
	return CMPRpsnr 

def extractImg(setName):
	'''
	extract both original and compressed images, store original image in originalImgList, and store correspond compressed image in 
	compressedImgList, each list within compressedImgList contains 20 final images'''
	originalImgList = [cv.imread(file) for file in glob.glob("./data/"+setName+'[0-9]'+".jpeg")]
	setSize = len(originalImgList)
	comparisonImgList = [[] for j in range(len(originalImgList)-1)]
	originalImgList = []
	predictedImgList =[]
	for i in range(setSize):
		originalImg = cv.imread("./data/"+ setName + str(i+1) +".jpeg")
		if i == 0:
			pass
		else:
			predictedImg = cv.imread('./data/'+setName+str(i+1)+'_predicted.jpeg')
			predictedImgList.append(predictedImg)
		originalImgList.append(originalImg)	
	compressedImgList = [[predictedImgList[i]] for i in range(len(originalImgList)-1)] # since 1st img is assumed mother, we don't have compressed img for mother, this leaves setSize - 1 list within compressedImgList
	for i in range(len(originalImgList) - 1): # assuming img1 is the mother image
		for j in range(0,20):
			compressedImg = cv.imread("./data/residual/"+setName+str(i+2)+"finalImg"+str(j+1)+".ppm")
			comparisonImg = cv.imread("./data/comparison/"+setName+str(i+2)+"decoded"+str(j+1)+".ppm")
			compressedImgList[i].append(compressedImg)
			comparisonImgList[i].append(comparisonImg)
	return originalImgList,compressedImgList, comparisonImgList


def test():
	originalImgList,compressedImgList = extractImg('rotated_img')
	value = groupPsnr(originalImgList, compressedImgList)
	print(value)
	# originalImg = cv.imread("./data/rotated_img2.ppm")
	# predictedImg = cv.imread("./data/residual/rotated_img2finalImg10.ppm")
	# print(PSNR(originalImg, predictedImg))
	return None

def groupPsnrPlot(setName, meanBppList, CMPRBppList):
	originalImgList, compressedImgList, comparisonImgList = extractImg(setName)
	meanPsnrList = groupPsnr(originalImgList, compressedImgList)
	CMPRPsnrList = CMPRgroupPsnr(originalImgList, comparisonImgList)

	plt.figure()

	plt.plot(meanBppList, meanPsnrList, color='r', label='GMIS')
	plt.plot(CMPRBppList, CMPRPsnrList, color='g', label='JPEG2000')
	plt.legend()
	plt.xlabel('Bpp')
	plt.ylabel('PSNR/dB')
	plt.title('Comparison of PSNR vs Bpp graph of GMIS and JPEG2000')
	plt.savefig('./data/comparison/RotatedImgGraph.png')
	plt.show()

def main():
	test()

def test2():
	orginalImg = cv.imread("./data/rotated_img2.ppm")
	decodedImg = cv.imread("./data/rotated_img1_decoded.ppm")
	pridictedImg = cv.imread("./data/comparison/rotated_img2_decode.ppm")
	print(PSNR(orginalImg,pridictedImg))


if __name__ == "__main__":
	# main()
	test()

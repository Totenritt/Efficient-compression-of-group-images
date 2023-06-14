import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import sys
import imutils
import glob
import testset_build
import mst
import string 
import re
import subprocess
def findLayerThresholds(inputByte):
    inputStr = str(inputByte)
    beginIndex = inputStr.find('Layer thresholds:') + 27
    endIndex = inputStr.find('Processed') - 2 
    threshold = inputStr[beginIndex:endIndex]
    threshold = re.sub(r'[^0-9,]', '', threshold)
    return threshold

def encoder():
    '''
    encode the predictive image of child image with respect to the parent image
    '''
    #read the image
    parentName = input('input the name of parent image: ')
    childName = input('input the name of child image: ')
    parentImg = cv.imread('./data/' + parentName +'.jpeg')
    childImg = cv.imread('./data/' +childName +'.jpeg')
    #generate predicted image
    predictImg = testset_build.generatePredictImg(parentImg,childImg, False, filename=childName +'_predicted.jpeg')
    #generate residual image
    childImg = childImg.astype(int) #cast to int type to avoid negative values clipping 
    predictImg = predictImg.astype(int)
    diffMatrix = np.subtract(childImg,predictImg)
    # convert image to 16bits, add 255 offset to residual image and same with name residual.ppm 
    residualImg = testset_build.codeResidual(diffMatrix)
    # command used to get threshold parameter
    thresholdCmd = "kdu_compress -i ./data/residual/offsetImg.ppm -o ./data/residual/offsetImg_output.jp2 Clayers=20"
    output = subprocess.check_output(thresholdCmd, shell=True)
    threshold = findLayerThresholds(output)
    # kakadu command for compress file 
    compressCmd = "kdu_compress -i ./data/residual/offsetImg.ppm -o ./data/residual/offsetImgoutput.jp2 -slope " + threshold
    subprocess.run(compressCmd, shell=True)
    # subprocess.run('kdu_expand -i ./data/residual/offsetImg_output.jp2 -o ./data/residual/decodedImg_20.ppm -layers 20')
    return 0

def decoder():
    expandCmd = 'kdu_expand -i /data/residual/offsetImg_output.jp2 -o ./data/residual/decodedImg_20.ppm -layers 20' 
    subprocess.run('expandCmd', shell=True)
if __name__ == '__main__':
    encoder()
# '''
# the image u need 
# '''
# img1 = cv.imread('./data/cropped_img2.jpeg')
# img2 = cv.imread('./data/cropped_img2_predicted.jpeg')
# '''
# create the ppm with a offset255, named it offsetImg.ppm
# '''
# img1int = img1.astype(int)
# img2int = img2.astype(int)
# diffimg = np.subtract(img1int,img2int)
# diffimg255 = diffimg.astype(np.uint16) + 255
# cv.imwrite('./data/residual/offsetImg.ppm', diffimg255)

# '''
# go cmd, input 'kdu_compress -i offsetImg.ppm -o offsetImg_output.jp2 Clayers=20'
# then it output '44670, 44370, 44070, 43770, 43470, 43170, 42870, 42570, 42270, 41970,
#         41670, 41370, 41070, 40770, 40470, 40170, 39870, 39570, 38723, 0'
# copy these number and delete the space within numbers, like '44670,44370,44070,43770,43470,
# 43170,42870,42570,42270,41970,41670,41370,41070,40770,40470,40170,39870,39570,38723,0'
# Then, input 'kdu_compress -i offsetImg.ppm -o offsetImgoutput.jp2 -slope 44670,...,0'
# Then input 'kdu_expand -i offsetImg_output.jp2 -o decodedImg_20.ppm -layers 20' layers 1-20. layers 1 is the most blurred
# So u get a decoded ppm image
# '''
# '''
# img3 is the decodedImg_20 of layers 20, which is 0 in 'slope 44670,...,0'
# '''
# img3 = cv.imread('./data/decodedImg_20.ppm',-1)
# img3_255 = img3.astype(np.uint8)-255
# decoded_targetImg = img2 + img3_255
# cv.imwrite('./data/decoded_targetImg.jpeg', decoded_targetImg)



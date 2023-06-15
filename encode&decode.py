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
    '''
    extract the threshold info from kakadu output log
    '''
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
    predictImg = testset_build.generatePredictImg(parentImg,childImg, False, filename=childName)
    #generate residual image
    childImg = childImg.astype(int) #cast to int type to avoid negative values clipping 
    predictImg = predictImg.astype(int)
    diffMatrix = np.subtract(childImg,predictImg) + 255
    diffMatrix = diffMatrix.astype(np.uint16)
    cv.imwrite('./data/residual/offsetImg.ppm', diffMatrix)
    # convert image to 16bits, add 255 offset to residual image and same with name residual.ppm 
    # residualImg = testset_build.codeResidual(diffMatrix)
    # command used to get threshold parameter
    thresholdCmd = "kdu_compress -i ./data/residual/offsetImg.ppm -o ./data/residual/offsetImg_output.jp2 Clayers=20"
    output = subprocess.check_output(thresholdCmd, shell=True)
    threshold = findLayerThresholds(output)
    # kakadu command for compress file 
    compressCmd = "kdu_compress -i ./data/residual/offsetImg.ppm -o ./data/residual/offsetImg_output.jp2 -slope " + threshold
    subprocess.run(compressCmd, shell=True)
    return 0

def decoder():
    predictedName = input('input the name of predicted image: ')
    predictedImg = cv.imread('./data/' + predictedName +'_predicted.jpeg')
    for i in range(1,21):
        # kakadu command for expand file
        expandCmd = 'kdu_expand -i ./data/residual/offsetImg_output.jp2 -o ./data/residual/decodedImg_'+ str(i) +'.ppm -layers ' + str(i)
        subprocess.run(expandCmd, shell=True)
        decodedImg = cv.imread('./data/residual/decodedImg_'+str(i)+'.ppm',-1) # -1 means read the image with original data format, in this case uint16
        decodedImg= decodedImg.astype(int) - 255
        predictedImg = predictedImg.astype(int)
        finalImg = decodedImg + predictedImg
        # finalImg = np.clip(finalImg,0,255)
        finalImg[finalImg >255] = 255
        finalImg[finalImg < 0] = 0
        finalImg = finalImg.astype(np.uint8)
        cv.imwrite('./data/residual/finalImg'+str(i)+'.ppm', finalImg)
    return 0 

if __name__ == '__main__':
    encoder()
    decoder()




import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import sys
import imutils
import glob
import testset_build
import string 
import re
import subprocess
import mst
def downsampling():
    '''
    down sample the phone.jpeg'''
    phone1 = cv.imread('./data/phone1.jpeg')
    phone2 = cv.imread('./data/phone2.jpeg')
    newSize = phone1.shape[1]//2,phone1.shape[0]//2
    zoomed_phone1 = cv.resize(phone1, newSize, interpolation=cv.INTER_LANCZOS4)
    zoomed_phone2 = cv.resize(phone2, newSize, interpolation=cv.INTER_LANCZOS4)
    cv.imwrite('./data/phone1_resize.jpeg',zoomed_phone1)
    cv.imwrite('./data/phone2_resize.jpeg',zoomed_phone2)
    return None

def findLayerThresholds(inputByte):
    '''
    extract the threshold info from kakadu output log
    '''
    inputStr = str(inputByte) # convert the input Byte to str
    beginIndex = inputStr.find('Layer thresholds:') + 27 #find threshold begin index
    endIndex = inputStr.find('Processed') - 2 
    threshold = inputStr[beginIndex:endIndex]
    threshold = re.sub(r'[^0-9,]', '', threshold)
    return threshold

def predict(parentName, childName):
    '''
    dummy function to call generatePredictImg, make no sense but it solves blue boundary issue
    '''
    parentImg = cv.imread('./data/'+ parentName +'.jpeg')
    childImg = cv.imread('./data/'+ childName +'.jpeg')
    #generate predicted image
    predictImg = testset_build.generatePredictImg(parentImg,childImg, False, filename=childName)
    return 0 

def encoder(parentName, childName):
    '''
    encode the predictive image of child image with respect to the parent image
    '''
    #read the image
    childImg = cv.imread('./data/' +childName +'.jpeg')
    #imread predicted image
    predictImg = cv.imread('./data/'+childName+'_predicted.jpeg')
    #imwrite predicted image.ppm and cpmpress it
    cv.imwrite('./data/'+childName+'.ppm',predictImg)
    compressCmd = "kdu_compress -i ./data/"+childName+".ppm -o ./data/"+childName+".jp2"
    subprocess.run(compressCmd, shell=True)
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
    return threshold

def decoder(threshold, childName):
    decodedCmd = "kdu_expand -i ./data/"+childName+".jp2 -o ./data/"+childName+"_decoded"+".ppm"
    subprocess.run(decodedCmd, shell=True)
    predictedImg = cv.imread("./data/"+childName+"_decoded"+".ppm")
    # kakadu command for compress file 
    compressCmd = "kdu_compress -i ./data/residual/offsetImg.ppm -o ./data/residual/offsetImg_output.jp2 -slope " + threshold
    subprocess.run(compressCmd, shell=True)
    for i in range(1,21):
        # kakadu command for expand file
        expandCmd = 'kdu_expand -i ./data/residual/offsetImg_output.jp2 -o ./data/residual/decodedImg_'+ str(i) +'.ppm -layers ' + str(i)
        subprocess.run(expandCmd, shell=True)
        decodedImg = cv.imread('./data/residual/decodedImg_'+str(i)+'.ppm',-1) # -1 means read the image with original data format, in this case uint16
        decodedImg= decodedImg.astype(int) - 255
        predictedImg = predictedImg.astype(int)
        finalImg = decodedImg + predictedImg
        finalImg = np.clip(finalImg,0,255)
        finalImg = finalImg.astype(np.uint8)
        cv.imwrite('./data/residual/'+childName + 'finalImg'+str(i)+'.ppm', finalImg)
    return 0 

def main():
    setcode = input('Please input the testset code \n 1 for cropped_img\n 2 for rotated_img\n 3 for zoomed_img \n 4 for set1 \n 5 for set2\n')
    if ord(setcode) < 49 or ord(setcode) >54:
        raise ValueError
    testset = {'1':'cropped_img', '2':'rotated_img', '3':'zoomed_img', '4':'testset1_', '5':'testset2_', '6':'phone'}
    setName = testset[setcode]
    imgList = [cv.imread(file) for file in glob.glob("./data/"+ setName +'[0-9]'+".jpeg")]
    g = mst.Graph(len(imgList))
    g.graph = mst.CalcSimilarityHist(imgList)
    mstList = g.primMST()
    for i in range(1, len(mstList)):
        parentName = setName + str(mstList[i]+1)
        childName = setName + str(i+1)
        predict(parentName, childName)
        threshold = encoder(parentName, childName)
        decoder(threshold, childName)

if __name__ == '__main__':
    main()
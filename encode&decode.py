import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import glob
import testset_build
import re
import subprocess
import mst
import metric


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
    # find layer threshold
    inputStr = str(inputByte) # convert the input Byte to str
    beginIndex = inputStr.find('Layer thresholds:') + 27 #find threshold begin index
    endIndex = inputStr.find('Processed') - 2 
    threshold = inputStr[beginIndex:endIndex]
    threshold = re.sub(r'[^0-9,]', '', threshold)
    return threshold

def findBpp(inputByte):
    # find bpp
    inputStr = str(inputByte)
    beginIndex = inputStr.find('= ') + 2 #find threshold begin index for encode set to +2
    endIndex = inputStr.find('bits/pel')# find threshold end index for encode set to -2
    bppStr = inputStr[beginIndex:endIndex]
    bppStr = re.sub(r'[^0-9,.]', '', bppStr)
    # bppList = [float(i) for i in bppStr.split(',')]
    return float(bppStr)

def predict(parentName, childName):
    '''
    dummy function to call generatePredictImg, make no sense but it solves blue boundary issue
    '''
    parentImg = cv.imread('./data/'+ parentName +'.jpeg')
    childImg = cv.imread('./data/'+ childName +'.jpeg')
    #generate predicted image
    predictImg,Homography = testset_build.generatePredictImg(parentImg,childImg, False, filename=childName)
    return Homography 

def encoder(parentName, childName):
    '''
    encode the predictive image of child image with respect to the parent image
    '''
    #read the image
    parentImg = cv.imread('./data/' +parentName +'.jpeg')
    childImg = cv.imread('./data/' +childName +'.jpeg')
    childWidth,childHeight,childChannel = childImg.shape
    #imread predicted image
    predictImg = cv.imread('./data/'+childName+'_predicted.jpeg')
    #imwrite parent image.ppm and compress it
    cv.imwrite('./data/'+parentName+'.ppm',parentImg)
    cv.imwrite('./data/'+childName+'.ppm',childImg)
    compressCmd = "kdu_compress -i ./data/"+parentName+".ppm -o ./data/"+parentName+".jp2"
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
    thresholdCmd = "kdu_compress -i ./data/residual/offsetImg.ppm -o ./data/residual/offsetImg_output.jp2 Qstep=0.000001 Clayers=20"
    # kdu_compress -i image.pgm -o out.j2c -rate 1.0,0.5,0.25
    output = subprocess.check_output(thresholdCmd, shell=True)
    threshold = findLayerThresholds(output)
    return threshold,childWidth,childHeight

def decoder(threshold,parentName, childName, Homography, Height, Width):
    # kakadu command for decode
    decodedCmd = "kdu_expand -i ./data/"+parentName+".jp2 -o ./data/"+parentName+"_decoded"+".ppm"
    output = subprocess.check_output(decodedCmd, shell=True)
    threshold = findLayerThresholds(output)
    parentImg = cv.imread("./data/"+parentName+"_decoded"+".ppm")
    predictedImg = cv.warpPerspective(parentImg, Homography, (Height, Width), flags = cv.INTER_LANCZOS4)
    # # kakadu command for compress file 
    # compressCmd = "kdu_compress -i ./data/residual/offsetImg.ppm -o ./data/residual/offsetImg_output.jp2 -slope " + threshold
    # subprocess.run(compressCmd, shell=True)
    bppList = []
    for i in range(1,21):
        # kakadu command for expand file
        expandCmd = 'kdu_expand -i ./data/residual/offsetImg_output.jp2 -o ./data/residual/decodedImg_'+ str(i) +'.ppm -layers ' + str(i)
        output = subprocess.check_output(expandCmd, shell=True)
        bppList.append(findBpp(output))
        decodedImg = cv.imread('./data/residual/decodedImg_'+str(i)+'.ppm',-1) # -1 means read the image with original data format, in this case uint16
        decodedImg= decodedImg.astype(int) - 255
        predictedImg = predictedImg.astype(int)
        finalImg = decodedImg + predictedImg
        finalImg = np.clip(finalImg,0,255)
        finalImg = finalImg.astype(np.uint8)
        cv.imwrite('./data/residual/'+childName + 'finalImg'+str(i)+'.ppm', finalImg)
    # make boundary blur
    boundaryFlag = 0
    if boundaryFlag == 1:
        BlurSize = 7 
        if int(BlurSize) % 2 != 1:
            raise ValueError
        # get boundary mask
        img255 = np.zeros((Height,Width))
        img255 = img255.astype(np.uint8)
        img255.fill(255)
        imgMatrix = cv.warpPerspective(img255, Homography, (Width, Height), flags =cv.INTER_LANCZOS4)
        imgMatrixBlur = cv.blur(imgMatrix,(BlurSize,BlurSize))
        imgBoundaryMask = np.zeros((Height,Width))
        imgBoundaryMask = imgBoundaryMask.astype(np.uint8)
        imgNonBoundaryMask = np.zeros((Height,Width))
        imgNonBoundaryMask = imgNonBoundaryMask.astype(np.uint8)
        for i in range(0, Height):
            for j in range(0, Width):
                if imgMatrixBlur[i][j] > 0 and imgMatrixBlur[i][j] < 255:
                    imgBoundaryMask[i][j] = 255
                    imgNonBoundaryMask[i][j] = 0
                else:
                    imgBoundaryMask[i][j] = 0
                    imgNonBoundaryMask[i][j] = 255
        imgresidual = cv.imread('./data/residual/'+childName + 'finalImg'+str(20)+'.ppm')
        imgresidual = imgresidual.astype(np.uint8)
        img2GuessBlur = cv.GaussianBlur(imgresidual, (BlurSize, BlurSize),0,0)
        img2GuessBlur = cv.bitwise_and(img2GuessBlur,img2GuessBlur, mask = imgBoundaryMask)
        img2BoundaryBlur = cv.bitwise_and(imgresidual, imgresidual, mask = imgNonBoundaryMask)
        imgfinal = img2GuessBlur + img2BoundaryBlur
        cv.imwrite('./data/residual/'+childName + 'finalImg'+str(20)+'.ppm', imgfinal)
    return bppList 

def comparisonGroup(originalImgList, setName):
    '''
    individually encode all images within set by jp2'''
    for i in range(len(originalImgList)):
        thresholdCmd = "kdu_compress -i ./data/" + setName + str(i+1)+ ".ppm -o ./data/comparison/"+setName + str(i+1)+ ".jp2 Clayers=20"
        output = subprocess.check_output(thresholdCmd, shell=True)
    bppList = [[] for i in range(len(originalImgList))]
    for j in range(len(originalImgList)):
        for i in range(1,21):
            # kakadu command for expand file
            expandCmd = 'kdu_expand -i ./data/comparison/'+setName+str(j+1)+'.jp2 -o ./data/comparison/'+ setName + str(j+1) +'decoded'+ str(i) +'.ppm -layers ' + str(i)
            output = subprocess.check_output(expandCmd, shell=True)
            bppList[j].append(findBpp(output))
    return bppList


def main():
    methodcode = input('Please input the MST method code \n 1 for SIFT\n 2 for Histgram\n')
    if ord(methodcode) < 49 or ord(methodcode) >50:
        raise ValueError
    setcode = input('Please input the testset code \n 1 for cropped_img\n 2 for rotated_img\n 3 for zoomed_img \n 4 for set1 \n 5 for set2\n 6 for transformed_img\n')
    if ord(setcode) < 49 or ord(setcode) >54:
        raise ValueError
    testset = {'1':'cropped_img', '2':'rotated_img', '3':'zoomed_img', '4':'testset1_', '5':'testset2_', '6':'transformed_img'}
    setName = testset[setcode]
    imgList = [cv.imread(file) for file in glob.glob("./data/"+setName+'[0-9]'+".jpeg")]#this method arrange image in a random order, now it's just used for extract testset size
    g = mst.Graph(len(imgList))
    setSize = len(imgList)
    imgList = []  
    overAllBppList = []
    for i in range(setSize):
        originalImg = cv.imread("./data/"+setName+str(i+1)+".jpeg")
        imgList.append(originalImg)
    if methodcode == '1': 
        g.graph = mst.CalcSimilaritySIFT(imgList)
    elif methodcode == '2':
        g.graph = mst.CalcSimilarityHist(imgList)
    mstList = g.primMST()
    # encode the set
    for i in range(1, len(mstList)):
        parentName = setName + str(mstList[i]+1)
        childName = setName + str(i+1)
        Homography = predict(parentName, childName)
        threshold,Width,Height = encoder(parentName, childName)
        bppList = decoder(threshold, parentName, childName, Homography, Height, Width)
        overAllBppList.append(bppList)
        # print(bppList)
    # encode the comparison group
    cmprBppList = comparisonGroup(imgList, setName)
    cmprBppList = np.mean(cmprBppList, axis = 0)
    # plot the group PSNR vs bpp graph
    meanBppList = list(np.mean(overAllBppList, axis = 0))
    meanBppList = [0] + meanBppList
    metric.groupPsnrPlot(setName, meanBppList, cmprBppList)
    return None


def test():
    setName = 'rotated_img' 
    imgList = [cv.imread(file) for file in glob.glob("./data/"+setName+'[0-9]'+".jpeg")]#this method arrange image in a random order, now it's just used for extract testset size
    setSize = len(imgList)
    imgList = []  
    for i in range(setSize):
        originalImg = cv.imread("./data/"+setName+str(i+1)+".jpeg")
        imgList.append(originalImg)
    cmprBppList = comparisonGroup(imgList, setName)
    return cmprBppList 

if __name__ == '__main__':
    main() 
    # test()
    #to do list 
    #add the first point (only prediction)
    #add the comparison group

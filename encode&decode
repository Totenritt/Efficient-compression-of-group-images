import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import sys
import imutils
import glob
'''
the image u need 
'''
img1 = cv.imread('cropped_img2.jpeg')
img2 = cv.imread('cropped_img2_predicted.jpeg')
'''
create the ppm with a offset255, named it offsetImg.ppm
'''
img1int = img1.astype(int)
img2int = img2.astype(int)
diffimg = np.subtract(img1int,img2int)+ 255
diffimg255 = diffimg.astype(np.uint16)
cv.imwrite('./data/offsetImg.ppm', diffimg255)

'''
go cmd, input 'kdu_compress -i offsetImg.ppm -o offsetImg_output.jp2 Clayers=20'
then it output '44670, 44370, 44070, 43770, 43470, 43170, 42870, 42570, 42270, 41970,
        41670, 41370, 41070, 40770, 40470, 40170, 39870, 39570, 38723, 0'
copy these number and delete the space within numbers, like '44670,44370,44070,43770,43470,
43170,42870,42570,42270,41970,41670,41370,41070,40770,40470,40170,39870,39570,38723,0'
Then, input 'kdu_compress -i offsetImg.ppm -o offsetImgoutput.jp2 -slope 44670,...,0'
Then input 'kdu_expand -i offsetImg_output.jp2 -o decodedImg_20.ppm -layers 20' layers 1-20. layers 1 is the most blurred
So u get a decoded ppm image
'''
'''
img3 is the decodedImg_20 of layers 20, which is 0 in 'slope 44670,...,0'
'''
img3 = cv.imread( './jpeg2000test/decodedImg.ppm' ,-1)
img3_255 = img3.astype(int) - 255
img2 = img2.astype(int)
img5 = img2 + img3_255
threshold = 255
img5[img5>threshold] =255
img5[img5<0] = 0
img5 = img5.astype(np.uint8)
cv.imwrite('./jpeg2000test/finalImg.ppm' , img5)
cv.imwrite('./data/decoded_targetImg.jpeg', decoded_targetImg)



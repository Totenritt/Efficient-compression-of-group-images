import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import sys
import imutils
import mst 
import glob
from scipy.stats import entropy

def Generate_Naive_Testset_Cropping(img):
    """
    generate a simple test (list) from input image, the test set contains 5 different images. 
    img1 is the assumed mother image"""
    print("image shape is:", str(img.shape)[1:-1])
    IMG_WIDTH = img.shape[0]
    IMG_LENGTH = img.shape[1]
    BORDER = IMG_WIDTH // 5 
    MOTHER_WIDTH = IMG_WIDTH - 2 * BORDER
    MOTHER_LENGTH = IMG_LENGTH - 2 * BORDER
    if MOTHER_LENGTH < 0 or MOTHER_WIDTH < 0:
        raise ValueError('mother length or width should be positive interger')
    #split the image to first test set 
    img1 = img[BORDER:MOTHER_WIDTH + BORDER, BORDER: MOTHER_LENGTH + BORDER,:] #mother
    img2 = img[BORDER:MOTHER_WIDTH + BORDER, BORDER//2 : round(3/2 * BORDER),:] # img2 should have 50% overlap with mother, same for img3
    img3 = img[BORDER:MOTHER_WIDTH + BORDER, IMG_LENGTH - round(3/2*BORDER):IMG_LENGTH - BORDER // 2,:]
    img4 = img[IMG_WIDTH - round(3/2 * BORDER):IMG_WIDTH - BORDER//2,BORDER //2 :round(3/2*BORDER),:] #img4 should have 50% overlap with img2
    img5 = img[IMG_WIDTH - round(3/2* BORDER):IMG_WIDTH - BORDER //2 ,IMG_LENGTH - round(3/2*BORDER):IMG_LENGTH - BORDER // 2,:] #img5 should have 50% overlap with img3
    imgList = [img1, img2, img3, img4, img5]
    for count,image in enumerate(imgList, start = 1):
        filename = './data/cropped_img'+ str(count)+'.jpeg'
        cv.imwrite(filename, image)
    return imgList

def get_center(img):
    centerY = int(img.shape[0] / 2)
    centerX = int(img.shape[1] / 2)
    return centerX, centerY

def Generate_Naive_Testset_Rotation(img):
    """generate a simple test (list) from input image , the test set contains 5 different images. 
    img1 is the assumed mother image"""
    IMG_WIDTH = img.shape[0]
    IMG_LENGTH = img.shape[1]
    print("image shape is:", str(img.shape)[1:-1])
    WINDOW_SIZE = np.floor(min(IMG_LENGTH, IMG_WIDTH)/ 2 / np.sqrt(2))
    SET_SIZE = 5
    mother_center = get_center(img) 
    window_start_X= int(np.rint(mother_center[0] - WINDOW_SIZE//2))
    window_end_X = int(np.rint(mother_center[0] + WINDOW_SIZE //2))
    window_start_Y= int(np.rint(mother_center[1] - WINDOW_SIZE//2))
    window_end_Y = int(np.rint(mother_center[1] + WINDOW_SIZE //2))
    mother = img[window_start_Y:window_end_Y,window_start_X:window_end_X]
    cv.imwrite('rotated_img1.jpeg',mother)
    imgList = [mother]
    rotation_angle = 360/SET_SIZE #increment of rotation_angle
    for i in range(SET_SIZE -1):
        rotated_img =  imutils.rotate_bound(img, rotation_angle*(i+1)) #for rotation around center and avoid out of boundary cropping 
        rotated_center = get_center(rotated_img)
        window_start_X= int(np.rint(rotated_center[0] - WINDOW_SIZE//2))
        window_end_X = int(np.rint(rotated_center[0] + WINDOW_SIZE //2))
        window_start_Y= int(np.rint(rotated_center[1] - WINDOW_SIZE//2))
        window_end_Y = int(np.rint(rotated_center[1] + WINDOW_SIZE //2))
        sample_img = rotated_img[window_start_Y:window_end_Y, window_start_X: window_end_X]
        imgList.append(sample_img)
        filename = 'rotated_img' + str(i+2) + '.jpeg'
        cv.imwrite(filename, sample_img)
    return imgList

def Generate_Naive_Testset_Scaling(img):
    SET_SIZE = 3 
    if SET_SIZE % 2 != 1:
        raise ValueError('set size must be odd number')
    SCALING_FACTOR = 2 
    IMG_WIDTH = img.shape[0]
    IMG_LENGTH = img.shape[1]
    new_width = IMG_WIDTH
    new_length = IMG_LENGTH
    imgList = [img]
    num_downsampling = (SET_SIZE-1)//2
    num_upsampling = num_downsampling
    for i in range(num_upsampling):
        new_width = SCALING_FACTOR * new_width 
        new_length = SCALING_FACTOR * new_length
        new_size = (new_length, new_width)
        zoomed_img = cv.resize(img, new_size, interpolation=cv.INTER_LANCZOS4)
        imgList.append(zoomed_img)
    new_width = IMG_WIDTH
    new_length = IMG_LENGTH
    for i in range(num_downsampling):
        new_width //= SCALING_FACTOR 
        new_length //= SCALING_FACTOR
        new_size = (new_length, new_width)
        zoomed_img = cv.resize(img, new_size, interpolation=cv.INTER_AREA)
        imgList.append(zoomed_img)
    for count,image in enumerate(imgList, start = 1):
        filename = './data/zoomed_img'+ str(count)+'.jpeg'
        cv.imwrite( filename, image)
    return imgList

def Generate_Naive_Testset_Translation(img):
    STEP_SIZE = 55.5
    IMG_WIDTH = img.shape[0]
    IMG_LENGTH = img.shape[1]
    BORDER = IMG_WIDTH // 3 
    MOTHER_WIDTH = IMG_WIDTH - 2 * BORDER
    MOTHER_LENGTH = IMG_LENGTH - 2 * BORDER
    MatrixLeft = np.float32([[1, 0, -STEP_SIZE], [0, 1, 0]]) #tranformation matrix up 
    MatrixRight = np.float32([[1, 0, STEP_SIZE], [0, 1, 0]]) #transformation matrix down 
    MatrixDown = np.float32([[1, 0, 0], [0, 1, STEP_SIZE]]) #tranformation matrix left
    MatrixUp= np.float32([[1, 0, 0], [0, 1, -STEP_SIZE]]) #tranformation matrix right
    transMatrixList = [MatrixDown, MatrixUp, MatrixRight, MatrixLeft]
    mother = img[BORDER:MOTHER_WIDTH + BORDER, BORDER: MOTHER_LENGTH + BORDER,:]
    imgList = [mother]
    for matrix in transMatrixList:
        shifted = cv.warpAffine(img, matrix, (IMG_LENGTH, IMG_WIDTH),flags=cv.INTER_LANCZOS4)
        cropped = shifted[BORDER:MOTHER_WIDTH + BORDER, BORDER: MOTHER_LENGTH + BORDER,:]
        imgList.append(cropped)
    for count,image in enumerate(imgList):
        if count ==0:
            filename = './data/transformed_img_original' + '.jpeg'
        elif count == 1:
            filename = './data/transformed_img_down'+'.jpeg'
        elif count ==2:
            filename = './data/transformed_img_up'+'.jpeg'
        elif count ==3:
            filename = './data/transformed_img_right'+'.jpeg'
        elif count ==4:
            filename = './data/transformed_img_left'+'.jpeg'
        cv.imwrite(filename, image)
    return imgList

def Generate_Testset_One(img):
    IMG_WIDTH = img.shape[0]
    IMG_LENGTH = img.shape[1]
    WINDOW_SIZE = 1250 
    assert (IMG_LENGTH > WINDOW_SIZE and IMG_WIDTH>WINDOW_SIZE)
    mother_center = get_center(img) 
    #mother image is windowed original image
    window_start_X= int(np.rint(mother_center[0] - WINDOW_SIZE//2))
    window_end_X = int(np.rint(mother_center[0] + WINDOW_SIZE //2))
    window_start_Y= int(np.rint(mother_center[1] - WINDOW_SIZE//2))
    window_end_Y = int(np.rint(mother_center[1] + WINDOW_SIZE //2))
    mother = img[window_start_Y:window_end_Y,window_start_X:window_end_X]
    imgList = [mother]
    #image 2 is mother image rotation clockwise by 45 degree
    rotation_angle = 45 
    rotated_img =  imutils.rotate_bound(img, rotation_angle) #for rotation around center and avoid out of boundary cropping 
    rotated_center = get_center(rotated_img)
    window_start_X= int(np.rint(rotated_center[0] - WINDOW_SIZE//2))
    window_end_X = int(np.rint(rotated_center[0] + WINDOW_SIZE //2))
    window_start_Y= int(np.rint(rotated_center[1] - WINDOW_SIZE//2))
    window_end_Y = int(np.rint(rotated_center[1] + WINDOW_SIZE //2))
    testset1_img2= rotated_img[window_start_Y:window_end_Y, window_start_X: window_end_X]
    imgList.append(testset1_img2)
    #image3 is mother image transform up by 55.5 pixel 
    STEP_SIZE = 15.5
    MatrixUp= np.float32([[1, 0, 0], [0, 1, -STEP_SIZE]]) #tranformation matrix up
    shifted = cv.warpAffine(img, MatrixUp, (IMG_LENGTH, IMG_WIDTH),flags=cv.INTER_LANCZOS4)
    window_start_X= int(np.rint(mother_center[0] - WINDOW_SIZE//2))
    window_end_X = int(np.rint(mother_center[0] + WINDOW_SIZE //2))
    window_start_Y= int(np.rint(mother_center[1] - WINDOW_SIZE//2))
    window_end_Y = int(np.rint(mother_center[1] + WINDOW_SIZE //2))
    testset1_img3 = shifted[window_start_Y:window_end_Y,window_start_X:window_end_X]
    imgList.append(testset1_img3)
    #image4 is mother image scale down by 2
    SCALING_FACTOR = 2
    MOTHER_WIDTH = mother.shape[0]
    MOTHER_LENGTH = mother.shape[1]
    new_width = MOTHER_WIDTH//SCALING_FACTOR
    new_length = MOTHER_LENGTH//SCALING_FACTOR
    new_size = (new_width, new_length)
    zoomed_img = cv.resize(mother, new_size, interpolation=cv.INTER_LANCZOS4)
    imgList.append(zoomed_img)
    #write image
    for count,image in enumerate(imgList, start = 1):
        filename = './data/testset1_'+ str(count)+'.jpeg'
        cv.imwrite( filename, image)
    return imgList

def Generate_Testset_Two(img):
    IMG_WIDTH = img.shape[0]
    IMG_LENGTH = img.shape[1]
    WINDOW_SIZE = 2880
    mother_center = get_center(img) 
    #mother image is windowed original image
    window_start_X= int(np.rint(mother_center[0] - WINDOW_SIZE//2))
    window_end_X = int(np.rint(mother_center[0] + WINDOW_SIZE //2))
    window_start_Y= int(np.rint(mother_center[1] - WINDOW_SIZE//2))
    window_end_Y = int(np.rint(mother_center[1] + WINDOW_SIZE //2))
    mother = img[window_start_Y:window_end_Y,window_start_X:window_end_X]
    imgList = [mother]
    #image2 is mother image scale down by 2
    SCALING_FACTOR = 2
    MOTHER_WIDTH = mother.shape[0]
    MOTHER_LENGTH = mother.shape[1]
    new_width = MOTHER_WIDTH//SCALING_FACTOR
    new_length = MOTHER_LENGTH//SCALING_FACTOR
    new_size = (new_length, new_width)
    img2 = cv.resize(mother, new_size, interpolation=cv.INTER_LANCZOS4)
    imgList.append(img2)
    #image 3 is mother image rotation counter clockwise by 30 degree
    rotation_angle = -30 
    rotated_img1 =  imutils.rotate_bound(img, rotation_angle) #for rotation around center and avoid out of boundary cropping 
    rotated_center = get_center(rotated_img1)
    window_start_X= int(np.rint(rotated_center[0] - WINDOW_SIZE//2))
    window_end_X = int(np.rint(rotated_center[0] + WINDOW_SIZE //2))
    window_start_Y= int(np.rint(rotated_center[1] - WINDOW_SIZE//2))
    window_end_Y = int(np.rint(rotated_center[1] + WINDOW_SIZE //2))
    img3= rotated_img1[window_start_Y:window_end_Y, window_start_X: window_end_X]
    imgList.append(img3)
    #image4 is image2 rotation clockwise by 45 degree 
    new_width = IMG_WIDTH*SCALING_FACTOR
    new_length = IMG_LENGTH*SCALING_FACTOR
    new_size = (new_length, new_width)
    zoomed_img = cv.resize(img, new_size, interpolation=cv.INTER_LANCZOS4)
    cv.imwrite('check.jpeg', zoomed_img)
    rotation_angle = 45 
    rotated_img = imutils.rotate_bound(zoomed_img, rotation_angle) #for rotation around center and avoid out of boundary cropping 
    rotated_center = get_center(rotated_img)
    window_start_X= int(np.rint(rotated_center[0] - WINDOW_SIZE))
    window_end_X = int(np.rint(rotated_center[0] + WINDOW_SIZE))
    window_start_Y= int(np.rint(rotated_center[1] - WINDOW_SIZE))
    window_end_Y = int(np.rint(rotated_center[1] + WINDOW_SIZE))
    img4= rotated_img[window_start_Y:window_end_Y, window_start_X: window_end_X]
    imgList.append(img4)
    #image5 is image 3 transform left by 33.3 pixel
    STEP_SIZE = 33.3 
    MatrixLeft = np.float32([[1, 0, -STEP_SIZE], [0, 1, 0]]) #tranformation matrix up 
    shifted = cv.warpAffine(rotated_img1, MatrixLeft, (rotated_img1.shape[1], rotated_img1.shape[0]),flags=cv.INTER_LANCZOS4)
    shifted_center = get_center(shifted)
    window_start_X= int(np.rint(shifted_center[0] - WINDOW_SIZE//2))
    window_end_X = int(np.rint(shifted_center[0] + WINDOW_SIZE //2))
    window_start_Y= int(np.rint(shifted_center[1] - WINDOW_SIZE//2))
    window_end_Y = int(np.rint(shifted_center[1] + WINDOW_SIZE //2))
    img5 = shifted[window_start_Y:window_end_Y,window_start_X:window_end_X]
    imgList.append(img5)
    #write image
    for count,image in enumerate(imgList, start = 1):
        filename = './data/testset2_'+ str(count)+'.jpeg'
        cv.imwrite(filename, image)
    return imgList

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

def detect_sift(img):
    sift = cv.SIFT_create() # extract SIFT feature
    grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    kp = sift.detect(grey, None) # keypoint location
    kp, des = sift.compute(grey, kp) # des is the eigenvector 
    #print(des.shape) # eigenvector dimension is 128 
    return kp, des, grey

def BFORCE_match_sift(img1, img2): # brute force matching, NO RANSAC
    kp1, des1, grey1 = detect_sift(img1)
    kp2, des2, grey2 = detect_sift(img2)
    bf = cv.BFMatcher(crossCheck=True)
    matches = bf.match(des1, des2)
    res = cv.drawMatches(img1, kp1, img2, kp2, matches, None)
    return res

def FLANN_match_sift(img1,img2,KPLimitNum): #if determined num of key points less than KPLimitNum, system exit. if NO Setting Limit, use 'KPLimitNum=None'
    if KPLimitNum == None or (isinstance (KPLimitNum,int) and KPLimitNum >= 1):
        pass
    else:
        print("in match_sift KPLimitNum should be None or a positive integer")
        sys.exit()
    kp1, des1, grey1 = detect_sift(img1)
    kp2, des2, grey2 = detect_sift(img2)
    FLANN_INDEX_KDTREE = 0
    indexParams = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    searchParams = dict(checks=10)
    flann = cv.FlannBasedMatcher(indexParams, searchParams)
    matches = flann.knnMatch(des1, des2, k=2)
    matchesMask = [[0, 0] for i in range(len(matches))]
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.3*n.distance and KPLimitNum == None:
            matchesMask[i] = [1, 0]
        elif m.distance < 0.3*n.distance and KPLimitNum > 0:
            matchesMask[i] = [1, 0]
            KPLimitNum = KPLimitNum - 1
    drawPrams = dict(matchColor=(0, 255, 0),
                 singlePointColor=(255, 0, 0),
                 matchesMask=matchesMask,
                 flags=0)
    res = cv.drawMatchesKnn(grey1, kp1, grey2, kp2, matches, None, **drawPrams)
    return res

def RANSAC_match_sift(img1, img2):
    kp1, des1, grey1 = detect_sift(img1)
    kp2, des2, grey2 = detect_sift(img2)
    '''
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1,des2,k=2)
    '''    
    bf = cv.BFMatcher(crossCheck=True)
    matches = bf.match(des1, des2)
    '''
    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)
    '''
    good = []
    for m in matches:
        good.append(m)
    MIN_MATCH_COUNT = 10
    if len(good)>MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()
        h,w,xxxx = img1.shape
        del xxxx
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv.perspectiveTransform(pts,M)
        img2 = cv.polylines(img2,[np.int32(dst)],True,255,3, cv.LINE_AA)
    else:
        print("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
        matchesMask = None
    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                    singlePointColor = None,
                    matchesMask = matchesMask, # draw only inliers
                    flags = 2)
    res = cv.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
    BFres = cv.drawMatches(img1, kp1, img2, kp2, matches, None)
    return BFres, res, M

def generate_histogram(img, do_print, filename):
    """
    img: can be a grayscale or color image. We calculate the Normalized histogram of this image.
    do_print: if or not print the result histogram
    @return: will return both histogram and the grayscale image 
    """
    if len(img.shape) == 3: # img is colorful, so we convert it to grayscale
        gr_img = np.mean(img, axis=-1)
    else:
        gr_img = img
    '''now we calc grayscale histogram'''
    gr_hist = np.zeros([511])
    for x_pixel in range(gr_img.shape[0]):
        for y_pixel in range(gr_img.shape[1]):
            pixel_value = int(gr_img[x_pixel, y_pixel])
            gr_hist[pixel_value + 255] += 1
    '''normalizing the Histogram'''
    gr_hist /= (gr_img.shape[0] * gr_img.shape[1])
    if do_print:
        print_histogram(gr_hist, filename, title="Normalized Histogram")
    return gr_hist, gr_img

def print_histogram(_histogram, filename, title):
    plt.figure()
    entro = entropy(_histogram)
    plt.title(title+' with entropy:'+str(entro)[:4])
    X = np.linspace(-255, 255, 511, endpoint=True)
    plt.plot(X, _histogram, color='#ef476f')
    plt.ylabel('Percentage of Pixels')
    plt.xlabel('Pixel Value')
    plt.savefig("./data/hist_" + filename +'.jpeg')

def plotDiff(parent, filename):
    '''
    plot the difference histogram between original image and predicted image
    for all image within testset   '''
    for i in range(1, len(parent)):
        parentImg = cv.imread('./data/'+ filename + str(parent[i]+1)+'.jpeg')
        childImg = cv.imread('./data/'+ filename + str(i+1) +'.jpeg')
        img2Reg = generatePredictImg(parentImg, childImg, True, filename = filename + str(i + 1))
    return None

def generatePredictImg(parentImg, childImg, toPrint, filename):
    '''
    generate predictive image of child image with respect to parent image
    if toPrint is false don't print the histogram
    '''
    BFres, res, Homography = RANSAC_match_sift(parentImg, childImg)
    childWidth,childHeight,childChannel = childImg.shape
    childReg = cv.warpPerspective(parentImg, Homography, (childHeight, childWidth))
    cv.imwrite('./data/'+filename+'_predicted.jpeg', childReg)
    childImg = childImg.astype(int)
    childReg = childReg.astype(int)
    diffMatrix = np.subtract(childImg,childReg)
    if toPrint:
        gr_hist1, gr_img1 = generate_histogram(diffMatrix, True, filename)
    return childReg

def codeResidual(img):
    '''
    add 255 to residual images save to ppm format  
    '''
    #convert the 8bit image to 16bit unsigned, and add an 255 offset
    img = img + 255
    img = img.astype(np.uint16) 
    # normalize the image to fit in 16bit ppm format
    # offsetImg = cv.normalize(img, None, 0, 2**16-1, cv.NORM_MINMAX, dtype=cv.CV_16U)
    #write the image 
    cv.imwrite('./data/residual/offsetImg.ppm', img)
    return img


if __name__ == '__main__':
    # img = cv.imread('./data/flower.jpg', cv.IMREAD_UNCHANGED)
    # scaling_test_set = Generate_Naive_Testset_Scaling(img)
    # simMatrix = CalcSimilarityHist(scaling_test_set)
    # print(simMatrix)
    #generate MST
    # setcode = input('Please input the testset code \n 1 for cropped_img\n 2 for rotated_img\n 3 for zoomed_img \n 4 for set1 \n 5 for set2\n')
    # if ord(setcode) < 49 or ord(setcode) >53:
    #     raise ValueError
    # testset = {'1':'cropped_img', '2':'rotated_img', '3':'zoomed_img', '4':'testset1_', '5':'testset2_'}
    # name = testset[setcode]
    # imgList = [cv.imread(file) for file in glob.glob("./data/"+ name +"?.jpeg")]
    # g = mst.Graph(len(imgList))
    # g.graph = CalcSimilarityHist(imgList)
    # print(g.graph)
    # parent = g.primMST()
    # codeResidual(parent, name)
    
# img2 = cv.imread('rotated_img2.jpeg')
# img1 = cv.imread('street.jpg')
# kp1, des1, grey1, kp1 = detect_sift(img1)
# kp2, des2, grey2, kp2 = detect_sift(img2)
# kp1convert=cv.KeyPoint_convert(kp1)
# FLANN_INDEX_KDTREE = 0
# # parameter1：indexParams
# #    for SIFT and SURF, we can pass index_params=dict(algorithm=FLANN_INDEX_KDTREE, trees=5)。
# #    for ORB, we can pass index_params=dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12）。
# indexParams = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
# # parameter2：searchParams designate traverse times, the higher the more accurat but cost more time
# searchParams = dict(checks=50)
# # use FlannBasedMatcher to find the nearest matching
# flann = cv.FlannBasedMatcher(indexParams, searchParams)
# # use knnMatch for matching and return result as matches
# matches = flann.knnMatch(des1, des2, k=2)
# # calcualte useful keypoint by mask
# matchesMask = [[0, 0] for i in range(len(matches))]
# # select required point by distance
# for i, (m, n) in enumerate(matches):
#     if m.distance < 0.5*n.distance: # 0.5 controls the number of points  
#         matchesMask[i] = [1, 0]

# drawPrams = dict(matchColor=(0, 255, 0),
#                  singlePointColor=(255, 0, 0),
#                  matchesMask=matchesMask,
#                  flags=0)
# # match result picture 
# res = cv.drawMatchesKnn(grey1, kp1, grey2, kp2, matches, None, **drawPrams)
# plt.imshow(res)
# plt.show()

# img = cv.imread('./data/book.jpg')
# testSet = Generate_Naive_Testset_Cropping(img)
# simMatrix = CalcSimilarityHist(testSet)
# print(simMatrix)

# img = cv.imread('street.jpeg', cv.IMREAD_UNCHANGED)
# rotation_test_set = Generate_Naive_Testset_Rotation(img)
# simMatrix = CalcSimilarityHist(rotation_test_set)
# print(simMatrix)

# img = cv.imread('pawel.jpeg')
# testSet = Generate_Naive_Testset_Scaling(img)
# simMatrix = CalcSimilarityHist(testSet)
# print(simMatrix)

# img = cv.imread('./data/pawel.jpeg')
# testSet = Generate_Naive_Testset_Translation(img)
# simMatrix = CalcSimilarityHist(testSet)
# print(simMatrix)

# img = cv.imread('./data/mountain.jpeg')
# testSet = Generate_Testset_Two(img)
# simMatrix = CalcSimilarityHist(testSet)
# print(simMatrix)

    img = cv.imread('./data/rotated_img1.jpeg')
    img = img.astype(np.uint16) + 255 
    # offsetImg = cv.normalize(img, None, 0, 2**16-1, cv.NORM_MINMAX, dtype=cv.CV_16U)
    cv.imwrite('./data/residual/residual.ppm', img)
    img = img - 255
    img = img.astype(np.uint8)
    cv.imwrite('./data/residual/residual.ppm', img)

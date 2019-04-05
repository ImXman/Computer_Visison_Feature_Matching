# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 16:48:35 2019

@author: Yang Xu
"""

import sys
import cv2
import numpy as np
import scipy.ndimage.filters as filters

frame1_path=str(sys.argv[1])
frame2_path=str(sys.argv[2])
descriptor=int(sys.argv[3])
ratio =float(sys.argv[4])

##harris corner detection to find points
def find_points(image,block_size=5,max_block=9,k=0.05):
    
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    #blur = cv2.GaussianBlur(gray,(7,7),3)
    #blur = np.float32(blur)
    #dst = cv2.cornerHarris(blur,2,3,0.05)
    dst = cv2.cornerHarris(gray,block_size,3,k)
    #find local maximum harris corner responses
    maxima = filters.maximum_filter(dst, max_block)
    maxima = (dst == maxima)
    maxima = maxima.astype(np.int)
    #dst = cv2.dilate(dst,None)
    dst = maxima*dst
    dst=dst.flatten()
    dst=dst[dst!=0]
    points = np.where(maxima==1)
    points = np.asarray(points).T
    ##rank corners
    orders =np.argsort(dst).tolist()
    points=points[orders]
    points=points[::-1,:]
    ##we only return top 500 corners as key points; you can change the number
    ##below
    points=points[:500,:]
    
    return points

##orientations of points
def ori_points(image,points):
    
    orientation = np.zeros((points.shape[0],1))
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(7,7),3)
    blur = np.float32(blur)
    sobel = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    ori_x = cv2.filter2D(blur,-1,sobel)
    ori_y = cv2.filter2D(blur,-1,sobel.T)
    for i in range(points.shape[0]):
        x,y = points[i,:]
        d=np.arctan2(ori_y[x,y],ori_x[x,y])*180/np.pi
        if d<0:
            orientation[i,:]=int(d+360.0)
        else:
            orientation[i,:]=int(d)
        
    return orientation
    
##use pixel intensity as descriptor
def points_intensity(image,points):
    
    intensity=np.zeros((points.shape[0],25))
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    #blur = cv2.GaussianBlur(gray,(7,7),3)
    #blur = np.float32(blur)
    new_image=cv2.copyMakeBorder(gray,2,2,2,2,cv2.BORDER_CONSTANT,value=0)
    m,n = gray.shape
    for i in range(points.shape[0]):
        x,y = points[i,:]
        a = new_image[x:x+5,y:y+5].flatten()
        ##my reason is that lighting issue should be addressed when pixel
        ##intensity is used as descriptor
        if np.std(a)==0:
            intensity[i,:] = np.zeros(25)
        else:
            intensity[i,:]=(a-np.mean(a))/np.std(a)
        #intensity[i,:]=a
        
    return intensity

##rotate the image without cutting the image
def rotation(image,angle):
    
    (h, w) = image.shape
    (cX, cY) = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    
    ##new bounding of image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
 
    ##adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
 
    ##perform the actual rotation and return the image
    dst=cv2.warpAffine(image, M, (nW, nH))
    
    return dst
    
def points_descriptor(image,points,orientation):
    
    descriptor=np.zeros((points.shape[0],25))
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    #blur = cv2.GaussianBlur(gray,(7,7),3)
    #blur = np.float32(blur)
    gray = np.float32(gray)
    ##instead of rotating the whole image, we only rotate a 40X40 patch with
    ##the interesting point as the center. Therefore, we need to find the
    ##coordinate after rotation, because the interesting point will still be 
    ##the center of the rotated pathc
    ##To make sure we can still patch a 40X40 subimage, I broaden the image
    ##with 30 empty pixels that is big enough
    gray=cv2.copyMakeBorder(gray,30,30,30,30,cv2.BORDER_CONSTANT,value=0)
    #scaler = StandardScaler()
    for i in range(points.shape[0]):
        x,y = points[i,:]
        patch = gray[x:x+61,y:y+61]
        new_patch = rotation(patch,orientation[i,:])
        (cX,cY) = (new_patch.shape[0]//2,new_patch.shape[1]//2)
        new_patch = new_patch[cX-20:cX+20,cY-20:cY+20]
        new_patch = cv2.resize(new_patch,(5,5),interpolation=cv2.INTER_AREA)
        a=new_patch.flatten()
        if np.std(a)==0:
            descriptor[i,:] = np.zeros(25)
        else:
            descriptor[i,:]=(a-np.mean(a))/np.std(a)
        
    return descriptor

def feature_match(des1,des2,ratio=0.5):
    
    match1=[]
    match2=[]
    
    for i in range(des1.shape[0]):
        ##if the vector is all zeros, we will not consider matching
        if np.std(des1[i,:])!=0:
            d=des2-des1[i,:]
            d=np.linalg.norm(d, axis=1)
            orders =np.argsort(d).tolist()
            if d[orders[0]]/d[orders[1]]<=ratio:
                match1.append((i,orders[0]))
        
    for i in range(des2.shape[0]):
        ##if the vector is all zeros, we will not consider matching
        if np.std(des2[i,:])!=0:
            d=des1-des2[i,:]
            d=np.linalg.norm(d, axis=1)
            orders =np.argsort(d).tolist()
            if d[orders[0]]/d[orders[1]]<=ratio:
                match2.append((orders[0],i))
            
    ##find good matches in rotation tests both ways
    match = list(set(match1).intersection(set(match2)))
    
    return match
    
###############################################################################

def main():
    
    f1 = cv2.imread(frame1_path)
    f2 = cv2.imread(frame2_path)

    ##find key points
    p1= find_points(f1,block_size=5,max_block=9,k=0.05)
    p2= find_points(f2,block_size=5,max_block=9,k=0.05)
    
    if descriptor==0:
        
        ##descriptor by pixel intensity
        des1 = points_intensity(f1,p1)
        des2 = points_intensity(f2,p2)
    else:    
        ##descriptor by orientation
        ori1 = ori_points(f1,p1)
        ori2 = ori_points(f2,p2)
        des1 = points_descriptor(f1,p1,ori1)
        des2 = points_descriptor(f2,p2,ori2)
    
    ##matching via ratio test both ways
    mtc= feature_match(des1,des2,ratio=ratio)

    ##we will make keypoint and dmatch lists to draw matches, 
    #and visualize our matching by opencv
    kp1 = [cv2.KeyPoint(p1[i,1], p1[i,0],5) for i in range(p1.shape[0])]
    kp2 = [cv2.KeyPoint(p2[i,1], p2[i,0],5) for i in range(p2.shape[0])]
    matches = [cv2.DMatch(i[0],i[1],1) for i in mtc]
    
    #Draw matches.
    draw_params = dict(matchColor = (0,255,0),
                       singlePointColor = (255,0,0),
                       #matchesMask = matchesMask,
                       flags = 0)
    matching = cv2.drawMatches(f1,kp1,f2,kp2,matches, None,**draw_params)
    cv2.imwrite("matching.jpeg", matching)

    ##image alignment
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)
    for i, match in enumerate(matches):
        points1[i, :] = kp1[match.queryIdx].pt
        points2[i, :] = kp2[match.trainIdx].pt

    ##RANSAC for alignment
    h, mask = cv2.findHomography(points2, points1, cv2.RANSAC)
    
    stitch = cv2.warpPerspective(f2, h, (f1.shape[1] + f2.shape[1], f1.shape[0]))
    cv2.imwrite("alignment.jpeg", stitch)
    
    ##stitch images
    ##cautious: it doesn't work for perspective case
    stitch[0:f2.shape[0], 0:f2.shape[1],:] = f1
    cv2.imwrite("stitched.jpeg", stitch)

if __name__ == '__main__':
    main()

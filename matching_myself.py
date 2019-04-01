# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 14:31:16 2019

@author: xuyan
"""

import cv2
import numpy as np
import seaborn as sns
import feature_matching as fm
from sklearn.cluster import KMeans

###############################################################################
img_close = cv2.imread('close.jpg')
img_close_re = cv2.resize(img_close,(960,720),interpolation=cv2.INTER_AREA)
gray_close= cv2.cvtColor(img_close_re,cv2.COLOR_BGR2GRAY)

img_close2 = cv2.imread('close2.jpg')
img_close2_re = cv2.resize(img_close2,(960,720),interpolation=cv2.INTER_AREA)
gray_close2= cv2.cvtColor(img_close_re,cv2.COLOR_BGR2GRAY)

img_far = cv2.imread('far.jpg')
gray_far= cv2.cvtColor(img_far,cv2.COLOR_BGR2GRAY)

###############################################################################
#sift = cv2.xfeatures2d.SIFT_create()

#kp1, des1 = sift.detectAndCompute(gray_close,None)
#kp2, des2 = sift.detectAndCompute(gray_far,None)

p1= fm.find_points(img_close_re,block_size=5,max_block=9,k=0.05)
p2= fm.find_points(img_close2_re,block_size=5,max_block=9,k=0.05)
#kp1 = [cv2.KeyPoint(p1[i,1], p1[i,0],5) for i in range(p1.shape[0])]
#kp2 = [cv2.KeyPoint(p2[i,1], p2[i,0],5) for i in range(p2.shape[0])]

##segamentation
img_f=[]
for i in range(img_close_re.shape[2]):
    img_f.append(img_close_re[:,:,i].flatten())
img_f= np.asarray(img_f).T
x=np.repeat(range(img_close_re.shape[0]),img_close_re.shape[1], axis=0)
y=np.tile(range(img_close_re.shape[1]),img_close_re.shape[0])
img_f=np.column_stack((img_f,x))
img_f=np.column_stack((img_f,y))
img_f = img_f.astype(np.float32)

kmeans = KMeans(n_clusters=16, random_state=0).fit(img_f)
img_cp = np.asarray(kmeans.labels_)
img_cp = img_cp.reshape(img_close_re.shape[0],img_close_re.shape[1])
sns.heatmap(img_cp,cmap="RdBu")
img_cp[img_cp!=1]=0

img_f2=[]
for i in range(img_close2_re.shape[2]):
    img_f2.append(img_close2_re[:,:,i].flatten())
img_f2= np.asarray(img_f2).T
x=np.repeat(range(img_close2_re.shape[0]),img_close2_re.shape[1], axis=0)
y=np.tile(range(img_close2_re.shape[1]),img_close2_re.shape[0])
img_f2=np.column_stack((img_f2,x))
img_f2=np.column_stack((img_f2,y))
img_f2 = img_f2.astype(np.float32)

kmeans2 = KMeans(n_clusters=16, random_state=0).fit(img_f2)
img_cp2 = np.asarray(kmeans2.labels_)
img_cp2 = img_cp2.reshape(img_close_re.shape[0],img_close_re.shape[1])
sns.heatmap(img_cp2,cmap="RdBu")
img_cp2[img_cp2!=1]=0

obj_p = np.where(img_cp==1)
obj_p = np.asarray(obj_p).T
aset = set([tuple(x) for x in p1])
bset = set([tuple(x) for x in obj_p])
common = list(aset.intersection(bset))
common = np.array(common)

obj_p = np.where(img_cp2==1)
obj_p = np.asarray(obj_p).T
aset = set([tuple(x) for x in p2])
bset = set([tuple(x) for x in obj_p])
common2 = list(aset.intersection(bset))
common2 = np.array(common2)

kp1 = [cv2.KeyPoint(common[i,1], common[i,0],5) for i in range(common.shape[0])]
#kp2 = [cv2.KeyPoint(common2[i,1], common2[i,0],5) for i in range(common2.shape[0])]
kp2 = [cv2.KeyPoint(p2[i,1], p2[i,0],5) for i in range(p2.shape[0])]

ori1 = fm.ori_points(img_close_re,common)
ori2 = fm.ori_points(img_close2_re,p2)
des1 = fm.points_descriptor(img_close_re,common,ori1)
des2 = fm.points_descriptor(img_close2_re,p2,ori2)

#des1 = sift.compute(gray_close,kp1)[1]
#des2 = sift.compute(gray_close2,kp2)[1]

###############################################################################
bf = cv2.BFMatcher()
matches = bf.match(des1,des2)
matches = bf.knnMatch(des1,des2, k=2)

mtc = fm.feature_match(des1,des2,ratio=0.85)
matches = [cv2.DMatch(i[0],i[1],1) for i in mtc]

good = []
for m,n in matches:
    if m.distance < 1*n.distance:
        good.append(m)

#matches = bf.match(des1,des2)
#matches = sorted(good, key = lambda x:x.distance)
matches = sorted(matches, key = lambda x:x.distance)

draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   #matchesMask = matchesMask,
                   flags = 0)
matching = cv2.drawMatches(img_close_re,kp1,img_far,kp2,None, None,**draw_params)
matching = cv2.drawMatches(img_close_re,kp1,img_far,kp2,good, None,**draw_params)
matching = cv2.drawMatches(img_close_re,kp1,img_close2_re,kp2,matches, None,**draw_params)

cv2.imwrite('keypoints.jpg',matching)
cv2.imwrite('matching_obj.jpg',matching)
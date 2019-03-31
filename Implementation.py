# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 23:24:19 2019

@author: Yang Xu
"""

import cv2
import feature_matching as fm

##read images
##translation
#f1 = "images/translation/yosemite1.jpg"
#f2 = "images/translation/yosemite2.jpg"

##illumination
f1 = "images/illuminatoin/img1.png"
f2 = "images/illuminatoin/img3.png"

##perspective
#f1 = "images/prespective/img1.png"
#f2 = "images/prespective/img2.png"

f1 = cv2.imread(f1)
f2 = cv2.imread(f2)

##find key points
p1= fm.find_points(f1,block_size=5,max_block=9,k=0.05)
p2= fm.find_points(f2,block_size=5,max_block=9,k=0.05)

##descriptor by pixel intensity
des1 = fm.points_intensity(f1,p1)
des2 = fm.points_intensity(f2,p2)

##descriptor by orientation
ori1 = fm.ori_points(f1,p1)
ori2 = fm.ori_points(f2,p2)
des1 = fm.points_descriptor(f1,p1,ori1)
des2 = fm.points_descriptor(f2,p2,ori2)

##matching via ratio test both ways
mtc= fm.feature_match(des1,des2,ratio=0.75)

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
cv2.imwrite("matching.png", matching)
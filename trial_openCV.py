# -*- coding: utf_8 -*-

import numpy as np
import cv2
import matplotlib.pyplot as plt


#cups = cv2.imread('./data/learn_chara/0/0-11.png')

# preprocess by blurring and grayscale
#cups_preprocessed  = cv2.cvtColor(cups, cv2.COLOR_BGR2GRAY)
 
# find binary image with thresholding
#cups_edges = cv2.Canny(cups_preprocessed, threshold1=90, threshold2=110)
#plt.imshow(cv2.cvtColor(cups_edges, cv2.COLOR_GRAY2RGB))
#cv2.imwrite('cups-edges.jpg', cups_edges)

#実行は、 ./data/my_learn8 ./data/my_test2

img = cv2.imread('./data/learn_chara/0/0-1.png')
gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
sift = cv2.xfeatures2d.SIFT_create()
kp = sift.detect(gray,None)
img=cv2.drawKeypoints(gray,kp,img)
cv2.imwrite('sift_keypoints.jpg',img)
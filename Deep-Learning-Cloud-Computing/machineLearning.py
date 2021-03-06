# -*- coding: utf-8 -*-
"""MIR_Project.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1-NY1E9oDGRUpnY83eFLkD-LKbnagxTHX
"""

from IPython.display import Image, HTML, display
from matplotlib import pyplot as plt
from keras.preprocessing import image
from keras.models import Model, load_model
from matplotlib.pyplot import imread
import numpy as np
import os
import cv2
import os
import csv
import math
from skimage.transform import rotate
from skimage.feature import local_binary_pattern
from skimage import data
from skimage.color import label2rgb


"""## **Calcul de l'histogramme de couleur**"""

def histogram_color(image):
    histR = cv2.calcHist([image],[0],None,[256],[0,256])
    histG = cv2.calcHist([image],[1],None,[256],[0,256])
    histB = cv2.calcHist([image],[2],None,[256],[0,256])
    hist = [histR, histG, histB]
    return hist

"""## **Descripteur SIFT**"""

def siftDescriptor(image):
  sift = cv2.xfeatures2d.SIFT_create()
  gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)        # Convertion en niveau de gris
  keypoints = sift.detect(gray,None)
  keypoints_sift, descriptor_sift = sift.compute(gray, keypoints)
  return keypoints_sift, descriptor_sift

"""## **Descripteur ORB**"""

def orbDescriptor(image):
  orb = cv2.ORB()
  gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)         # Convertion au niveau de gris
  keypoints, descriptor_orb = orb.detectAndCompute(gray, None)
  return keypoints, descriptor_orb

"""## **Descripteur LBP**

"""

def lbpDescriptor(image):
# settings for LBP
  METHOD = 'uniform'
  radius = 3
  n_points = 8 * radius
  gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)         # Convertion au niveau de gris
  lbp = local_binary_pattern(gray, n_points, radius, METHOD)
  return lbp

"""## **Calculer le descripteur SIFT**"""

def Cal_SIFT(image):
  if image is not None:
    kp, des = siftDescriptor(image)
  if des is not None:
    sift_features = des.tolist()
  sift_featuresA = np.array(sift_features)
  return sift_featuresA

"""## **Calculer le descripteur ORB**"""

def Cal_ORB(image):
  orb = cv2.ORB_create()
  if image is not None:
    kp, des = orb.detectAndCompute(image, None)
  if des is not None:
    orb_features = des.tolist()
  else:
    orb_featuresA = []

  orb_featuresA = np.array(orb_features)
  return orb_featuresA

"""## **Calculer le descripteur LBP**"""

def Cal_LBP(image):
  if image is not None:
    des = lbpDescriptor(image)
    lbp_features = des.tolist()
  lbp_featuresA = np.array(lbp_features)
  return lbp_featuresA

"""**## Calcul du descripteur GLCM**"""

from skimage.feature import greycomatrix
def glcm(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    glcm = greycomatrix(image, distances=[1], angles=[0, np.pi/4, np.pi/2],
                    symmetric=True, normed=True)
    return glcm

"""## **Focntions de calcul de similarité**"""

def euclidean(l1, l2):
    n = min(len(l1), len(l2))
    return np.sqrt(np.sum((l1[:n] - l2[:n])**2))

def chiSquareDistance(l1, l2):
    n = min(len(l1), len(l2))
    return np.sum((l1[:n] - l2[:n])**2 / l2[:n])

def bhatta(l1, l2):
    n = min(len(l1), len(l2))
    N_1, N_2 = np.sum(l1[:n])/n, np.sum(l2[:n])/n
    score = np.sum(np.sqrt(l1[:n] * l2[:n]))
    num = round(score, 2)
    den = round(math.sqrt(N_1*N_2*n*n), 2)
    return math.sqrt( 1 - num / den )

def flann(a,b):
    a = np.float32(a)
    b = np.float32(b)
    FLANN_INDEX_KDTREE = 1
    INDEX_PARAM_TREES = 5
    SCH_PARAM_CHECKS = 50
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=INDEX_PARAM_TREES)
    sch_params = dict(checks=SCH_PARAM_CHECKS)
    flannMatcher = cv2.FlannBasedMatcher(index_params, sch_params)
    matches = list(map(lambda x: x.distance, flannMatcher.match(a, b)))
    return np.mean(matches)

def bruteForceMatching(a, b):
    a = a.astype('uint8')
    b = b.astype('uint8')
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = list(map(lambda x: x.distance, bf.match(a, b)))
    return np.mean(matches)


"""## **Combiner les descripteurs**"""

#2 images + 2 descpteur, + methode de calcule de distance !
def combiner(image1, image2, Descp1,Descp2):
  DescpA1 =Descp1(image1)
  DescpA2 =Descp1(image2)
  DescpB1 =Descp2(image1)
  DescpB2 =Descp2(image2)

  combined_Features1= np.concatenate((DescpA1, DescpB1), axis=None)
  combined_Features2= np.concatenate((DescpA2, DescpB2), axis=None)

  return combined_Features1, combined_Features2


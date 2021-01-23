from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications import vgg16
from keras.applications import vgg19
from keras.applications import resnet50
from keras.applications import inception_v3
from keras.applications import mobilenet
from keras.applications import xception
from keras.layers.pooling import GlobalAveragePooling2D
from matplotlib.pyplot import imread
import matplotlib.pyplot as plt
import numpy as np
import operator
import math
import os
import tensorflow as tf
from keras.models import Model
from keras.models import load_model
import csv


import imghdr
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense, Activation,Dropout
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.models import load_model
from keras.optimizers import Adam
from tensorflow.python.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.preprocessing import image
from keras.utils import *
import warnings 
warnings.filterwarnings('ignore')

#dÃ©finition des fonctions de calcul des distances

def euclidianDistance(l1,l2): 
    distance = 0
    length = min(len(l1),len(l2))
    for i in range(length):
        distance += pow((l1[i] - l2[i]), 2)
    return math.sqrt(distance)
	
def chi2_distance(histA, histB, eps = 1e-10):
    #calculating the chi squared distance
    d = 0.5 * np.sum([((a-b) ** 2) / (a + b + eps) for (a, b) in zip(histA, histB)])
    
    #return the chi squared distance
    return d
	

def search(queryFeatures,direction,mesuredistance,sortie ):
         
    #fonction qui permet de trouver les images les plus semblables Ã  une image d'entrer
    #Inpout : queryFeatures: features de l'image d'entrer; mesuredistance : permet de choisir comment les distances seront calculer
    #sortie : nombre de voisins Ã  la sortie
    #Output : une matrice de la taille de Sortie, comprenant les features et le nom des plus proches voisins 
    #make a dictionary for thr results
    results = {}
    
    #open the index file for reading
    with open(direction) as f:
        # initializing the csv reader
        reader = csv.reader(f)

        if mesuredistance == 1:
          print('dans 1')
        
          #loop over the rows in the index
          for row in reader:
            

            # parse out the imageID and features, then calculate the chi-squared distance between the saved features and the features of our image
            features = [float(x) for x in row[1:]]  
            d = euclidianDistance(queryFeatures,features)
            # now we have the distance between the two feature vectors. we now update the results dictionary
            results[row[0]] = d

        if mesuredistance == 2:
          print('dans 2')
          #loop over the rows in the index
          for row in reader:
            

            # parse out the imageID and features, then calculate the chi-squared distance between the saved features and the features of our image
            features = [float(x) for x in row[1:]]  
            d = chi2_distance(features, queryFeatures)
            # now we have the distance between the two feature vectors. we now update the results dictionary
            results[row[0]] = d
            
        # closing the reader
        f.close()
        
    # sort the results such that the dictionary starts with smaller values as they will be closest to the given image
    results = sorted([(v,k) for (k,v) in results.items()])
    
    #return our results
    return results[:sortie]

#Fonction qui permet d'extraire les caractÃ©ristiques de l'image recherche 
def image_entree(image, model):
  #files = 'static/uploads/'
  #target = os.path.join(files, requete)
  #print (target)
  #image = load_img(target, target_size=shape)
  # convert the image pixels to a numpy array
  
  image = img_to_array(image)
  # reshape data for the model
  image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
  # prepare the image for the VGG model
  image = preprocess_input(image) #fonction importer
  # predict the probability across all output classes
  feature = model.predict(image)
  #feature = np.array(feature[0])
  return feature[0]
  
#2 images + 2 descpteur, + methode de calcule de distance !
def combiner(image, modelvgg16,modelInception):
  # fonction qui permet de combiner les caractéristique de deux modèles
  # Input > image: image requête ,  modelvgg16,modelInception : les deux modèles
  #output > combined_Features1 : les caratéristiques combinées, qr : nom de l'image
  DescpA1 = (modelvgg16.predict(image))[0]  
  DescpB1= (modelInception.predict(image))[0] 


  combined_Features1= np.concatenate((DescpA1, DescpB1), axis=None)

  return combined_Features1

def rappelPrecision(voisins):
  
  tempon2 =os.path.splitext(os.path.basename(voisins[0][1]))[0]
  print("===========tempon :",tempon2)
  imgClasse=int((tempon2.split('_'))[0])

  precision=[0]
  rappel = [0]

  for idx,elmt in enumerate(voisins):
    tempon2 =os.path.splitext(os.path.basename(elmt[1]))[0]
    elmtClass=int((tempon2.split('_'))[0])
    print("===========image classe :",elmtClass)

    if imgClasse==elmtClass:
      precision.append(len(precision)/(idx+1))
      rappel.append((len(precision)-1)/500)

  return [rappel,precision] 

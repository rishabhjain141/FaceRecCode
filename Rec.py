from facerec.feature import *
from facerec.distance import *
from facerec.classifier import NearestNeighbor
from facerec.model import PredictableModel
from PIL import Image
import numpy as np
import cv2
import math,operator
import os,sys
from facerec.lbp import *
from scipy import stats


def read_images(path, sz=None):
    """Reads the images in a given folder, resizes images on the fly if size is given.

    Args:
        path: Path to a folder with subfolders representing the subjects (persons).
        sz: A tuple with the size Resizes

    Returns:
        A list [X,y]

            X: The images, which is a Python list of numpy arrays.
            y: The corresponding labels (the unique number of the subject, person) in a Python list.
    """
    c = 0
    X,y = [], []
    for dirname, dirnames, filenames in os.walk(path):
        for subdirname in dirnames:
            subject_path = os.path.join(dirname, subdirname)
            for filename in os.listdir(subject_path):
                name,ext=os.path.splitext(filename)
                if (ext=='.jpg'):
                    try:
                        im = Image.open(os.path.join(subject_path, filename))
                        im = im.convert("L")
                        if (im.size[0]>100 and im.size[1]>100):
    #                     if (1):
                            # resize to given size (if given)
                            if (sz is not None):
                                im = im.resize(sz, Image.ANTIALIAS)
                            X.append(np.asarray(im, dtype=np.uint8))
                            y.append(c)
    #                 except IOError, (errno, strerror):
    #                     print "I/O error({0}): {1}".format(errno, strerror)
                    except:
                        print "Unexpected error:", sys.exc_info()[0]
                        raise
            c = c+1
    return [X,y]

Xtrain,ytrain=read_images('/home/rishabh/f1/faces aviral train/',(100,129))
Xtest,ytest=read_images('/home/rishabh/f1/faces aviral/',(100,129))

mod1=PredictableModel(PCA(num_components=50),NearestNeighbor(k=1))
mod2=PredictableModel(PCA(num_components=50),NearestNeighbor(k=1,dist_metric=CosineDistance()))    
mod3=PredictableModel(Fisherfaces(num_components=50),NearestNeighbor(k=1))
mod4=PredictableModel(Fisherfaces(num_components=50),NearestNeighbor(k=1,dist_metric=CosineDistance()))
mod5=PredictableModel(SpatialHistogram(),NearestNeighbor(k=1))
mod6=PredictableModel(SpatialHistogram(),NearestNeighbor(k=1,dist_metric=CosineDistance())) 
mod7=PredictableModel(SpatialHistogram(lbp_operator=LPQ()),NearestNeighbor(k=1))
mod8=PredictableModel(SpatialHistogram(lbp_operator=LPQ()),NearestNeighbor(k=1,dist_metric=CosineDistance()))
mod9=PredictableModel(SpatialHistogram(),NearestNeighbor(k=1,dist_metric=ChiSquareDistance())) 
mod10=PredictableModel(SpatialHistogram(),NearestNeighbor(k=1,dist_metric=NormalizedCorrelation())) 

mod1.compute(Xtrain,ytrain)
mod2.compute(Xtrain,ytrain)
mod3.compute(Xtrain,ytrain)
mod4.compute(Xtrain,ytrain)
mod5.compute(Xtrain,ytrain)
mod6.compute(Xtrain,ytrain)
mod7.compute(Xtrain,ytrain)
mod8.compute(Xtrain,ytrain)
mod9.compute(Xtrain,ytrain)
mod10.compute(Xtrain,ytrain)


#For Training Size 3

p=np.array(np.ones(len(Xtest))*9,dtype=int)
count=0
for i in range(len(Xtest)):
     d10=mod10.predict(Xtest[i])
     if (d10[1]['distances']<0.33):
         count+=1
         p[i]=int(d10[0])
#         print 'mod9',(d10[1]['distances']),p[i],ytest[i]
         continue
    d9=mod9.predict(Xtest[i])
    if (d9[1]['distances']<40):
        count+=1
        p[i]=int(d9[0])
#         print 'mod9',abs(d9[1]['distances']),p[i],ytest[i]
        continue
     d6=mod6.predict(Xtest[i])
     if (abs(d6[1]['distances'])>0.68):
         count+=1
         p[i]=int(d6[0])
#         print 'mod6',abs(d6[1]['distances']),p[i],ytest[i]
         continue
    d1=mod1.predict(Xtest[i])
    if (d1[1]['distances']<1):
        count+=1
        p[i]=d1[0]
#         print 'mod1',abs(d1[1]['distances']),p[i],ytest[i]
        continue

    d4=mod4.predict(Xtest[i])
    if (abs(d4[1]['distances'])>0.999):
        count+=1
        p[i]=d4[0]
#         print 'mod4',abs(d4[1]['distances']),p[i],ytest[i]
        continue

    d2=mod2.predict(Xtest[i])
    if (abs(d2[1]['distances'])>0.96):
        count+=1
        p[i]=int(d2[0])
#         print 'mod2',abs(d2[1]['distances']),p[i],ytest[i]
        continue
    

    d8=mod8.predict(Xtest[i])
    if (abs(d8[1]['distances'])>0.6):
        count+=1
        p[i]=int(d8[0])
#         print 'mod8',abs(d8[1]['distances']),p[i],ytest[i]
        continue
    
    d7=mod7.predict(Xtest[i])
    if (abs(d7[1]['distances'])<0.95):
        count+=1
        p[i]=int(d7[0])
#         print 'mod7',abs(d7[1]['distances']),p[i],ytest[i]
        continue 
    

    d3=mod3.predict(Xtest[i])
    if (d3[1]['distances']<290):
        count+=1
        p[i]=int(d3[0])
#         print 'mod3',abs(d3[1]['distances']),p[i],ytest[i]
        continue
    
     d5=mod5.predict(Xtest[i])
     if (d5[1]['distances']<1.35):
         count+=1
         p[i]=int(d5[0])
 #         print 'mod5',abs(d5[1]['distances']),p[i],ytest[i]
         continue
        

print np.sum(p==ytest)
print count
print len(ytest)






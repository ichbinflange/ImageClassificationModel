from fileinput import filename
import matplotlib.pyplot as plt
import numpy as np
from numpy import dtype
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers.experimental.preprocessing import RandomFlip,RandomRotation
import cv2
import pandas as pd 

#### add a scritpt to get dataset from cloud unzip and start preprocessing *
#-------------------------------------------------------------------------
#Images were renamed and all train files must be in sub directories containing classes

#from keras.preprocessing.images import ImageDataGenerator
from PIL import Image,ImageOps

import os
import glob

import pandas as pd


##class datapreprocessing

##Data Augumentation 
def data_augumenter():
    data_augumentation = tf.keras.Sequential()
    data_augumentation.add(RandomFlip('horizontal'))
    data_augumentation.add(RandomRotation(0.2))

    return data_augumentation

##takes in an Image by default opens it and converts it to numpy as well as resizing the image
def DisplayImageToArray(img,size):

    image = Image.open(img)
    image = ImageOps.fit(image,size,Image.ANTIALIAS)
    img_frompil = np.array(image)

    return img_frompil

##concatenate data for preprocessing
def concatenatedata(xin):
    count = 0
    xret = []
    for i in range(len(xin)):
        count = count + i
        xret = xret + xin[count]
    return xret


##transform th output from names to binary target variables cast the values to data type int 32
def transformy(yin):
    y = pd.Series(yin)
    yout = pd.get_dummies(y)
    yout = yout.to_numpy('int32')

    return yout


##load and preprocess the dataset and get it ready for training

def loadArrayofImages(inputpath,foldername):

    ##loads the dataset which is already saved as directory, creates a dictionary with keys and values for targets and dataset
    mainpath = os.path.join(os.getcwd(),foldername)
    file_list = glob.glob(os.path.join(os.getcwd(),foldername,'*'))
    dic = {}
    count = 0
    for subfolders in file_list:
        classname = os.path.basename(subfolders)
        for abspath in glob.glob(os.path.join(mainpath,classname,'*')):
            imgpath = os.path.basename(abspath)
            filename = imgpath.split('_')
            if filename[0] not in dic.keys():
                dic[filename[0]]= [abspath]
            else:
                dic[filename[0]].append(abspath)

    ##get the data from dictionaries
    x = []
    y = []
    imgsize = 224,224
    for key,value in dic.items():
        images = [DisplayImageToArray(x,imgsize) for x in value]
        labels = [key]*len(images)
        x.append(images)
        y.append(labels)

    ##concatenatedata
    xtrain= concatenatedata(x)
    ytrain = concatenatedata(y)

    ##shuffle the dataset
    xtrain,ytrain = shuffle(xtrain,ytrain)

    ###cast xtrain to float 32 and transform ydata to binary targetvalues
    xtrain = np.asarray(xtrain,dtype= np.float32)
    ytrain = transformy(ytrain)


    return xtrain,ytrain

inputpath = r'C:\Users\IT\Desktop\Objectclassification'
foldername = 'Train'
x,y = loadArrayofImages(inputpath,foldername)

#class to create a dataset

###rename train folder by replacing filesnames with under_score
###early dataset used had a bad naming convection
#train is the old folder that contains dataset and invalid naming convectin
#train1 is the new folder that will contain the dataset
def renamefiles():
    file_list = glob.glob(os.path.join(os.getcwd(),'train','*.jpg'))
    for names in file_list:
        imagepath = os.path.basename(names)
        imagepath = imagepath.split('.')
        newname = ('_'.join(imagepath[:2]))+'.jpg'
        newpath = glob.glob(os.path.join(os.getcwd(),'train1',newname))
        os.rename(names,newname)

#renamefiles() schuld be used only once and in a different script please add destination folder where renamed filed to be stored


def createdatasetdir(inputpath,foldername):

    os.chdir (inputpath)
    if not os.path.exists(foldername):
        os.mkdir(foldername)
    mainpath = os.path.join(os.getcwd(),foldername)
    imagedirectiory = r'C:\Users\IT\Desktop\Objectclassification'
    os.chdir(mainpath)

    file_list = glob.glob(os.path.join(imagedirectiory ,'train1','*.jpg'))
    for abspaths in file_list:
        imgpath = os.path.basename(abspaths)
        filename = imgpath.split('_')
        os.chdir(mainpath)
        if not os.path.exists(filename[0]):
            os.mkdir(filename[0])
            path = glob.glob(os.path.join(mainpath,filename[0]))
            os.chdir(r'{}'.format(path[0]))
            img = Image.open(abspaths)
            img.save(imgpath)
        else:
            path = glob.glob(os.path.join(mainpath,filename[0]))
            os.chdir(r'{}'.format(path[0]))
            img = Image.open(abspaths)
            img.save(imgpath)

#inputpath = r'C:\Users\IT\Desktop\Objectclassification'
#createdatasetdir(inputpath,'Train')
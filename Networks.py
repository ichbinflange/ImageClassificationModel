from glob import glob
#import keras
import numpy as np
import matplotlib.pyplot as plt

from Image_Pipelines import loadArrayofImages,data_augumenter
from keras.models import Sequential,Model

import mlflow
import mlflow.keras
from keras.callbacks import EarlyStopping
import tensorflow as tf

from tensorflow.keras.applications import efficientnet,inception_v3

from PIL import Image,ImageOps

import glob
import os

os.environ['TP_CPP_MIN_LOG_LEVEL'] = '3'

##set input path and name of training folder
inputpath = r'C:\Users\IT\Desktop\Objectclassification'
foldername = 'Train'

#set neuralnetwork architechture name
name = 'inceptionet_v3'

##load Images and convert to numpy array
xtrain,ytrain = loadArrayofImages(inputpath,foldername)


#set mlflowparameters
remote_uri = 'htttp://127.0.0.1:5000'
mlflow.set_tracking_uri(remote_uri)
experiment = name

#set up tensorflow Model
input_shape = (224,224,3)
preprocessing_input = tf.keras.applications.inception_v3.preprocess_input
base_model = tf.keras.applications.inception_v3.InceptionV3(input_shape =input_shape,include_top = False,weights ='imagenet')
base_model.trainable = False
inputs = tf.keras.Input(shape = input_shape)

x = data_augumenter()(inputs)
x = preprocessing_input(x)
x = base_model(x,training = False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.Dense(2,activation ='softmax')(x)
model = tf.keras.Model(inputs,outputs)


model.summary()


mlflow.tensorflow.autolog()

es = EarlyStopping(monitor='val_loss',mode ='auto',patience = 3)
model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001),loss = 'binary_crossentropy',metrics = ['accuracy'])
history = model.fit(xtrain,ytrain,epochs = 10,validation_split = 0.2,batch_size = 16,callbacks = [es])

model.save('Savedmodels/inceptionv3')


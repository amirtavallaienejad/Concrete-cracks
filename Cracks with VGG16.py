# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 21:52:49 2023

@author: up202111331
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.layers import Dense,Flatten
from keras.models import Model
from keras.applications import vgg16
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

import os

root_dir=r'C:\Users\up202111331\Desktop\2nd Semester\Machine Learning\Second Report\Concrete Crack Images'

for dirname, _, filenames in os.walk(root_dir):
    for filename in filenames:
        print(os.path.join(dirname, filename))




data_gen = ImageDataGenerator(rotation_range=30 ,
                              horizontal_flip=True, 
                              vertical_flip = True,
                              rescale = 1/255.0,
                             validation_split=0.1)

train_generator = data_gen.flow_from_directory(root_dir,
                                              target_size = (224,224),
                                              batch_size = 16,
                                              class_mode = 'binary',
                                               color_mode = 'rgb',
                                               subset='training',shuffle=True
                                              )

valid_generator = data_gen.flow_from_directory(root_dir,
                                              target_size = (224,224),
                                              batch_size = 16,
                                              class_mode = 'binary',
                                               color_mode = 'rgb',
                                               subset='validation',shuffle=True
                                              )



IMAGE_SIZE = [224,224]

## Add Preprocessing to the front
vgg_16 = vgg16.VGG16(input_shape=IMAGE_SIZE + [3],weights='imagenet',include_top=False)


for layer in vgg_16.layers:
    layer.trainable=False
    
## Add layer
    
x = Flatten()(vgg_16.output)
pred = Dense(1,activation='sigmoid')(x)

## Create a model object
model = Model(inputs=vgg_16.input , outputs=pred)

## Let's compile our model
model.compile(optimizer='adam',
             loss='binary_crossentropy',
             metrics=['accuracy'])

model.summary()


## Fit the Model
result = model.fit(train_generator,
                   validation_data=valid_generator,
                   epochs=10,
                   callbacks=[
                        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True
        )
                   ]
                    )






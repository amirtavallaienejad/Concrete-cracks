# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 13:51:23 2023

@author: up202111331
"""


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from pathlib import Path
from sklearn.model_selection import train_test_split

import tensorflow as tf


from keras.layers import Dense,Flatten
from keras.models import Model

from sklearn.metrics import confusion_matrix, classification_report



positive_dir = Path(r'C:\Users\up202111331\Desktop\2nd Semester\Machine Learning\Second Report\Concrete Crack Images\Positive')
negative_dir = Path(r'C:\Users\up202111331\Desktop\2nd Semester\Machine Learning\Second Report\Concrete Crack Images\Negative')

# Creating DataFrames


def generate_df(image_dir, label):
    filepaths = pd.Series(list(image_dir.glob(r'*.jpg')), name='Filepath').astype(str)
    labels = pd.Series(label, name='Label', index=filepaths.index)
    df = pd.concat([filepaths, labels], axis=1)
    return df


positive_df = generate_df(positive_dir, label="POSITIVE")
negative_df = generate_df(negative_dir, label="NEGATIVE")

all_df = pd.concat([positive_df, negative_df], axis=0).sample(frac=1.0, random_state=1).reset_index(drop=True)

print(all_df)





train_df, test_df = train_test_split(
    all_df.sample(6000, random_state=1),
    train_size=0.7,
    shuffle=True,
    random_state=1
)


# Loading Image Data

train_gen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

test_gen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255
)

train_data = train_gen.flow_from_dataframe(
    train_df,
    x_col='Filepath',
    y_col='Label',
    target_size=(229,229),
    color_mode='rgb',
    class_mode='binary',
    batch_size=32,
    shuffle=True,
    seed=42,
    subset='training'
)

val_data = train_gen.flow_from_dataframe(
    train_df,
    x_col='Filepath',
    y_col='Label',
    target_size=(229,229),
    color_mode='rgb',
    class_mode='binary',
    batch_size=32,
    shuffle=True,
    seed=42,
    subset='validation'
)

test_data = train_gen.flow_from_dataframe(
    test_df,
    x_col='Filepath',
    y_col='Label',
    target_size=(229,229),
    color_mode='rgb',
    class_mode='binary',
    batch_size=32,
    shuffle=False,
    seed=42
)



# Training 


from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import Model, layers
import tensorflow as tf




# Import the Inception V3 model
from tensorflow.keras.applications.inception_v3 import InceptionV3

# save the pretrained weights
local_weights_file = r'C:\Users\up202111331\Desktop\2nd Semester\Machine Learning\Second Report\Second Coding\InceptionV3\inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'

# run the model
pre_trained_model = InceptionV3(input_shape = (229, 229, 3),
                                include_top = False,
                                weights = None)

# set the weight to pretrained weight
pre_trained_model.load_weights(local_weights_file)

# Make all the layers in the pre-trained model
for layer in pre_trained_model.layers:
    layer.trainable = False
    
pre_trained_model.summary()


# use the part of pre_trained model from input layer until the layer 'mixed7'
last_layer = pre_trained_model.get_layer('mixed7')
last_output = last_layer.output

# Then add our layers

# Flatten teh output layer to 1 dimension
x = layers.Flatten()(last_output)

# Add fully connected layer with 1,024 hidden units and ReLU activation
x = layers.Dense(1024, activation='relu')(x)

# Add dropout layer to prevent overfitting
x = layers.Dropout(0.2)(x)

# Add a final sigmoid layer for classification
x = layers.Dense(1, activation = 'sigmoid')(x)

model = Model(pre_trained_model.input, x)
model.compile(optimizer = RMSprop(learning_rate=0.0001),
             loss = 'binary_crossentropy',
             metrics = ['accuracy'])

model.summary()



## Fit the Model
result = model.fit(train_data,
                    validation_data=val_data,
                    epochs=20,
                    callbacks=[
                        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True
        )
                    ]
                    )
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=100,
    batch_size = 32,
    verbose = 1
)



#visualizing



fig = px.line(
    history.history,
    y=['loss', 'val_loss'],
    labels={'index': "Epoch", 'value': "Loss"},
    title="Training and Validation Loss Over Time"
)

fig.show()
    
def evaluate_model(model, test_data):
    
    results = model.evaluate(test_data, verbose=0)
    loss = results[0]
    acc = results[1]
    
    print("    Test Loss: {:.5f}".format(loss))
    print("Test Accuracy: {:.2f}%".format(acc * 100))
    
    y_pred = np.squeeze((model.predict(test_data) >= 0.5).astype(np.int))
    cm = confusion_matrix(test_data.labels, y_pred)
    clr = classification_report(test_data.labels, y_pred, target_names=["NEGATIVE", "POSITIVE"])
    
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='g', vmin=0, cmap='Blues', cbar=False)
    plt.xticks(ticks=np.arange(2) + 0.5, labels=["NEGATIVE", "POSITIVE"])
    plt.yticks(ticks=np.arange(2) + 0.5, labels=["NEGATIVE", "POSITIVE"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix of InceptionV3 Network")
    plt.show()
    
    print("Classification Report:\n----------------------\n", clr)


evaluate_model(model, test_data)




# detecing the incorrect data



y_pred = np.squeeze((model.predict(test_data) >= 0.5).astype(np.int))
mistake_idx = (y_pred != test_data.labels).nonzero()[0]
print(len(mistake_idx), "mistakes.")
print("Indices:", mistake_idx)


# Display the detection mistakes
plt.figure(figsize=(20, 10))

for i, idx in enumerate(mistake_idx):
    
    # Get batch number and image number (batch of 32 images)
    batch = idx // 32
    image = idx % 32
    
    plt.subplot(6,6, i+1)
    plt.imshow(test_data[batch][0][image])
    plt.title("No crack detected" if y_pred[idx] == 0 else "Crack detected", color='red')
    plt.axis('off')
    
plt.suptitle("Detection Mistakes in InceptionV3 network", fontsize=20)
plt.show()


# evaluation the performance


import matplotlib.pyplot as plt

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

# x-axis: epochs, y-axis: acc
plt.plot(epochs, acc, 'r', label='Trainig accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')

plt.legend()

plt.show()

plt.plot(epochs, loss, 'r', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and validation loss')

plt.legend()

plt.show()







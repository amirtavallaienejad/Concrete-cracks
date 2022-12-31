# -*- coding: utf-8 -*-

import numpy as np 
#“NumPy, short for Numerical Python, has long been a cornerstone of numerical computing in Python. 
# It provides the data structures, algorithms, and library glue 
# needed for most scientific applications involving numerical data in Python. 
# NumPy contains, among other things:
# A fast and efficient multidimensional array object ndarray
# Functions for performing element-wise computations with arrays or mathematical operations between arrays
# Tools for reading and writing array-based datasets to disk
# Linear algebra operations, Fourier transform, and random number generation
# A mature C API to enable Python extensions and native C or C++ code to access NumPy’s data

import pandas as pd 
# “pandas provides high-level data structures and functions designed to make
#  working with structured or tabular data intuitive and flexible.
#  Since its emergence in 2010, it has helped enable Python to be a
#  powerful and productive data analysis environment.”
# “The pandas name itself is derived from panel data, an econometrics term 
# for multidimensional structured datasets, and a play on the phrase Python data analysis.”

import matplotlib.pyplot as plt

# “matplotlib is the most
#  popular Python library for producing plots and other
#  two-dimensional data visualizations. It was originally created by
#  John D. Hunter and is now maintained by a large team of 
#  developers. It is designed for creating plots suitable for
#  publication.  While there are other visualization libraries
#  available to Python programmers, matplotlib is still widely
#  used and integrates reasonably well with the rest of the  ecosystem.”

import seaborn as sns

#Seaborn is a Python data visualization library based on matplotlib. 
#It provides a high-level interface for drawing attractive and informative statistical graphics.

import plotly.express as px

#Plotly's Python graphing library makes interactive, publication-quality graphs. 
#Examples of how to make line plots, scatter plots, area charts, bar charts, error bars, 
#box plots, histograms, heatmaps, subplots, multiple-axes, polar charts, and bubble charts. 

from pathlib import Path

#This module offers classes representing filesystem paths with semantics appropriate for different operating systems. 
#Path classes are divided between pure paths, which provide purely computational operations without I/O, 
#and concrete paths, which inherit from pure paths but also provide I/O operations.

from sklearn.model_selection import train_test_split

#Scikit-learn is a Python module integrating a wide range of state-of-the-art machine learning algorithms 
#for medium-scale supervised and unsupervised problems. This package focuses on bringing machine learning 
#to non-specialists using a general-purpose high-level language. Emphasis is put on ease of use, performance, 
#documentation, and API consistency. It has minimal dependencies and is distributed under the simplified BSD license, 
#encouraging its use in both academic and commercial settings.

import tensorflow as tf

#TensorFlow is a free and open-source software library for machine learning and artificial intelligence. 
#It can be used across a range of tasks but has a particular focus on training and inference of deep neural networks.
#TensorFlow was developed by the Google Brain team for internal Google use in research and production. 
#The initial version was released under the Apache License 2.0 in 2015. Google released the updated version of TensorFlow, 
#named TensorFlow 2.0, in September 2019.
#TensorFlow can be used in a wide variety of programming languages, including Python, JavaScript, C++, and Java. 
#This flexibility lends itself to a range of applications in many different sectors.

from sklearn.metrics import confusion_matrix, classification_report


#In the field of machine learning and specifically the problem of statistical classification, a confusion matrix, also known as an error matrix,
#is a specific table layout that allows visualization of the performance of an algorithm, typically a supervised learning one
#(in unsupervised learning it is usually called a matching matrix). Each row of the matrix represents the instances in an actual class while 
#each column represents the instances in a predicted class, or vice versa – both variants are found in the literature.
#The name stems from the fact that it makes it easy to see whether the system is confusing two classes (i.e. commonly mislabeling one as another).

positive_dir = Path(r'C:\Users\up202111331\Desktop\2nd Semester\Machine Learning\Second Report\Concrete Cracks\Positive')
negative_dir =Path(r'C:\Users\up202111331\Desktop\2nd Semester\Machine Learning\Second Report\Concrete Cracks\Negative')


# Creating DataFrames

def generate_df(image_dir, label):
    filepaths = pd.Series(list(image_dir.glob(r'*.jpg')), name='Filepath').astype(str)
    labels = pd.Series(label, name='Label', index=filepaths.index)
    df = pd.concat([filepaths, labels], axis=1)
    return df

#labeling the photos

positive_df = generate_df(positive_dir, label="POSITIVE")
negative_df = generate_df(negative_dir, label="NEGATIVE")



all_df = pd.concat([positive_df, negative_df], axis=0).sample(frac=1.0, random_state=1).reset_index(drop=True)
all_df

print(all_df)

#Train_test_splitting

train_df, test_df = train_test_split(
    all_df.sample(6000, random_state=1),
    train_size=0.7,
    shuffle=True,
    random_state=1
)

# Loading Image Data

#normalizing train and make a validation part
#tf.keras.preprocessing.image.ImageDataGenerator
#Generate batches of tensor image data with real-time data augmentation.

train_gen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

#normalizing test

test_gen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255
)


#The flow_from_dataframe() method takes the Pandas DataFrame and the path to a directory and generates batches of augmented/normalized data.

#Arguments specific to flow_from_dataframe:
	#directory — (str)Path to the directory which contains all the images.set this to None if your x_col contains absolute_paths pointing to each image files instead of just filenames.
	#x_col — (str) The name of the column which contains the filenames of the images.
	#y_col — (str or list of str) If class_mode is not “raw” or not “input” you should pass the name of the column which contains the class names.None, if used for test_generator.
	#class_mode — (str) Similar to flow_from_directory, this accepts “categorical”(default), ”binary”, ”sparse”, ”input”, None and also an extra argument “raw”.If class_mode is set to “raw” it treats the data in the column or list of columns of the dataframe as raw target values(which means you should be sure that data in these columns must be of numerical datatypes), will be helpful if you’re building a model for regression task like predicting the angle from the images of steering wheel or building a model that needs to predict multiple values at the same time.For Test generator: Set this to None, to return only the images.
	#batch_size: For train and valid generator you can keep this according to your needs but for test generator:Set this to some number that divides your total number of images in your test set exactly.Why this only for test_generator? Actually, you should set the “batch_size” in both train and valid generators to some number that divides your total number of images in your train set and valid respectively, but this doesn’t matter before because even if batch_size doesn’t match the number of samples in the train or valid sets and some images gets missed out every time we yield the images from generator, but it would be sampled the very next epoch you train.But for the test set, you should sample the images exactly once, no less or no more. If Confusing, just set it to 1(but maybe a little bit slower).
	#shuffle: Set this to False(For Test generator only, for others set True), because you need to yield the images in “order”, to predict the outputs and match them with their unique ids or filenames.
	#drop_duplicates: If you’re for some reason don’t want duplicate entries in your dataframe’s x_col, set this to False, default is True.
	#validate_filenames: whether to validate image filenames in x_col. If True, invalid images will be ignored. Disabling this option can lead to speed-up in the instantiation of this class if you have a huge amount of files, default is True.

train_data = train_gen.flow_from_dataframe(
    train_df,
    x_col='Filepath',
    y_col='Label',
    target_size=(120, 120),
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
    target_size=(120, 120),
    color_mode='rgb',
    class_mode='binary',
    batch_size=32,
    shuffle=True,
    seed=42,
    subset='validation'
)

test_data = test_gen.flow_from_dataframe(
    test_df,
    x_col='Filepath',
    y_col='Label',
    target_size=(120, 120),
    color_mode='rgb',
    class_mode='binary',
    batch_size=32,
    shuffle=False,
    seed=42
)


# Training
#2D convolution layer (e.g. spatial convolution over images).
#This layer creates a convolution kernel that is convolved with the layer input to produce a tensor of outputs. 
#If use_bias is True, a bias vector is created and added to the outputs. Finally, if activation is not None, it is applied to the outputs as well.

#Arguments

#filters: Integer, the dimensionality of the output space (i.e. the number of output filters in the convolution).
#kernel_size: An integer or tuple/list of 2 integers, specifying the height and width of the 2D convolution window. Can be a single integer to specify the same value for all spatial dimensions.
#strides: An integer or tuple/list of 2 integers, specifying the strides of the convolution along the height and width. Can be a single integer to specify the same value for all spatial dimensions. Specifying any stride value != 1 is incompatible with specifying any dilation_rate value != 1.
#padding: one of "valid" or "same" (case-insensitive). "valid" means no padding. "same" results in padding with zeros evenly to the left/right or up/down of the input. When padding="same" and strides=1, the output has the same size as the input.
#data_format: A string, one of channels_last (default) or channels_first. The ordering of the dimensions in the inputs. channels_last corresponds to inputs with shape (batch_size, height, width, channels) while channels_first corresponds to inputs with shape (batch_size, channels, height, width). It defaults to the image_data_format value found in your Keras config file at ~/.keras/keras.json. If you never set it, then it will be channels_last. Note that the channels_first format is currently not supported by TensorFlow on CPU.
#dilation_rate: an integer or tuple/list of 2 integers, specifying the dilation rate to use for dilated convolution. Can be a single integer to specify the same value for all spatial dimensions. Currently, specifying any dilation_rate value != 1 is incompatible with specifying any stride value != 1.
#groups: A positive integer specifying the number of groups in which the input is split along the channel axis. Each group is convolved separately with filters / groups filters. The output is the concatenation of all the groups results along the channel axis. Input channels and filters must both be divisible by groups.
#activation: Activation function to use. If you don't specify anything, no activation is applied (see keras.activations).
#use_bias: Boolean, whether the layer uses a bias vector.
#kernel_initializer: Initializer for the kernel weights matrix (see keras.initializers). Defaults to 'glorot_uniform'.
#bias_initializer: Initializer for the bias vector (see keras.initializers). Defaults to 'zeros'.
#kernel_regularizer: Regularizer function applied to the kernel weights matrix (see keras.regularizers).
#bias_regularizer: Regularizer function applied to the bias vector (see keras.regularizers).
#activity_regularizer: Regularizer function applied to the output of the layer (its "activation") (see keras.regularizers).
#kernel_constraint: Constraint function applied to the kernel matrix (see keras.constraints).
#bias_constraint: Constraint function applied to the bias vector (see keras.constraints).



#Max pooling operation for 2D spatial data.

#pool_size

#integer or tuple of 2 integers, window size over which to take the maximum. (2, 2) will take the max value over a 2x2 pooling window. 
#If only one integer is specified, the same window length will be used for both dimensions.

#strides

#Integer, tuple of 2 integers, or None. Strides values. Specifies how far the pooling window moves for each pooling step. 
#If None, it will default to pool_size.

#padding

#One of "valid" or "same" (case-insensitive). "valid" means no padding. 
#"same" results in padding evenly to the left/right or up/down of the input such that output has the same height/width dimension as the input.

#data_format
#A string, one of channels_last (default) or channels_first. 
#The ordering of the dimensions in the inputs. channels_last corresponds to inputs with shape (batch, height, width, channels) while channels_first corresponds to inputs with shape (batch, channels, height, width). 
#It defaults to the image_data_format value found in your Keras config file at ~/.keras/keras.json. If you never set it, then it will be "channels_last".

#tf.keras.layers.GlobalAveragePooling2D
#Global average pooling operation for spatial data.


#tf.keras.layers.Dense
#Just regular densely-connected NN layer.



inputs = tf.keras.Input(shape=(120, 120, 3))
x = tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu')(inputs)
x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
x = tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(x)
x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

#tf.keras.Model
#Model groups layers into an object with training and inference features.



model = tf.keras.Model(inputs=inputs, outputs=outputs)

#compile
#Configures the model for training.
#optimizer.  String (name of optimizer) or optimizer instance. 
#loss.   Loss function. May be a string (name of loss function), or a tf.keras.losses.Loss instance
#metrics.    List of metrics to be evaluated by the model during training and testing.

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print(model.summary())

#fit

#Trains the model for a fixed number of epochs (iterations on a dataset).

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=100,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(  #Stop training when a monitored metric has stopped improving.
            monitor='val_loss',  #Quantity to be monitored.
            patience=3,  #Number of epochs with no improvement after which training will be stopped.
            restore_best_weights=True
        )
    ]
)


fig = px.line(
    history.history,
    y=['loss', 'val_loss'],
    labels={'index': "Epoch", 'value': "Loss"},
    title="Training and Validation Loss Over Time"
)

fig.show()

# Results


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
    plt.title("Confusion Matrix")
    plt.show()
    
    print("Classification Report:\n----------------------\n", clr)


evaluate_model(model, test_data)

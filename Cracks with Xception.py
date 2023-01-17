# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 16:20:04 2023

@author: up202111331
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import os
from PIL import Image



curdir_pos = r'C:\Users\up202111331\Desktop\2nd Semester\Machine Learning\Second Report\Concrete Crack Images\Negative' # positive labels
curdir_neg = r'C:\Users\up202111331\Desktop\2nd Semester\Machine Learning\Second Report\Concrete Crack Images\Positive' # negative labels
pos_filepaths = os.listdir(curdir_pos)
neg_filepaths = os.listdir(curdir_neg)
print(pos_filepaths[0:10])


tf.random.set_seed(42)
np.random.seed(42)


full_path_pos = [os.path.join(curdir_pos, img) for img in pos_filepaths]
full_path_neg = [os.path.join(curdir_neg, img) for img in neg_filepaths]
print(full_path_pos[0])
print(full_path_neg[0])


# Splitting in Train, Valid and Test Set


no_crack_files = [[img, 0] for img in full_path_pos] # no cracks
crack_files = [[img, 1] for img in full_path_neg] # cracks
all_files = crack_files + no_crack_files


np.random.shuffle(all_files)

# Compressing Data to TFRecords


BytesList = tf.train.BytesList
FloatList = tf.train.FloatList
Int64List = tf.train.Int64List
Feature = tf.train.Feature
Features = tf.train.Features
Example = tf.train.Example



def create_example(item):
    full_path = item[0]
    crack = item[1]
    image = tf.io.serialize_tensor(np.array(Image.open(full_path)))
    
    example = Example(
        features=Features(
            feature={
                "image": Feature(bytes_list=BytesList(value=[image.numpy()])),
                "crack": Feature(int64_list=Int64List(value=[crack])),
            }
        )
    )
    return example


def create_tf_record(set_, filename):
    with tf.io.TFRecordWriter("%s.tfrecord" %filename) as f:
        for item in set_:
            example = create_example(item)
            f.write(example.SerializeToString())
            
            
            
create_tf_record(all_files[:28001], "train_data")



create_tf_record(all_files[28001:32001], "valid_data")


create_tf_record(all_files[32001:], "test_data")


@tf.function
def preprocess(tfrecord):
    feature_descriptions = {
        "image": tf.io.FixedLenFeature([], tf.string, default_value=""),
        "crack": tf.io.FixedLenFeature([], tf.int64, default_value=-1),
    }
    example = tf.io.parse_single_example(tfrecord, feature_descriptions)
    image = tf.io.parse_tensor(example["image"], out_type=tf.uint8)
    image = tf.reshape(image, shape=[227, 227, 3])
    image = tf.image.resize(image, [224,224]) # reshape to the same dimensions as the training data of our pretrained model
    image = keras.applications.xception.preprocess_input(image)
    return image, example["crack"]

def crack_dataset(filepaths, n_read_threads=5, shuffle_buffer_size=None,
                  n_parse_threads=5, batch_size=32, cache=True):
    dataset = tf.data.TFRecordDataset(filepaths,
                                      num_parallel_reads=n_read_threads)
    if cache:
        dataset = dataset.cache()
    if shuffle_buffer_size:
        dataset = dataset.shuffle(shuffle_buffer_size)
    dataset = dataset.map(preprocess, num_parallel_calls=n_parse_threads)
    dataset = dataset.batch(batch_size)
    return dataset.prefetch(1)




train_set = crack_dataset("./train_data.tfrecord")
valid_set = crack_dataset("./valid_data.tfrecord")
test_set = crack_dataset("./test_data.tfrecord")



for image, crack in train_set.take(3):
    plt.imshow(image[0])
    plt.axis('off')
    plt.show()
    print("Label:", crack[0].numpy())
    
    
    
    
    
# Training Model

base_model = keras.applications.xception.Xception(weights="imagenet", include_top=False)
avg = keras.layers.GlobalAveragePooling2D()(base_model.output)
output = keras.layers.Dense(1, activation="sigmoid")(avg)
model = keras.models.Model(inputs=base_model.input, outputs=output)




for layer in base_model.layers: # freeze the weights of the base model
    layer.trainable = False
    
optimizer = keras.optimizers.Nadam(learning_rate=1e-4)
model.compile(loss="mean_absolute_error", optimizer=optimizer,
              metrics=["accuracy"]
              )
history = model.fit(train_set,
                    validation_data=valid_set,
                    epochs=5)



for layer in base_model.layers: # unfreeze the weights and continue training
    layer.trainable = True

checkpoint_cb = keras.callbacks.ModelCheckpoint('my_model.h5', save_best_only=True)
early_stopping_cb = keras.callbacks.EarlyStopping(patience=5,
                                                  restore_best_weights=True)
history = model.fit(train_set,
                    validation_data=valid_set,
                    epochs=10,
                    callbacks=[checkpoint_cb, early_stopping_cb])


model = keras.models.load_model('my_model.h5')


mae = model.evaluate(test_set)





fig = px.line(
    history.history,
    y=['loss', 'val_loss'],
    labels={'index': "Epoch", 'value': "Loss"},
    title="Training and Validation Loss Over Time"
)

fig.show()
    







# EVALUATIONNNNNNNNNNNNNNNN







import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from pathlib import Path
from sklearn.model_selection import train_test_split

import tensorflow as tf

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

from sklearn.model_selection import train_test_split

train_df, test_df = train_test_split(
    all_df.sample(6000, random_state=1),
    train_size=0.7,
    shuffle=True,
    random_state=1
)

train_gen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

test_data = train_gen.flow_from_dataframe(
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









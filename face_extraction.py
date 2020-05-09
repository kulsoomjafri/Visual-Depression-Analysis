import os
import tensorflow_hub as hub
import tensorflow as tf
from sklearn.model_selection import train_test_split
#import split_folders
import pandas as pd
import numpy as np
import pathlib
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau
# from keras.preprocessing.image import load_img, img_to_array
import time
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix
import itertools
AUTOTUNE = tf.data.experimental.AUTOTUNE
import PIL.Image
import urllib
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import KFold
# load and evaluate a saved model
from numpy import loadtxt
from keras.models import load_model
IMG_HEIGHT = 48
IMG_WIDTH = 48
batch_size=3
def prepare_for_training(ds, cache=True, shuffle_buffer_size=1000):
  # This is a small dataset, only load it once, and keep it in memory.
  # use `.cache(filename)` to cache preprocessing work for datasets that don't
  # fit in memory.
  if cache:
    if isinstance(cache, str):
      ds = ds.cache(cache)
    else:
      ds = ds.cache()

  ds = ds.shuffle(buffer_size=shuffle_buffer_size)

  # Repeat forever
  ds = ds.repeat()

  ds = ds.batch(50)

  # `prefetch` lets the dataset fetch batches in the background while the model
  # is training.
  ds = ds.prefetch(buffer_size=AUTOTUNE)

  return ds  

def decode_img(img):
  # convert the compressed string to a 3D uint8 tensor
  img = tf.image.decode_jpeg(img, channels=1)
  # Use `convert_image_dtype` to convert to floats in the [0,1] range.
  img = tf.image.convert_image_dtype(img, tf.float32)
  # resize the image to the desired size.
  return tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT])

def get_label(file_path):
  # convert the path to a list of path components
  parts = tf.strings.split(file_path, os.path.sep)
  # The second to last is the class-directory
  return parts[-2] == CLASS_NAMES
  
def process_path(file_path):
  label = get_label(file_path)
  # load the raw data from the file as a string
  img = tf.io.read_file(file_path)
  img = decode_img(img)
  return img, label

def build_model(loaded):
  x = tf.keras.layers.Input(shape=(48, 48, 1), name='inputs')
  # Wrap what's loaded to a KerasLayer
  keras_layer = hub.KerasLayer(loaded, trainable=True)(x)
  model = tf.keras.Model(x, keras_layer)
  return model


cwd_dir = os.getcwd()#get the current directory

import_dir = os.path.join(cwd_dir, 'babby\\')
model_dir = os.path.join(cwd_dir, 'saved_model\\')
# load model
# model = load_model('model.h5')
# another_strategy = tf.distribute.MirroredStrategy()
# with another_strategy.scope():
# loaded = tf.saved_model.load(export_dir)
loaded =  tf.keras.models.load_model(model_dir)


# model = build_model(loaded)
# summarize model.
loaded.summary()
# load dataset

model=loaded


#extracting class names from folders and ignoring other files
data_dir = pathlib.Path(import_dir)  #creating glob
CLASS_NAMES = ['depressed', 'happy' ,'neutral']
print ("the classes are", CLASS_NAMES)
cpt=0

sample_test_lib = pathlib.Path(import_dir)
print(sample_test_lib)
list_ds= tf.data.Dataset.list_files(str(sample_test_lib/'*'))
labeled_ds = list_ds.map(process_path, num_parallel_calls=100)
test_dataset = labeled_ds.skip(10)
test_dataset = test_dataset.take(20)
test_data_gen = prepare_for_training(test_dataset) 
batch_size=50
tests_batch,_ = next(iter(test_data_gen))
predictor=tests_batch
predictions = model.predict(predictor)
print(predictions)
b = tf.math.argmax(input = predictions.transpose())
c = tf.keras.backend.eval(b)
print(c)
for n in range(25):
    ax = plt.subplot(5,5,n+1)
    img2=np.squeeze(predictor[n])
    plt.imshow(img2, cmap='gray')
    plt.title(CLASS_NAMES[c[n]])
    plt.axis('off')
    
    
plt.show()


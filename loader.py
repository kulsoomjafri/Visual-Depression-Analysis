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

def build_model(loaded):
  x = tf.keras.layers.Input(shape=(48, 48, 1), name='inputs')
  # Wrap what's loaded to a KerasLayer
  keras_layer = hub.KerasLayer(loaded, trainable=True)(x)
  model = tf.keras.Model(x, keras_layer)
  return model


cwd_dir = os.getcwd()#get the current directory

export_dir = os.path.join(cwd_dir, 'saved_model\\')
# load model
# model = load_model('model.h5')
# another_strategy = tf.distribute.MirroredStrategy()
# with another_strategy.scope():
# loaded = tf.saved_model.load(export_dir)
loaded =  tf.keras.models.load_model(export_dir)


# model = build_model(loaded)
# summarize model.
loaded.summary()
# load dataset

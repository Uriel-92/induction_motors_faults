# Script for data augmentation
import tensorflow as tf
from tensorflow.keras import layers
import pandas as pd
import cv2
import numpy as np


data_augmentation = tf.keras.Sequential([
  layers.RandomFlip("horizontal_and_vertical"),
  layers.RandomRotation(0.2),
])


# Read and plot the data
df = pd.read_csv("images.csv")
#df["labels"].value_counts().plot(kind='bar')

class_max = df["labels"].value_counts().idxmax()
max_value = df["labels"].value_counts().max()

for class_name in list(df["labels"].unique()):
    if(class_name != class_max):
        if(sum(df["labels"] == class_name) < max_value):
            mask = (df["labels"] == class_name)
            path = df["path"].loc[mask].iloc[0]
            img = cv2.imread(path)
            for i in range(max_value - sum(df["labels"]== class_name)):
                imgs = data_augmentation(img)
                imgs = np.clip(imgs.numpy(), 0, 255)
                imgs = np.uint8(imgs)
                cv2.imwrite("faults/"+class_name+str(i)+".bmp", imgs)
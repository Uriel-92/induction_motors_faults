# Feature extractor
import os
os.environ["KERAS_BACKEND"] = "torch"
import keras_core as keras
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import cv2

X = []
df = pd.read_csv("images.csv")
labels = {value:indx for indx, value in enumerate(df["labels"].unique())}
y = df["labels"].map(labels).values

for path in df["path"]:
    img = cv2.imread(path)
    img = cv2.resize(img, (250, 250), cv2.INTER_AREA)
    img = img/255
    X.append(img)

X = np.array(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                    random_state=42, stratify=y)

model = keras.saving.load_model("IM-faults.keras")

base_model = keras.Model(model.input, model.get_layer("dense").output)

features_train = base_model.predict(X_train)
features_test = base_model.predict(X_test)
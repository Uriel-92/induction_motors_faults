# Script to generate CNN model
import os
os.environ["KERAS_BACKEND"] = "torch"
import keras_core as keras
from keras_core import layers

import pandas as pd
import numpy as np
import cv2

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def get_model(input_shape = (250, 250, 3)):
    input_ = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(input_)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(256, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    
    # Flatten layer
    x = layers.Flatten()(x)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    output = layers.Dense(11, activation="softmax")(x)
    
    model = keras.Model(input_, output, name="IM-Faults")
    
    keras.utils.plot_model(model, to_file="IM-Faults.png", show_shapes=True)
    
    return model


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

model = get_model()
model.compile(optimizer=keras.optimizers.Adam(), 
              loss=keras.losses.SparseCategoricalCrossentropy(), 
              metrics=['accuracy'])

best_model = keras.callbacks.ModelCheckpoint(
        "IM-faults.keras",
        monitor="val_loss",
        save_best_only=True
    )

history = model.fit(X_train, y_train, batch_size=16, epochs=20, 
                    validation_split=0.1, callbacks=[best_model])

history = pd.DataFrame(history.history)
history.plot(kind="line")

y_pred = np.argmax(model.predict(X_test), axis=1)
result = accuracy_score(y_test, y_pred)
print("Accuracy_score: {:.2f}".format(result))
# Script for dataaugmentation
import os
os.environ["KERAS_BACKEND"] = "torch"
import numpy as np
import keras_core as keras
import pandas as pd
import cv2


def get_model(input_shape=(150, 150, 3)):
    input_ = keras.layers.Input(shape=input_shape)
    x = keras.layers.Conv2D(32, kernel_size=3, activation='relu')(input_)
    x = keras.layers.MaxPooling2D()(x)
    
    x = keras.layers.Conv2D(32, kernel_size=3, activation='relu')(x)
    x = keras.layers.MaxPooling2D()(x)
    
    x = keras.layers.Conv2D(64, kernel_size=3, activation='relu')(x)
    x = keras.layers.MaxPooling2D()(x)
    
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(64, activation='relu')(x)
    x = keras.layers.Dropout(0.5)(x)
    output = keras.layers.Dense(11, activation="softmax")(x)
    
    model = keras.Model(input_, output)
    print(model.summary())
    return model

model = get_model()
model.compile(loss=keras.losses.SparseCategoricalCrossentropy(),
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

df = pd.read_csv("images.csv")
images = []
for path in df["path"]:
    img = cv2.imread(path)
    img = cv2.resize(img, (150, 150))
    img = img/255
    images.append(img)

columns = {value:indx for indx, value in enumerate(df["labels"].unique())}
y = df["labels"].map(columns).values
X = np.array(images)

best_model = keras.callbacks.ModelCheckpoint("best_model_torch.keras",
                                             monitor='val_loss',
                                             save_best_only=True)

history = model.fit(X, y, epochs=20, batch_size=8, validation_split=0.1,
                    callbacks=[best_model])

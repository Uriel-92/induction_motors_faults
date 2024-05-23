# Script for image analysis

import pandas as pd
import os
import numpy as np

PATH = "faults/"
folders = os.listdir(PATH)

path = []
labels = []

for folder in folders:
    images_path = os.path.join(PATH, folder)
    images = os.listdir(images_path)
    for image in images:
        path.append(os.path.join(images_path, image))
        labels.append(folder)

df = pd.DataFrame(np.array(path), columns=["path"])
df["labels"] = np.array(labels)
df.to_csv("images.csv", index=False)
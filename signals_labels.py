# Script for generate signals and labels in a csv
import numpy as np
import cv2
import pandas as pd
import os
from entropy_cues import entropyCues

# Path to the faults
PATH = "faults2/"
signals = []
labels = []

# Walk throught the folders
folders = os.listdir(PATH)
for folder in folders:
    images_folders = os.path.join(PATH, folder)
    images = os.listdir(images_folders)
    for image in images:
        route = os.path.join(images_folders, image)
        img = cv2.imread(route, 0)
        entropy = entropyCues(img)
        signals.append(entropy.entropy())
        labels.append(folder)

sig = pd.DataFrame(np.array(signals))
target = pd.DataFrame(labels, columns=["Labels"])

sig.to_csv("entropy_signlas2.csv", index=False, header=False)
target.to_csv("labels2.csv", index=False)
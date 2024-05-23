# Script for test different ensemble models
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
import pickle as pk
from matplotlib import pyplot as plt
import seaborn as sns

def plot_confusion_matrix(cnf_matrix, columns, title):
    plt.figure(figsize=(12,8))
    cnf_matrix = cnf_matrix/np.sum(cnf_matrix)
    g = sns.heatmap(cnf_matrix, annot=True, cbar=False, fmt=".2%", 
                    cmap="Blues", annot_kws={"size": 12})
    g.set_yticklabels(columns, rotation=0)
    g.set_xticklabels(columns)
    plt.xticks(fontsize=14, rotation=45)
    plt.yticks(fontsize=14)
    plt.title(title, weight="bold", fontsize=16)
    plt.show()


xgb_params = {"max_depth": np.arange(1, 6, 1), "tree_method": ("exact", 
            "approx", "hist"), "n_estimators": [10, 40, 80, 100, 200, 300]}
rf_params = {"n_estimators": [10, 40, 80, 100, 200, 300], 
             "max_depth": np.arange(1, 10, 1)}

models = [XGBClassifier(), RandomForestClassifier()]
titles = ["Convolutional Neural Network", "Random_forest_classifier"]
params = [xgb_params, rf_params]

df = pd.read_csv("images.csv")
columns = ["Rotor", "Healthy", "10%-s 3-p",
           "50%-s 1-p", "50%-s 2-p", "30%-s 2-p",
           "10%-s 2-p", "10%-s 1-p", "30%-s 3-p",
           "30%-s 1-p", "Coling"]

X_train = np.load("features_train.npy")
y_train = np.load("label_features.npy")
X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")

for model, param, title in zip(models, params, titles):
    clf = GridSearchCV(model, param, cv=5, n_jobs=3)
    clf.fit(X_train, y_train)
    best_model = clf.best_estimator_
    y_pred = best_model.predict(X_test)
    cnf_matrix = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cnf_matrix, columns, title)
    print(accuracy_score(y_test, y_pred))
    with open("models/" + title + ".pkl", "wb") as f:
        pk.dump(best_model, f)

# Models
lr_model = pk.load(open("models/Logistic_regression.pkl", "rb"))
svm_model = pk.load(open("models/Support_vector_machine.pkl", "rb"))
knn_model = pk.load(open("models/KNN.pkl", "rb"))
dt_model = pk.load(open("models/Decision_tree.pkl", "rb"))

predictions = [lr_model.predict(X_test), svm_model.predict(X_test),
               knn_model.predict(X_test), dt_model.predict(X_test)]
predictions = np.array(predictions)
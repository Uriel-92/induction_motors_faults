# Script for generate machine learning models
import numpy as np
import pickle as pk
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


def plot_confusion_matrix(cnf_matrix, columns, title):
    plt.figure(figsize=(12,8))
    cnf_matrix = cnf_matrix/np.sum(cnf_matrix)
    g = sns.heatmap(cnf_matrix, annot=True, cbar=False, fmt=".2%", 
                    cmap="Blues", annot_kws={"size": 12})
    g.set_yticklabels(columns, rotation=0)
    g.set_xticklabels(columns)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.title(title, weight="bold", fontsize=16)
    plt.show()

# Control variables
titles = ["Logistic regression", "Support vector machine", 
          "Decision tree", "KNN"]
model_names = ["lr", "svm", "dt", "knn"]

# Read the data
df = pd.read_csv("median2.csv")
columns = list(df["Labels"].unique())
values = {name:indx for indx, name in enumerate(columns, 0)}
df["Median2"] = df["Median"].map(lambda s: s**2)
y = df["Labels"].map(values).values
X = df[["Median", "Median2"]].values

# Standarize the data and split in train and test sets
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=42)
# Generate the params to search
log_param = {"C": np.arange(0.1, 10, 0.2, dtype=np.float16)}
svc_param = {"C": np.arange(0.1, 10, 0.2, dtype=np.float16),
             "kernel": ["linear", "poly", "rbf"]}
tree_param = {"max_depth": [None, 2, 5, 7, 10]}
knn_param = {"n_neighbors": np.arange(1, 15, 1, dtype=np.int16)}

# Create the models
params = [log_param, svc_param, tree_param, knn_param]
models = [LogisticRegression(), SVC(), DecisionTreeClassifier(), 
          KNeighborsClassifier()]

# Train the models and save it
for param, model, title, name in zip(params, models, titles, 
                                     model_names):
    clf = GridSearchCV(model, param, cv=3)
    clf.fit(X_train, y_train)
    print(clf.best_params_)
    best_model = clf.best_estimator_
    y_pred = best_model.predict(X_test)
    cnf_matrix = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cnf_matrix, columns, title)
    print(accuracy_score(y_test, y_pred))
    # Save the best model
    #pk.dump(best_model, "/models/"+name+"_model.pkl")
    
    
    



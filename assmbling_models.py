# Script for assembling learning
import numpy as np
import pickle as pk
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV
from matplotlib import pyplot as plt
import seaborn as sns
sns.set_theme()

# Function to plot the confusion matrix
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

# Parameters definition
lr_params = {"C": np.arange(0.1, 5, 0.1), "max_iter": [1000]}
svc_params = {"C": np.arange(0.1, 5, 0.1), "kernel": ("linear", "poly", "rbf")}
knn_params = {"n_neighbors": np.arange(1, 10, 1, dtype=np.int8)}
dt_params = {"max_depth": [None, 3, 5, 7, 9, 11, 13, 15]}
sgd_params = {"loss": ["hinge", "log_loss", "modified_huber", "squared_hinge", 
                       "perceptron", "squared_error", "huber", "epsilon_insensitive",
                       "squared_epsilon_insensitive"]}

models = [LogisticRegression(), SVC(), KNeighborsClassifier(), 
          DecisionTreeClassifier(), SGDClassifier()]
titles =["Logistic_regression", "Support_vector_machine", "KNN", "Decision_tree",
         "Stochastic_gradient_descent"]
params = [lr_params, svc_params, knn_params, dt_params, sgd_params]

df = pd.read_csv("images.csv")
columns = ["Rotor", "Healthy", "10%-stator 3-phase",
           "50%-stator 1-phase", "50%-stator 2-phase", "30%-stator 2-phase",
           "10%-stator 2-phase", "10%-stator 1-phase", "30%-stator 3-phase",
           "30%-stator 1-phase", "Cooling"]


# Train dataset
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
    #with open("models/" + title + ".pkl", "wb") as f:
     #   pk.dump(best_model, f)
# Script for dimensional reduction
from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.tree import DecisionTreeClassifier

# Load variables
X_train = np.load("features_train.npy")
y_train = np.load("label_features.npy")
X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")

# Models
k = 100
kmeans = KMeans(n_clusters=k)
X_train_dist = kmeans.fit_transform(X_train)
representative_data = np.argmin(X_train_dist, axis=0)
X_representative = X_train[representative_data]
y_representative = y_train[representative_data]

y_train_propagated = np.empty(len(X_train), dtype=np.int32)
for i in range(k):
    y_train_propagated[kmeans.labels_==i] = y_representative[i]


# Propagated
percentile_closest = 20
X_clusters_dist = X_train_dist[np.arange(len(X_train)), kmeans.labels_]
for i in range(k):
    in_cluster = (kmeans.labels_ == i)
    cluster_dist = X_clusters_dist[in_cluster]
    cutoff_dist = np.percentile(cluster_dist, percentile_closest)
    above_cutoff = (X_clusters_dist > cutoff_dist)
    X_clusters_dist[in_cluster & above_cutoff] = -1

partially_propagated = (X_clusters_dist != -1)
X_train_partially_propagated = X_train[partially_propagated]
y_train_partially_propagated = y_train_propagated[partially_propagated]

dt = DecisionTreeClassifier(max_depth=25)
dt.fit(X_train_partially_propagated, y_train_partially_propagated)
print(dt.score(X_test, y_test))
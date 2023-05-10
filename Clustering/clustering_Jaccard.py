import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.cluster import KMeans

# Load data from input file

X = np.loadtxt(
    'C:\GIT_KRSuchan\Study_AI\Clustering\data_clustering.txt', delimiter=',')

# Initialize variables
scores = []
values = np.arange(2, 10)

# Iterate through the defined range
for num_clusters in values:
    # Train the KMeans clustering model
    kmeans = KMeans(init='k-means++', n_clusters=num_clusters, n_init=10)
    kmeans.fit(X)
    score = metrics.silhouette_score(X, kmeans.labels_,
                                     metric='euclidean', sample_size=len(X))
    print("\nNumber of clusters =", num_clusters)
    print("Silhouette score =", score)

    scores.append(score)

# Extract best score and optimal number of clusters
num_clusters = np.argmax(scores) + values[0]
print('\nOptimal number of clusters =', num_clusters)

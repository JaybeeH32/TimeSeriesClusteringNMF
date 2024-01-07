from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.mixture import GaussianMixture
import numpy as np


def kmeans(A, d, direct=True):
    model = KMeans(n_clusters=d, n_init='auto')
    model.fit(A)
    predicted_clusters = model.predict(A)
    if direct:
        return predicted_clusters
    else:
        # artefact to align all clustering methods on nmf 
        n = len(A)
        arr = np.zeros(shape=(n, d))
        for i in range(n):
            arr[i, predicted_clusters[i]] = 1.0
        return arr, 0, 0

def spectral_clustering(A, d, direct=True):
    model = SpectralClustering(n_clusters=d)
    predicted_clusters = model.fit_predict(A)
    if direct:
        return predicted_clusters
    else:
        # artefact to align all clustering methods on nmf 
        n = len(A)
        arr = np.zeros(shape=(n, d))
        for i in range(n):
            arr[i, predicted_clusters[i]] = 1.0
        return arr, 0, 0

def gaussian_mixture(A, d, direct=True):
    model = GaussianMixture(n_components=d)
    predicted_clusters = model.fit_predict(A)
    if direct:
        return predicted_clusters
    else:
        # artefact to align all clustering methods on nmf 
        n = len(A)
        arr = np.zeros(shape=(n, d))
        for i in range(n):
            arr[i, predicted_clusters[i]] = 1.0
        return arr, 0, 0
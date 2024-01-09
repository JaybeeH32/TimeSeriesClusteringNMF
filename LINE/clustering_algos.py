from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.mixture import GaussianMixture
import numpy as np
from libraries.randomwalk import Node2Vec
import networkx as nx


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
    
def node2vec_clustering(A, d, n_dim=128, n=50, length=25, p=0.5, q=2, direct=True):
    G = nx.from_numpy_array(A)
    node2vec = Node2Vec(G, n_dim, n, length, p, q)
    embeddings = node2vec.Skipgram_embeddings()
    predicted_clusters = kmeans(embeddings, d, direct=direct)
    return predicted_clusters

def line_clustering(dataset, eps, d):
    emb_path = "output" + f'\{dataset};eps={eps}.npz.npy' 
    embeddings = np.load(emb_path)
    predicted_clusters = kmeans(embeddings, d)
    return predicted_clusters
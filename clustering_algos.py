from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.mixture import GaussianMixture


def kmeans(A, d):
    model = KMeans(n_clusters=d)
    model.fit(A)
    predicted_clusters = model.predict(A)
    return predicted_clusters

def spectral_clustering(A, d):
    model = SpectralClustering(n_clusters=d)
    predicted_clusters = model.fit_predict(A)
    return predicted_clusters

def gaussian_mixture(A, d):
    model = GaussianMixture(n_components=d)
    predicted_clusters = model.fit_predict(A)
    return predicted_clusters
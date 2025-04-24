
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_distances

def cluster_embeddings(embeddings, eps=0.3):
    if len(embeddings) == 0:
        return 0
    db = DBSCAN(metric="cosine", eps=eps, min_samples=1).fit(embeddings)
    return len(set(db.labels_))

def cluster_embeddings_kmeans(X: np.ndarray, max_k: int = 10):
    """
    使用 KMeans 自动选择最佳聚类数 k
    返回：labels, best_k
    """
    best_k = 2
    best_score = -1
    best_labels = None

    for k in range(2, min(max_k, len(X))):
        kmeans = KMeans(n_clusters=k, random_state=0).fit(X)
        score = silhouette_score(X, kmeans.labels_)
        if score > best_score:
            best_k = k
            best_score = score
            best_labels = kmeans.labels_

    return best_labels, best_k


def cluster_embeddings_agglomerative(X: np.ndarray, threshold: float = 0.75):
    """
    使用层次聚类（Agglomerative），需设定距离阈值
    返回：labels
    """
    clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=threshold)
    return clustering.fit_predict(X)


def cluster_embeddings_dbscan(X: np.ndarray, eps: float = 0.2):
    """
    使用 DBSCAN 聚类（自动决定类数，支持噪声）
    返回：labels
    """
    db = DBSCAN(metric="cosine", eps=eps, min_samples=2).fit(X)
    return db.labels_


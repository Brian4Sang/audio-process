# utils/cluster_embeddings.py
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_distances

def cluster_embeddings(embeddings, eps=0.3):
    if len(embeddings) == 0:
        return 0
    db = DBSCAN(metric="cosine", eps=eps, min_samples=1).fit(embeddings)
    return len(set(db.labels_))

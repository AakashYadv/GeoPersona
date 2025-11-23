
import numpy as np
import hdbscan
from sklearn.decomposition import PCA

class PersonaClustering:
    def cluster(self, embeddings):
        X = np.array(embeddings)
        if len(X) < 2:
            return [0] * len(X)

        pca = PCA(n_components=min(5, X.shape[1]))
        Xr = pca.fit_transform(X)
        labels = hdbscan.HDBSCAN(min_cluster_size=3).fit_predict(Xr)
        return labels


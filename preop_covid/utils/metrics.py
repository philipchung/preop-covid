from typing import Union

import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean, pdist


def xie_beni_index(X: np.array, labels: Union[np.array, pd.Series, pd.DataFrame]) -> float:
    """
    Compute Xie-Beni Index.  A measure of separation vs. compactness.
    Smaller Xie-Beni Index indicates better cluster compactness.

    XB = sum(|x_c - mean_c|^2) / (n * min_dist(c)^2)
    where
    x_c is a data point in a cluster c
    mean_c is the cluster center of cluster c
    n is the total number of data points
    min_dist(c) is the minimum distance between all cluster centers

    Args:
        X: array-like of shape (n_samples, n_features).
        labels: array-like of shape (n_samples,) or (n_samples, n_labels).
            If shape is (n_samples,), then we assume each sample is only labeled with a
            single cluster.  Each value in the vector (n_samples,) is the cluster id label
            that corresponds to the sample in X.
            If shape is (n_samples, n_labels), then we assume each sample can be labeled
            with multiple clusters.  This is a boolean label matrix in which each row
            (1, n_labels) is a boolean vector that denotes whether the sample belongs
            to each of the n_labels clusters.  Cluster identities are derived from column
            names if a DataFrame is provided, otherwise, cluster identities will be
            named after the position of the column in the matrix.
    Returns:
        Xie-Beni Index.  Smaller indicates better cluster compactness & separation.

    References:
    1. Xie X and Beni G. "A validity measure for fuzzy clustering".
    IEEE Transactions on Pattern Analysis and Machine Intelligence,
    (1991) 13:8, pp. 841-847
    2. Desgraupes B. "Clustering Indices".  (2017)
    3. Halkidi M, Batistakis Y and Vazirgiannis M. “On Clustering Validation
    Techniques.” Journal of Intelligent Information Systems 17 (2004): 107-145.
    """
    _X, _labels = X.copy(), labels.copy()
    if isinstance(_labels, pd.Series) or isinstance(_labels, pd.DataFrame):
        _col_names = _labels.columns.tolist()
        _labels = _labels.to_numpy()
    else:
        _col_names = None

    if len(_labels.shape) == 1:
        exclusive_clusters = True
        num_rows = _labels.shape[0]
        # Exclusive cluster labels. Value of each element is label identity.
        cluster_ids = list(set(labels))
    else:
        exclusive_clusters = False
        num_rows, num_cols = _labels.shape
        # Non-exclusive cluster labels. Use jth column position as label identity.
        # We can map these back to _col_names later.
        cluster_ids = list(range(num_cols))

    # Compute Xie-Beni Index
    sum_mean_squared_norm = 0
    cluster_centers = []
    for cluster_id in cluster_ids:
        # Get data points in cluster
        if exclusive_clusters:
            indices = np.where(labels == cluster_id)[0]
            cluster_X = X[indices, :]
        else:
            indices = _labels[:, cluster_id]
            cluster_X = X[indices, :]
        # Cluster center
        cluster_center = np.mean(cluster_X, axis=0)
        cluster_centers += [cluster_center]
        # Mean squared euclidean distances of points in cluster w.r.t cluster center
        cluster_mean_squared_norms = [euclidean(x, cluster_center) ** 2 for x in cluster_X]
        sum_mean_squared_norm += np.sum(cluster_mean_squared_norms)
    # Min distance between cluster centers
    centers = np.stack(cluster_centers)
    min_dist = min(pdist(centers))
    xb_index = sum_mean_squared_norm / (num_rows * (min_dist**2))
    return xb_index

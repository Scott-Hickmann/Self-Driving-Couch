import numpy as np
import random
from scipy.spatial.distance import squareform, pdist, cdist
from skimage.util import img_as_float
from skimage import color

def kmeans_fast(features, k, num_iters=100):

    N, D = features.shape

    assert N >= k, 'Number of clusters cannot be greater than number of points'

    idxs = np.random.choice(N, size=k, replace=False)
    centers = features[idxs]
    assignments = np.zeros(N, dtype=np.uint32)

    for n in range(num_iters):
        distances = cdist(features, centers)
        new_assignments = np.argmin(distances, axis=1)
        if (np.array_equal(new_assignments, assignments)):
            break
        assignments = new_assignments
        for i in range(centers.shape[0]):
            cluster = features[assignments==i]
            centers[i] = np.mean(cluster, axis=0)    
    return assignments

def hierarchical_clustering(features, k):

    N, D = features.shape

    assert N >= k, 'Number of clusters cannot be greater than number of points'

    assignments = np.arange(N, dtype=np.uint32)
    centers = np.copy(features)
    n_clusters = N
    
    while n_clusters > k:
        pairs = squareform(pdist(centers))
        pairs += np.max(pairs)*np.eye(n_clusters)
        min_indices = np.where(pairs==np.min(pairs))[0]
        first = min_indices[0]
        second = min_indices[1]
        len1 = np.count_nonzero(assignments==first)
        len2 = np.count_nonzero(assignments==second)
        sum1 = np.sum(features[assignments==first], axis=0)
        sum2 = np.sum(features[assignments==second], axis=0)
        new_center = (sum1 + sum2) / (len1 + len2)
        centers = np.delete(centers, [first, second], axis=0)
        centers = np.vstack((centers, new_center))
        middle = [i for i in range(N) if (assignments[i] > first) & (assignments[i] < second)]
        right = [i for i in range(N) if (assignments[i] > second)]
        reassign = [i for i in range(N) if (assignments[i]==first) | (assignments[i]==second)]
        assignments[middle] -= 1
        assignments[right] -= 2
        assignments[reassign] = centers.shape[0] - 1
        n_clusters -= 1    
    return assignments

def color_features(img):
    
    H, W, C = img.shape
    img = img_as_float(img)
    features = np.zeros((H*W, C))

    features = img.reshape(H*W, C)
    return features

def color_position_features(img):
    
    H, W, C = img.shape
    color = img_as_float(img)
    features = np.zeros((H*W, C+2))

    positions = np.mgrid[0:H, 0:W].T
    positions = np.transpose(positions, (1, 0, 2))
    combined = np.dstack((color, positions))
    features = combined.reshape(H*W, C+2)

    mean = np.mean(features, axis=0)
    mean = np.tile(mean, (H*W, 1))
    std = np.std(features, axis=0)
    std = np.tile(std, (H*W, 1))
    features -= mean
    features /= std
    
    return features

def compute_accuracy(mask_gt, mask):

    TP_mask = (mask_gt==1) & (mask==1)
    TP = np.count_nonzero(TP_mask==1)
    TN_mask = (mask_gt==0) & (mask==0)
    TN = np.count_nonzero(TN_mask==1)
    accuracy = (TP + TN) / (mask.shape[0]*mask.shape[1])
    return accuracy

def evaluate_segmentation(mask_gt, segments):
    
    num_segments = np.max(segments) + 1
    best_accuracy = 0

    for i in range(num_segments):
        mask = (segments == i).astype(int)
        accuracy = compute_accuracy(mask_gt, mask)
        best_accuracy = max(accuracy, best_accuracy)
    return best_accuracy

import numpy as np


# See Algorithm 1 on page 6 in [YS]
def knn(training_points, training_labels, test_point, test_label):
    # 2. Calculate pairwise distances (d_i = d(x_i, x)) between test point and training points using euclidean distance.
    distances = np.sum(np.square((training_points - test_point)), axis=1)
    # using np.linalg.norm
    # distances = np.linalg.norm(training_points - test_point, axis=1)

    # 3. Sort distances (d_i's) in ascending order and get corresponding indices
    distances_sorted_indices = np.argsort(distances)

    # 4. Calculate the summed up sign of the labels of the K nearest neighbors,
    # by summing the labels (majority vote) of the K nearest neighbors and taking the sign of the sum.
    ys = np.sign(np.cumsum(training_labels[distances_sorted_indices]))

    # If voting tie, only look at the K-1 neighbors
    for i in range(len(ys)):
        if ys[i] == 0:
            ys[i] = ys[i - 1]

    errors = np.where(ys != test_label, 1, 0)

    return errors

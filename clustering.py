import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

kmtest_df = pd.read_csv('kmtest.csv', sep='\s+', header=None)
iris_df = pd.read_csv('iris.csv',header=None)

kmtest_data = kmtest_df.values.astype(float)
iris_data = iris_df.iloc[:,:4].values


def kmeans(data, K, max_iters=100):
    n_samples, n_features = data.shape


    random_indices = np.random.choice(n_samples, K, replace=False)
    centroids = data[random_indices]


    for _ in range(max_iters):

        assignments = np.zeros(n_samples, dtype=int)
        for i in range(n_samples):
            current_point = data[i]
            distances = [np.linalg.norm(current_point - centroid) for centroid in centroids]
            closest_index = np.argmin(distances)
            assignments[i] = closest_index

        old_centroids = np.copy(centroids)


        for k in range(K):
            points_in_cluster = data[assignments == k]
            if len(points_in_cluster) > 0:
                centroids[k] = np.mean(points_in_cluster, axis=0)


        if np.array_equal(old_centroids, centroids):
            print(f"Converged after {_ + 1} iterations.")
            break

    return centroids, assignments


K_value = 3
final_centroids, final_assignments = kmeans(kmtest_data, K_value)

print("\nFinal Centroids:")
print(final_centroids)

print("\nCluster assignment for each data point:")
print(final_assignments)











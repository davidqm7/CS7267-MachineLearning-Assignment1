import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#A. What to do: 1.Part a

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


#A. What to do: 1.Part b
mean = np.mean(kmtest_data, axis=0)
std = np.std(kmtest_data, axis=0)

kmtest_normalized = (kmtest_data - mean) / std

k_values_to_test = [2, 3, 4, 5]

colors = ['red', 'blue', 'green', 'purple', 'orange']

for K in k_values_to_test:
    centroids, assignments = kmeans(kmtest_normalized, K)

    plt.figure(figsize=(8, 6))

    for k in range(K):
        cluster_points = kmtest_normalized[assignments == k]

        plt.scatter(cluster_points[:, 0],
                    cluster_points[:, 1],
                    c=colors[k],
                    label=f'Cluster {k + 1}')

    plt.scatter(centroids[:, 0],
                centroids[:, 1],
                c='black',
                marker='X',
                s=200,
                label='Centroids')

    plt.title(f'K-Means Clustering (K={K}) on Normalized Data')
    plt.xlabel('Feature 1 (Normalized)')
    plt.ylabel('Feature 2 (Normalized)')
    plt.legend()
    plt.grid(True)

    plt.show()


#A. What to do: 2.Part a

K = 3
num_runs = 5
iris_data_to_cluster = iris_data

results = []

for run in range(num_runs):
    print(f"--- Running K-Means: Run #{run + 1} ---")

    centroids, assignments = kmeans(iris_data_to_cluster, K)

    current_sse = 0

    for k in range(K):
        points_in_cluster = iris_data_to_cluster[assignments == k]

        centroid = centroids[k]

        current_sse += np.sum((points_in_cluster - centroid) ** 2)

    print(f"SSE for this run: {current_sse:.4f}")

    results.append({
        'sse': current_sse,
        'centroids': centroids,
        'assignments': assignments
    })

    sorted_results = sorted(results, key=lambda x: x['sse'])

    best_result = sorted_results[0]

    worst_result = sorted_results[-1]

    print("\n--- Comparison ---")
    print(f"Best SSE: {best_result['sse']:.4f}")
    print(f"Worst SSE: {worst_result['sse']:.4f}")

#A. What to do: 2.Part b






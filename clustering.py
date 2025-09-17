# Name: David Quintanilla
# Number: 000960982
# Project 1

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



def kmeans(data, K, max_iters=100):
   
    n_samples, n_features = data.shape
    random_indices = np.random.choice(n_samples, K, replace=False)
    centroids = data[random_indices]

    for i in range(max_iters):
        
        assignments = np.zeros(n_samples, dtype=int)
        for j in range(n_samples):
            distances = [np.linalg.norm(data[j] - centroid) for centroid in centroids]
            assignments[j] = np.argmin(distances)

       
        old_centroids = np.copy(centroids)
        for k in range(K):
            points_in_cluster = data[assignments == k]
            if len(points_in_cluster) > 0:
                centroids[k] = np.mean(points_in_cluster, axis=0)

        
        if np.array_equal(old_centroids, centroids):
            print(f"Converged after {i + 1} iterations.")
            break
            
    return centroids, assignments


def plot_iris_clusters(data, assignments, centroids, title):
    
    K = len(centroids)
    colors = ['red', 'blue', 'green', 'purple', 'orange']

    plt.figure(figsize=(8, 6))
    for k in range(K):
        cluster_points = data[assignments == k]
        plt.scatter(cluster_points[:, 2], cluster_points[:, 3], c=colors[k], label=f'Cluster {k+1}')
    
    plt.scatter(centroids[:, 2], centroids[:, 3], c='black', marker='X', s=200, label='Centroids')
    plt.title(title)
    plt.xlabel('Attribute 3 (Petal Length)')
    plt.ylabel('Attribute 4 (Petal Width)')
    plt.legend()
    plt.grid(True)
    plt.show()


def main():

    kmtest_df = pd.read_csv('kmtest.csv', sep=r'\s+', header=None)
    iris_df = pd.read_csv('iris.csv', header=None)
    kmtest_data = kmtest_df.values.astype(float)
    iris_data = iris_df.iloc[:, :4].values

    # --- Part 1: K-Means on kmtest data ---
    print("======================================================")
    print("Part 1: Analysis of kmtest Dataset")
    print("======================================================")
    
    # Part 1.b: With Normalization
    print("\n--- Part 1.b: Clustering with Normalization ---")
    mean = np.mean(kmtest_data, axis=0)
    std = np.std(kmtest_data, axis=0)
    kmtest_normalized = (kmtest_data - mean) / std
    
    k_values_to_test = [2, 3, 4, 5]
    colors = ['red', 'blue', 'green', 'purple', 'orange']

    for K_val in k_values_to_test:
        centroids, assignments = kmeans(kmtest_normalized, K_val)
        plt.figure(figsize=(8, 6))
        for k in range(K_val):
            cluster_points = kmtest_normalized[assignments == k]
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], c=colors[k], label=f'Cluster {k+1}')
        plt.scatter(centroids[:, 0], centroids[:, 1], c='black', marker='X', s=200, label='Centroids')
        plt.title(f'K-Means Clustering (K={K_val}) on Normalized Data')
        plt.xlabel('Feature 1 (Normalized)')
        plt.ylabel('Feature 2 (Normalized)')
        plt.legend()
        plt.grid(True)
        plt.show()

    # --- Part 2: K-Means on iris data ---
    print("\n======================================================")
    print("Part 2: Analysis of Iris Dataset")
    print("======================================================")
    
    # Part 2.a: Run multiple times to find best/worst
    print("\n--- Part 2.a/b: Finding Best and Worst Clustering ---")
    K = 3
    num_runs = 5
    results = []
    for run in range(num_runs):
        print(f"--- Running K-Means: Run #{run + 1} ---")
        centroids, assignments = kmeans(iris_data, K)
        current_sse = np.sum([np.sum((iris_data[assignments == k] - centroids[k])**2) for k in range(K)])
        print(f"SSE for this run: {current_sse:.4f}")
        results.append({'sse': current_sse, 'centroids': centroids, 'assignments': assignments})
    
    sorted_results = sorted(results, key=lambda x: x['sse'])
    best_result = sorted_results[0]
    worst_result = sorted_results[-1]
    print(f"\nBest SSE found: {best_result['sse']:.4f}")
    print(f"Worst SSE found: {worst_result['sse']:.4f}")

    # Part 2.b: Plot best and worst
    plot_iris_clusters(iris_data, best_result['assignments'], best_result['centroids'], 'Best Clustering Result (Lowest SSE)')
    plot_iris_clusters(iris_data, worst_result['assignments'], worst_result['centroids'], 'Worst Clustering Result (Highest SSE)')

    # Part 2.c: Plot original data 
    print("\n--- Part 2.c: Plotting Original Iris Species Data ---")
    true_labels = iris_df[4].astype('category').cat.codes.values
    species_names = iris_df[4].unique()
    plt.figure(figsize=(8, 6))
    for i, name in enumerate(species_names):
        points = iris_data[true_labels == i]
        plt.scatter(points[:, 2], points[:, 3], c=colors[i], label=name)
    plt.title('Original Iris Species (Ground Truth)')
    plt.xlabel('Attribute 3 (Petal Length)')
    plt.ylabel('Attribute 4 (Petal Width)')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Part 2.d: Calculate distance between centers
    print("\n--- Part 2.d: Calculating Distances Between Centers ---")
    original_centers = np.array([np.mean(iris_data[true_labels == k], axis=0) for k in range(K)])
    best_kmeans_centroids = best_result['centroids']
    print("Original Species Centers:\n", original_centers)
    print("Best K-Means Centroids:\n", best_kmeans_centroids)

    total_distance = 0
    for centroid in best_kmeans_centroids:
        distances = [np.linalg.norm(centroid - true_center) for true_center in original_centers]
        total_distance += np.min(distances)
    print(f"\nAverage distance to nearest true center: {total_distance / K:.4f}")



if __name__ == "__main__":
    main()
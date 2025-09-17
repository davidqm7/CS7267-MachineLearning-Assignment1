import pandas as pd  # For loading and managing data from CSV files
import numpy as np   # For numerical operations, especially with arrays
import matplotlib.pyplot as plt  # For creating plots and visualizations


# =============================================================================
# 2. FUNCTION DEFINITIONS
# =============================================================================
def kmeans(data, K, max_iters=100):
    """Performs K-Means clustering on the given data."""
    # Get the number of data points (samples) and features (dimensions)
    n_samples, n_features = data.shape
    
    # --- Initialization Step ---
    # Randomly select K unique data points to be the initial centroids
    random_indices = np.random.choice(n_samples, K, replace=False)
    centroids = data[random_indices]

    # Main loop: This will run for a maximum of 'max_iters' iterations
    for i in range(max_iters):
        # --- Assignment Step ---
        # Create an array to store the cluster assignment for each data point
        assignments = np.zeros(n_samples, dtype=int)
        
        # For each data point...
        for j in range(n_samples):
            # Calculate the Euclidean distance from this point to every centroid
            distances = [np.linalg.norm(data[j] - centroid) for centroid in centroids]
            # Assign the point to the cluster of the closest centroid
            assignments[j] = np.argmin(distances)

        # --- Update Step ---
        # Keep a copy of the old centroids to check if they change
        old_centroids = np.copy(centroids)
        
        # For each cluster (from 0 to K-1)...
        for k in range(K):
            # Get all the data points that were assigned to this cluster
            points_in_cluster = data[assignments == k]
            # If the cluster is not empty, calculate the mean to get the new centroid
            if len(points_in_cluster) > 0:
                centroids[k] = np.mean(points_in_cluster, axis=0)

        # --- Convergence Check ---
        # If the centroids did not move in this iteration, the algorithm has converged
        if np.array_equal(old_centroids, centroids):
            print(f"Converged after {i + 1} iterations.")
            break  # Exit the loop
            
    return centroids, assignments


def plot_iris_clusters(data, assignments, centroids, title):
    """Plots the iris clustering results using features at index 2 and 3."""
    # Determine the number of clusters from the number of centroids
    K = len(centroids)
    # Define a list of colors for the plots
    colors = ['red', 'blue', 'green', 'purple', 'orange']

    # Create a new plot figure
    plt.figure(figsize=(8, 6))
    
    # For each cluster...
    for k in range(K):
        # Select the data points belonging to the current cluster
        cluster_points = data[assignments == k]
        # Create a scatter plot using the 3rd and 4th columns (Petal Length/Width)
        plt.scatter(cluster_points[:, 2], cluster_points[:, 3], c=colors[k], label=f'Cluster {k+1}')
    
    # Plot the centroids on top, making them large and distinct
    plt.scatter(centroids[:, 2], centroids[:, 3], c='black', marker='X', s=200, label='Centroids')
    
    # Add helpful labels and a title
    plt.title(title)
    plt.xlabel('Attribute 3 (Petal Length)')
    plt.ylabel('Attribute 4 (Petal Width)')
    plt.legend()
    plt.grid(True)
    plt.show()


# =============================================================================
# 3. MAIN ANALYSIS SCRIPT
# =============================================================================
def main():
    """Main function to run the entire clustering analysis."""
    # --- Data Loading ---
    # Load kmtest.csv, specifying that columns are separated by whitespace
    kmtest_df = pd.read_csv('kmtest.csv', sep='\s+', header=None)
    # Load iris.csv, which has no header row
    iris_df = pd.read_csv('iris.csv', header=None)
    
    # Convert the DataFrames to NumPy arrays of type float for calculations
    kmtest_data = kmtest_df.values.astype(float)
    # For iris, select only the first 4 columns (the features)
    iris_data = iris_df.iloc[:, :4].values

    # --- Part 1: K-Means on kmtest data ---
    print("======================================================")
    print("Part 1: Analysis of kmtest Dataset")
    print("======================================================")
    
    # Part 1.b: With Normalization
    print("\n--- Part 1.b: Clustering with Normalization ---")
    # Calculate the mean and standard deviation for each column (axis=0)
    mean = np.mean(kmtest_data, axis=0)
    std = np.std(kmtest_data, axis=0)
    # Apply the Z-score normalization formula
    kmtest_normalized = (kmtest_data - mean) / std
    
    k_values_to_test = [2, 3, 4, 5]
    colors = ['red', 'blue', 'green', 'purple', 'orange']

    # Loop through each value of K for the analysis
    for K_val in k_values_to_test:
        # Run the K-Means algorithm on the normalized data
        centroids, assignments = kmeans(kmtest_normalized, K_val)
        
        # Create a new plot for this K value
        plt.figure(figsize=(8, 6))
        # Plot the points for each cluster
        for k in range(K_val):
            cluster_points = kmtest_normalized[assignments == k]
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], c=colors[k], label=f'Cluster {k+1}')
        # Plot the centroids
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
    results = [] # To store the results from each run
    
    # Run the algorithm 5 times to account for random initialization
    for run in range(num_runs):
        print(f"--- Running K-Means: Run #{run + 1} ---")
        centroids, assignments = kmeans(iris_data, K)
        
        # Calculate the Sum of Squared Errors (SSE) for this run
        current_sse = np.sum([np.sum((iris_data[assignments == k] - centroids[k])**2) for k in range(K)])
        print(f"SSE for this run: {current_sse:.4f}")
        
        # Save the SSE, centroids, and assignments for this run
        results.append({'sse': current_sse, 'centroids': centroids, 'assignments': assignments})
    
    # Sort the results by SSE to find the best (lowest) and worst (highest)
    sorted_results = sorted(results, key=lambda x: x['sse'])
    best_result = sorted_results[0]
    worst_result = sorted_results[-1]
    print(f"\nBest SSE found: {best_result['sse']:.4f}")
    print(f"Worst SSE found: {worst_result['sse']:.4f}")

    # Part 2.b: Plot best and worst results
    plot_iris_clusters(iris_data, best_result['assignments'], best_result['centroids'], 'Best Clustering Result (Lowest SSE)')
    plot_iris_clusters(iris_data, worst_result['assignments'], worst_result['centroids'], 'Worst Clustering Result (Highest SSE)')

    # Part 2.c: Plot original data (ground truth)
    print("\n--- Part 2.c: Plotting Original Iris Species Data ---")
    # Get the true labels from the 5th column and convert to numbers (0, 1, 2)
    true_labels = iris_df[4].astype('category').cat.codes.values
    species_names = iris_df[4].unique()
    
    # Plot the data using the true labels for color
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
    # Calculate the true center for each species
    original_centers = np.array([np.mean(iris_data[true_labels == k], axis=0) for k in range(K)])
    best_kmeans_centroids = best_result['centroids']
    print("Original Species Centers:\n", original_centers)
    print("Best K-Means Centroids:\n", best_kmeans_centroids)

    total_distance = 0
    # For each K-Means centroid, find the closest true center and sum the distances
    for centroid in best_kmeans_centroids:
        distances = [np.linalg.norm(centroid - true_center) for true_center in original_centers]
        total_distance += np.min(distances)
    print(f"\nAverage distance to nearest true center: {total_distance / K:.4f}")


# =============================================================================
# 4. SCRIPT EXECUTION
# =============================================================================
# This is a standard Python convention. The code inside this block will only
# run when the script is executed directly (not when it's imported).
if __name__ == "__main__":
    main()
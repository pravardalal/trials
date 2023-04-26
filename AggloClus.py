import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

class AgglomerativeClustering:
    def __init__(self, n_clusters):
        self.n_clusters = n_clusters
    
    def fit(self, X):
        # Initialize each observation as its own cluster
        clusters = [[i] for i in range(X.shape[0])]
        # Compute the distance matrix
        dist_matrix = np.zeros((X.shape[0], X.shape[0]))
        for i in range(X.shape[0]):
            for j in range(i+1, X.shape[0]):
                dist_matrix[i,j] = np.linalg.norm(X[i,:] - X[j,:])
        # Compute the dendrogram
        dendrogram = np.zeros((X.shape[0]-1, 4))
        for i in range(X.shape[0]-1):
            # Find the closest pair of clusters
            min_dist = np.inf
            for j in range(len(clusters)):
                for k in range(j+1, len(clusters)):
                    dist = np.min(dist_matrix[clusters[j],:][:,clusters[k]])
                    if dist < min_dist:
                        min_dist = dist
                        merge_clusters = (j,k)
            # Merge the two closest clusters
            dendrogram[i,0] = merge_clusters[0]
            dendrogram[i,1] = merge_clusters[1]
            dendrogram[i,2] = min_dist
            dendrogram[i,3] = len(clusters[merge_clusters[0]]) + len(clusters[merge_clusters[1]])
            clusters[merge_clusters[0]] += clusters[merge_clusters[1]]
            del clusters[merge_clusters[1]]
            # Update the distance matrix
            for j in range(len(clusters)-1):
                for k in range(j+1, len(clusters)):
                    min_dists = []
                    for l in clusters[j]:
                        for m in clusters[k]:
                            min_dists.append(dist_matrix[l,m])
                    dist_matrix[j,k] = min(min_dists)
                    dist_matrix[k,j] = dist_matrix[j,k]
        # Assign labels based on the dendrogram
        self.labels_ = np.zeros(X.shape[0], dtype=np.int32)
        for i in range(X.shape[0]):
            for j in range(dendrogram.shape[0]):
                if dendrogram[j,0] <= i < dendrogram[j,3]:
                    self.labels_[i] = j + X.shape[0] - self.n_clusters
                    break
        # Store the dendrogram
        self.dendrogram = dendrogram
    
    def plot_dendrogram(self):
        # Plot the dendrogram
        plt.figure(figsize=(10,6))
        plt.title('Dendrogram')
        plt.xlabel('Observations')
        plt.ylabel('Distance')
        plt.xticks([])
        plt.yticks([])
        for i in range(self.dendrogram.shape[0]):
            x1 = self.dendrogram[i,0]
            x2 = self.dendrogram[i,1]
            y1 = self.dendrogram[i,2]
            y2 = self.dendrogram[i,3]
            plt.plot([x1,x1,x2,x2], [y1,y2,y2,y1], 'k-')
        plt.show()


# Load the Iris dataset
iris = load_iris()
X = iris.data

# Create an instance of the AgglomerativeClustering class
n_clusters = 3
model = AgglomerativeClustering(n_clusters)

# Fit the model to the data and plot the dend
model.fit(X)
model.plot_dendrogram()

# Print the cluster labels
print(model.labels_)

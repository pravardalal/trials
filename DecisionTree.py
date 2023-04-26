import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Define the DecisionTree class
class DecisionTree:
    def __init__(self, max_depth=5, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
    
    def fit(self, X, y):
        # Define the node structure
        class Node:
            def __init__(self, feature_idx=None, threshold=None, left=None, right=None, is_leaf=False, label=None):
                self.feature_idx = feature_idx
                self.threshold = threshold
                self.left = left
                self.right = right
                self.is_leaf = is_leaf
                self.label = label
        
        # Define the entropy calculation function
        def entropy(y):
            _, counts = np.unique(y, return_counts=True)
            p = counts / len(y)
            return -np.sum(p * np.log2(p))
        
        # Define the information gain calculation function
        def info_gain(X, y, feature_idx, threshold):
            left_idx = X[:, feature_idx] < threshold
            left_y = y[left_idx]
            right_y = y[~left_idx]
            p_left = len(left_y) / len(y)
            p_right = 1 - p_left
            ig = entropy(y) - p_left * entropy(left_y) - p_right * entropy(right_y)
            return ig
        
        # Define the recursive splitting function
        def split(X, y, depth):
            # Check if the stopping criterion is met
            if depth >= self.max_depth or len(X) < self.min_samples_split or len(np.unique(y)) == 1:
                label = np.bincount(y).argmax()
                return Node(is_leaf=True, label=label)
            
            # Find the best feature and threshold to split on
            best_feature_idx, best_threshold, best_ig = None, None, 0
            for feature_idx in range(X.shape[1]):
                thresholds = np.unique(X[:, feature_idx])
                for threshold in thresholds:
                    ig = info_gain(X, y, feature_idx, threshold)
                    if ig > best_ig:
                        best_feature_idx, best_threshold, best_ig = feature_idx, threshold, ig
            
            # Split the data and recursively call the function on the left and right nodes
            left_idx = X[:, best_feature_idx] < best_threshold
            right_idx = ~left_idx
            left_node = split(X[left_idx], y[left_idx], depth+1)
            right_node = split(X[right_idx], y[right_idx], depth+1)
            return Node(feature_idx=best_feature_idx, threshold=best_threshold, left=left_node, right=right_node)
        
        # Train the decision tree by recursively splitting the data
        self.root = split(X, y, depth=0)
    
    def predict(self, X):
        # Predict the class labels for the input data
        def traverse(node, x):
            if node.is_leaf:
                return node.label
            if x[node.feature_idx] < node.threshold:
                return traverse(node.left, x)
            else:
                return traverse(node.right, x)
        
        y_pred = np.array([traverse(self.root, x) for x in X])
        return y_pred

# Load the iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split the data into training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#Train the decision tree
dt = DecisionTree()
dt.fit(X_train, y_train)

#Predict the labels for the test data
y_pred = dt.predict(X_test)

#Evaluate the performance of the model
accuracy = np.sum(y_pred == y_test) / len(y_test)
print("Accuracy:", accuracy)
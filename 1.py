from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Generate sample data
X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
                           random_state=1, n_clusters_per_class=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=1)

# Create a multi-layer perceptron classifier
mlp = MLPClassifier(hidden_layer_sizes=(10,), max_iter=1000)

# Train the classifier on the training data
mlp.fit(X_train, y_train)

# Predict the classes of the test data
y_pred = mlp.predict(X_test)

# Compute the accuracy of the classifier
accuracy = mlp.score(X_test, y_test)
print(f"Accuracy: {accuracy:.2f}")

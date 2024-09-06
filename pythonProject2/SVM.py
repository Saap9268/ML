import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Step 1: Import Libraries
# Libraries already imported at the beginning of the code

# Step 2: Generate Synthetic Data
# Generate a dataset with a nonlinear decision boundary
X, y = datasets.make_moons(n_samples=300, noise=0.2, random_state=42)

# Step 3: Visualize the Data
plt.figure(figsize=(8, 6))
plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='red', label='Class 0', edgecolors='k')
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='blue', label='Class 1', edgecolors='k')
plt.title('Synthetic Moons Dataset')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()

# Step 4: Train the SVM Model
# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the SVM model with an RBF kernel
svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)

# Fit the model on the training data
svm_model.fit(X_train, y_train)

# Step 5: Evaluate and Visualize Results
# Predict on the test set
y_pred = svm_model.predict(X_test)

# Print accuracy score
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Function to plot decision boundary
def plot_decision_boundary(model, X, y, title="Decision Boundary"):
    h = .02  # Step size in the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.coolwarm)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o', cmap=plt.cm.coolwarm)
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

# Plot the decision boundary
plot_decision_boundary(svm_model, X_test, y_test, title="SVM Decision Boundary with RBF Kernel")

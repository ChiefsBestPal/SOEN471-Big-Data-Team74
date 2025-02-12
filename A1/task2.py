from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import GridSearchCV

# Remove the 'CustomerID' column (not useful for prediction) and separate features from the target.
X = df.drop(columns=["CustomerID", "Churn"])  # All input features except for 'CustomerID' and target 'Churn'
y = df["Churn"]  # The target variable: predicting the likelihood of a customer to churn

# Split the dataset into training and testing sets.
# - 70% of data is used for training and 30% for testing.
# - random_state=42 ensures the split is reproducible.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42) #Separating dataset into 2 parts, 70% will be used for training and the remaining 30% will be used for testing

# Initialize a Decision Tree Classifier with a fixed random state for reproducibility
dt = DecisionTreeClassifier(random_state=42)

# TEST 2 AGAIN

#TODO: DOCUMENTATION HERE
# params = {
#     'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, None],
#     'min_samples_split': [2, 3, 4, 5, 6, 7, 8, 9, 10],
#     'min_samples_leaf': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
#     'criterion': ['gini', 'entropy'],
#     # 'max_features': ['sqrt', 'log2', None],
#     # 'min_weight_fraction_leaf': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
# }

# Define a dictionary of hyperparameters (the grid) to test during model tuning.
# These hyperparameters control the complexity and behavior of the tree.
params = {
    'max_depth': [3, 5, 7, 10, None],  # Maximum depth of the tree; 'None' means no limit.
                                       # Shallow trees (e.g., 3 or 5) help avoid overfitting,
                                       # while deeper trees (e.g., 7 or 10) allow capturing more complex patterns.
    'min_samples_split': [2, 5, 10],    # Minimum number of samples required to split an internal node.
                                        # Higher values help reduce model complexity by requiring more data for a split.
    'min_samples_leaf': [1, 2, 4],       # Minimum number of samples required to be at a leaf node.
                                         # Ensures that leaves have a sufficient number of samples to avoid overfitting.
    'criterion': ['gini', 'entropy']      # The function to measure the quality of a split:
                                          # 'gini' calculates Gini impurity, and 'entropy' uses information gain.
}

# Grid Search for hyperparameter tuning
# Perform 5-fold cross-validation on a decision tree model with hyperparameters defined in params and evaluate them by accuracy and fitting the best configuration on the training data
grid_search = GridSearchCV(dt, param_grid=params, cv=5, scoring="accuracy")
grid_search.fit(X_train, y_train)

# Display the best hyperparameters found during grid search.
print("Best Hyperparameters:", grid_search.best_params_)

# Retrieve the best decision tree model based on grid search results.
best_dt = grid_search.best_estimator_

# Use the best decision tree model to predict the target variable on the test set.
y_pred = best_dt.predict(X_test)

# Calculate and print various performance metrics:
print("Accuracy:", accuracy_score(y_test, y_pred)) # - Accuracy: the overall correctness of the model.
print("Precision:", precision_score(y_test, y_pred)) # - Precision: the proportion of positive identifications that were actually correct.
print("Recall:", recall_score(y_test, y_pred)) # - Recall: the proportion of actual positives that were correctly identified.
print("F1 Score:", f1_score(y_test, y_pred)) # - F1 Score: the harmonic mean of precision and recall.
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred)) # - Confusion Matrix: a table that outlines the performance of the model.

# Visualize the structure of the optimized decision tree.
plt.figure(figsize=(15, 8))
plot_tree(best_dt, feature_names=X.columns, class_names=["No Churn", "Churn"], filled=True)
plt.title("Optimized Decision Tree")
plt.show()
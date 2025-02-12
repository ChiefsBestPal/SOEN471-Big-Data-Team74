from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# # Define Hyperparameter Grid for Random Forest Tuning
param_grid = {
    'n_estimators': [50, 100, 200],             # Number of trees in the forest:
                                               # More trees can improve model performance at the cost of increased computation.
    'max_depth': [1, 3, 5, 7, 10, None],           # Maximum depth of each tree:
                                               # Limits tree complexity; 'None' means no depth limit.
    'min_samples_split': [2, 5, 10],            # Minimum number of samples required to split an internal node:
                                               # Higher values prevent overfitting by requiring more data to make a split.
    'min_samples_leaf': [1, 2, 4],              # Minimum number of samples required at a leaf node:
                                               # Ensures leaves have enough samples to be representative.
    'max_features': ['sqrt', 'log2', None]      # Number of features to consider when looking for the best split:
                                               # 'sqrt' and 'log2' introduce randomness; 'None' uses all features.
}

# Train Random Forest
rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(rf, param_grid, cv=5, n_jobs=-1, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Best parameters
print("Best Hyperparameters:", grid_search.best_params_)

# best_rf = RandomForestClassifier(
#     n_estimators=100,          # Number of trees in the forest
#     # max_depth=3,              # Maximum depth of the trees
#     # min_samples_split=10,       # Minimum samples to split an internal node
#     # min_samples_leaf=1,        # Minimum samples required to be a leaf node
#     # criterion='entropy',          # Split criterion ('gini' or 'entropy')
#     random_state=42            # For reproducibility
# )

# best_rf.fit(X_train, y_train)

# Use the best model
best_rf = grid_search.best_estimator_
y_pred_best_rf = best_rf.predict(X_test)

# Evaluate Random Forest
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_best_rf))
print("Random Forest Precision:", precision_score(y_test, y_pred_best_rf))
print("Random Forest Recall:", recall_score(y_test, y_pred_best_rf))
print("Random Forest F1 Score:", f1_score(y_test, y_pred_best_rf))
print("Random Forest Confusion Matrix:\n", confusion_matrix(y_test, y_pred_best_rf))

# Feature importance (from the best model)
feature_importances = pd.Series(best_rf.feature_importances_, index=X.columns)
feature_importances.nlargest(10).plot(kind="barh")
plt.title("Feature Importances in Random Forest")
plt.show()
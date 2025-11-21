# Import libraries
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from scipy.stats import randint

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("✅ Data Loaded Successfully!")
print("Training Shape:", X_train.shape)
print("Testing Shape:", X_test.shape)

# Define parameter grid
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [3, 5, 7, None],
    'min_samples_split': [2, 5, 10]
}

# Initialize model
rf = RandomForestClassifier(random_state=42)

# Grid Search
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=3,
    scoring='accuracy',
    verbose=1,
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

# Print best parameters and score
print("\n🎯 Best Parameters from GridSearchCV:", grid_search.best_params_)
print("✅ Best Accuracy (CV):", grid_search.best_score_)

best_rf_grid = grid_search.best_estimator_
y_pred_grid = best_rf_grid.predict(X_test)

print("\n🎯 Test Accuracy (GridSearchCV):", accuracy_score(y_test, y_pred_grid))

# Define parameter distributions
param_dist = {
    'n_estimators': randint(50, 200),
    'max_depth': randint(3, 10),
    'min_samples_split': randint(2, 10)
}

# Random Search
random_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_dist,
    n_iter=10,   # Try 10 random combinations
    cv=3,
    scoring='accuracy',
    random_state=42,
    verbose=1,
    n_jobs=-1
)

random_search.fit(X_train, y_train)

# Print best parameters and score
print("\n🎯 Best Parameters from RandomizedSearchCV:", random_search.best_params_)
print("✅ Best Accuracy (CV):", random_search.best_score_)

best_rf_random = random_search.best_estimator_
y_pred_random = best_rf_random.predict(X_test)

print("\n🎯 Test Accuracy (RandomizedSearchCV):", accuracy_score(y_test, y_pred_random))

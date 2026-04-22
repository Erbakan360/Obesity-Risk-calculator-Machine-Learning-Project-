# ==== KNN.py ====
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Forward feature selection with CV (optional)
def knn_forward_selection(x_train, y_train, numeric_cols):
    xSelected = []
    cRateBest = 0
    max_features = 10
    all_features = list(x_train.columns)

    while len(xSelected) < max_features:
        cRate = [0] * len(all_features)
        for i, feature in enumerate(all_features):
            if feature not in xSelected:
                temp_features = xSelected + [feature]
                scaler_cols = [f for f in temp_features if f in numeric_cols]

                # Create pipeline with polynomial feature expansion and scaling
                pipe = Pipeline([
                    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
                    ('scaler', StandardScaler()),
                    ('knn', KNeighborsClassifier(n_neighbors=5))
                ])

                scores = cross_val_score(pipe, x_train[temp_features], y_train.values.ravel(), cv=5)
                cRate[i] = scores.mean()

        max_score = max(cRate)
        best_feature_index = cRate.index(max_score)

        if max_score > cRateBest:
            xSelected.append(all_features[best_feature_index])
            cRateBest = max_score
        else:
            break

    print("\nSelected features (Forward Selection + CV):", xSelected)
    return xSelected

def run_knn(x_train, y_train, x_test, y_test, selected_features, numeric_cols):
    print("\nRunning KNN with Polynomial Feature Expansion and Cross-Validation")
    y_train = y_train.values.ravel()
    y_test = y_test.values.ravel()

    # --- Graph 1: Compare raw vs. CV accuracy over different k ---
    raw_accuracies = []
    cv_accuracies = []
    k_range = list(range(1, 11))

    for k in k_range:
        pipe = Pipeline([
            ('poly', PolynomialFeatures(degree=2, include_bias=False)),
            ('scaler', StandardScaler()),
            ('knn', KNeighborsClassifier(n_neighbors=k))
        ])
        pipe.fit(x_train[selected_features], y_train)
        y_pred = pipe.predict(x_test[selected_features])
        raw_accuracies.append(accuracy_score(y_test, y_pred))

        scores = cross_val_score(pipe, x_train[selected_features], y_train, cv=5)
        cv_accuracies.append(scores.mean())

    plt.figure(figsize=(8, 5))
    plt.plot(k_range, raw_accuracies, label='Test Accuracy (No CV)', marker='o')
    plt.plot(k_range, cv_accuracies, label='Cross-Validated Accuracy', marker='x')
    plt.title("KNN Accuracy vs. Number of Neighbors")
    plt.xlabel("k (Number of Neighbors)")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # --- Graph 2: Accuracy gain through feature space lifting (PolyFeatures) ---
    k_best = 5
    pipe_no_poly = Pipeline([
        ('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier(n_neighbors=k_best))
    ])
    acc_no_poly = cross_val_score(pipe_no_poly, x_train[selected_features], y_train, cv=5).mean()

    pipe_poly = Pipeline([
        ('poly', PolynomialFeatures(degree=2, include_bias=False)),
        ('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier(n_neighbors=k_best))
    ])
    acc_poly = cross_val_score(pipe_poly, x_train[selected_features], y_train, cv=5).mean()

    plt.figure(figsize=(6, 4))
    plt.bar(['No PolyFeatures', 'With PolyFeatures'], [acc_no_poly, acc_poly],
            color=['gray', 'steelblue'])
    plt.title(f"Impact of Feature Lifting (k={k_best})")
    plt.ylabel("CV Accuracy")
    plt.ylim(0.5, 1.0)
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()

    # --- Grid Search to finalize best k ---
    param_grid = {'knn__n_neighbors': k_range}
    final_pipe = Pipeline([
        ('poly', PolynomialFeatures(degree=2, include_bias=False)),
        ('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier())
    ])
    grid = GridSearchCV(final_pipe, param_grid, cv=5)
    grid.fit(x_train[selected_features], y_train)

    print(f"\nBest parameters from GridSearchCV: {grid.best_params_}")
    print(f"Best cross-validated accuracy: {grid.best_score_:.3f}")

    y_pred = grid.predict(x_test[selected_features])
    acc = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {acc:.3f}")

    # --- Final confusion matrix ---
    plt.figure()
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix - Final KNN")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

    # Final GridSearchCV to find best k
    pipe = Pipeline([
        ('poly', PolynomialFeatures(degree=2, include_bias=False)),
        ('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier())
    ])

    param_grid = {
        'knn__n_neighbors': k_range
    }

    grid = GridSearchCV(pipe, param_grid, cv=5)
    grid.fit(x_train[selected_features], y_train.values.ravel())

    print(f"\nBest parameters from GridSearchCV: {grid.best_params_}")
    print(f"Best cross-validated accuracy: {grid.best_score_:.3f}")

    y_pred = grid.predict(x_test[selected_features])
    acc = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {acc:.3f}")

    plt.figure()
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix - KNN with PolyFeatures + CV")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

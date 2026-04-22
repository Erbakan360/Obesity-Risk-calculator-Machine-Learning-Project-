import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.svm import SVC


def run_svm_comparison(x_train, y_train):
    # Flatten label array if needed
    if hasattr(y_train, 'values'):
        y_train = y_train.values.ravel()
    else:
        y_train = y_train.ravel()

    # ------------------------
    # Base SVM pipeline
    # ------------------------
    svm_base = Pipeline([
        ('scaler', StandardScaler()),
        ('select', SelectKBest(score_func=f_classif, k=15)),
        ('model', SVC(kernel='rbf'))
    ])

    param_grid = {
        'model__C': [0.1, 1, 10],
        'model__gamma': [0.01, 0.1, 1]
    }

    grid_base = GridSearchCV(svm_base, param_grid=param_grid, cv=5, scoring='accuracy')
    grid_base.fit(x_train, y_train)
    base_score = grid_base.best_score_

    # ------------------------
    # Feature Space Lifting
    # ------------------------
    numeric_cols = [col for col in x_train.columns if col.startswith("remainder__")]
    x_train_numeric = x_train[numeric_cols]

    poly = PolynomialFeatures(degree=2, include_bias=False)
    x_train_poly = poly.fit_transform(x_train_numeric)
    poly_feature_names = poly.get_feature_names_out(numeric_cols)
    x_train_poly_df = pd.DataFrame(x_train_poly, columns=poly_feature_names, index=x_train.index)

    # Combine with categorical columns
    categorical_cols = [col for col in x_train.columns if not col.startswith("remainder__")]
    x_train_lifted = pd.concat([x_train[categorical_cols], x_train_poly_df], axis=1)

    # ------------------------
    # Lifted SVM pipeline
    # ------------------------
    svm_lifted = Pipeline([
        ('scaler', StandardScaler()),
        ('select', SelectKBest(score_func=f_classif, k=15)),
        ('model', SVC(kernel='rbf'))
    ])

    grid_lifted = GridSearchCV(svm_lifted, param_grid=param_grid, cv=5, scoring='accuracy')
    grid_lifted.fit(x_train_lifted, y_train)
    lifted_score = grid_lifted.best_score_

    # ------------------------
    # Results and Visualization
    # ------------------------
    print("=============== SVM Summary ===============")
    print(f"Base SVM Best Accuracy (CV):   {base_score:.4f}")
    print(f"Lifted SVM Best Accuracy (CV): {lifted_score:.4f}")
    print("===========================================")

    # Accuracy comparison bar plot
    plt.figure(figsize=(7, 4))
    plt.bar(['Base SVM', 'Lifted SVM'], [base_score, lifted_score], color=['mediumseagreen', 'salmon'])
    plt.title("SVM Accuracy: Before vs After Feature Lifting")
    plt.ylabel("Cross-Validation Accuracy")
    plt.ylim(0.7, 1.0)
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()

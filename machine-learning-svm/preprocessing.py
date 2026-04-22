# ==== preprocessing.py ====
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split

def load_and_preprocess():
    data_set = pd.read_csv("data.csv")

    # Basic dataset info
    print("\nColumn names:")
    print(", ".join(data_set.columns))

    print("\nNull values:")
    print(data_set.isnull().sum())

    print("\nNon-numeric columns:")
    print(", ".join(data_set.select_dtypes(exclude=np.number).columns.tolist()))

    print("\nData types:")
    for col, dtype in data_set.dtypes.items():
        print(f" - {col}: {dtype}")

    x = data_set.iloc[:, :-1]
    y = data_set.iloc[:, -1].values

    # One-hot encode categorical columns
    xtransforming = ColumnTransformer(transformers=[
        ('encoder', OneHotEncoder(), [0, 4, 5, 8, 9, 11, 14, 15])
    ], remainder='passthrough')

    x = np.array(xtransforming.fit_transform(x))
    column_names = xtransforming.get_feature_names_out()

    # Encode target variable
    ytransforming = LabelEncoder()
    y = ytransforming.fit_transform(y)

    # Split the dataset
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=94469)

    # Convert to DataFrames
    x_train = pd.DataFrame(x_train, columns=column_names)
    x_test = pd.DataFrame(x_test, columns=column_names)
    y_train = pd.DataFrame(y_train)
    y_test = pd.DataFrame(y_test)

    numeric_cols = [col for col in column_names if col.startswith("remainder__")]

    # === Data Visualization ===
    num_cols = len(data_set.columns)
    cols = 4
    rows = (num_cols + cols - 1) // cols

    plt.figure(figsize=(cols * 4, rows * 3))
    plt.suptitle("Unaltered Data Set Histograms", fontsize=16)

    for i, col in enumerate(data_set.columns):
        plt.subplot(rows, cols, i + 1)
        plt.hist(data_set[col], bins=12, color='green', edgecolor='black')
        plt.xlabel(col, fontsize=8)
        plt.ylabel("Occurrences", fontsize=8)
        plt.xticks(rotation=45)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

    # Histogram for numeric features after encoding
    plt.figure(figsize=(15, 10))
    for i, col in enumerate(numeric_cols):
        plt.subplot(2, 4, i + 1)
        plt.hist(x_train[col], bins=10, color='steelblue', edgecolor='black')
        plt.title(f"Occurrences of {col}")
        plt.xlabel(col)
        plt.ylabel("No. of Occurrences")
    plt.tight_layout()
    plt.show()

    return x_train, x_test, y_train, y_test, column_names, numeric_cols

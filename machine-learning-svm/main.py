from preprocessing import load_and_preprocess
from svm import run_svm_comparison
from KNN import knn_forward_selection, run_knn
from Bayes import run_bayes

# Step 1: Load preprocessed data
x_train, x_test, y_train, y_test, column_names,numeric_cols = load_and_preprocess()

# Step 2: Run the optimized SVM model
run_svm_comparison(x_train, y_train)

# Optional: Uncomment these if you want to run other models
selected_features = knn_forward_selection(x_train, y_train,numeric_cols)
run_knn(x_train, y_train, x_test, y_test, selected_features,numeric_cols)
run_bayes(x_train, y_train, x_test, y_test)

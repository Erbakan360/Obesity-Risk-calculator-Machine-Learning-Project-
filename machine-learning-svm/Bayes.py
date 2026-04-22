from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

def bayes_forward_selection(x_train, y_train, x_test, y_test):
    xSelected = []
    cRateBest = 0
    max_features = 10
    all_features = list(x_train.columns)

    while len(xSelected) < max_features:
        cRate = [0] * len(all_features)
        for i, feature in enumerate(all_features):
            if feature not in xSelected:
                temp_features = xSelected + [feature]
                nb = GaussianNB()
                nb.fit(x_train[temp_features], list(y_train[0]))
                score = nb.score(x_test[temp_features], list(y_test[0]))
                cRate[i] = score

        max_score = max(cRate)
        best_feature_index = cRate.index(max_score)

        if max_score > cRateBest:
            xSelected.append(all_features[best_feature_index])
            cRateBest = max_score
        else:
            break

    print("\nSelected features (Forward Selection for Bayes):", xSelected)
    return xSelected

def run_bayes(x_train, y_train, x_test, y_test, selected_features=None):
    nb = GaussianNB()
    if selected_features:
        nb.fit(x_train[selected_features], list(y_train[0]))
        y_pred = nb.predict(x_test[selected_features])
    else:
        nb.fit(x_train, list(y_train[0]))
        y_pred = nb.predict(x_test)

    print("\nNaive Bayes Classifier Results:")
    print("Accuracy:", accuracy_score(y_test, y_pred))

    plt.figure()
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Greens")
    plt.title("Confusion Matrix - Naive Bayes")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

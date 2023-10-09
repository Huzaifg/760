import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from knn import NearestNeighborClassifier
from logisticRegression import LogisticRegressionClassifier
from sklearn import metrics


def generate_roc_points(y_true, y_scores):
    sorted_indices = np.argsort(y_scores)
    y_true_sorted = y_true[sorted_indices]

    # Initialize variables
    num_neg = np.sum(y_true == 0)
    num_pos = np.sum(y_true == 1)
    TP = 0
    FP = 0
    last_TP = 0
    roc_curve = []

    for i in range(len(sorted_indices)):
        if i > 0 and y_scores[sorted_indices[i]] != y_scores[sorted_indices[i - 1]] and y_true_sorted[i] == 0 and TP > last_TP:
            FPR = FP / num_neg
            TPR = TP / num_pos
            roc_curve.append((FPR, TPR))
            last_TP = TP

        if y_true_sorted[i] == 1:
            TP += 1
        else:
            FP += 1

    # Add the last point
    FPR = FP / num_neg
    TPR = TP / num_pos
    roc_curve.append((FPR, TPR))

    return roc_curve


if __name__ == '__main__':
    # Load the emails data file into pandas dataframe
    emails = pd.read_csv('./data/emails.csv', header='infer')
    # Drop the first column
    emails = emails.drop('Email No.', axis=1)

    num_cols = len(emails.columns) - 1

    # Single training test split
    train_data = emails.iloc[0:4000]
    test_data = emails.iloc[4000:]

    # Train with KNN and test
    clf = NearestNeighborClassifier()
    clf.fit(train_data.iloc[:, 0:num_cols].values,
            train_data.iloc[:, num_cols].values, k=5)
    y_pred_knn = clf.predict(test_data.iloc[:, 0:num_cols].values)
    y_probs_knn = clf.predict_proba()

    # Train with Logistic Regression and test
    clf = LogisticRegressionClassifier()
    clf.fit(train_data.iloc[:, 0:num_cols].values,
            train_data.iloc[:, num_cols].values, lr=0.00001, epochs=10000)
    y_pred_lr = clf.predict(test_data.iloc[:, 0:num_cols].values)
    y_probs_lr = clf.predict_proba()

    fpr_lr, tpr_lr, thresholds = metrics.roc_curve(
        test_data.iloc[:, num_cols].values, y_probs_lr, pos_label=1)
    lr_area = metrics.auc(fpr_lr, tpr_lr)
    fpr, tpr, thresholds = metrics.roc_curve(
        test_data.iloc[:, num_cols].values, y_probs_knn, pos_label=1)
    knn_area = metrics.auc(fpr, tpr)
    # Plot ROC curve
    plt.figure()
    # plt.plot([x[0] for x in roc_curve], [x[1] for x in roc_curve])
    plt.plot(fpr, tpr, label=f"KNN with K = 5 (area = {knn_area.round(2)})")
    plt.plot(fpr_lr, tpr_lr,
             label=f"Logistic Regression (area = {lr_area.round(2)})")
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig('./images/roc_curve.png')
    plt.show()

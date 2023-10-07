import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from logisticRegression import LogisticRegressionClassifier


if __name__ == "__main__":
    # Read the emails data
    emails = pd.read_csv('./data/emails.csv', header='infer')
    # Drop the first column
    emails = emails.drop('Email No.', axis=1)

    num_cols = len(emails.columns) - 1
    test_start = 0
    test_end = 1000
    num_folds = 5
    test_data = [None]*num_folds
    train_data = [None]*num_folds

    # split data into 5 folds each has training and testing sets

    for i in range(1, num_folds+1):
        test_data[i-1] = emails[test_start:test_end]
        train_data[i-1] = emails.drop(emails.index[test_start:test_end])
        test_start = test_end
        test_end = test_end + 1000

    # Train on each and test for accuracy, precision and recall
    accuracies = []
    precisions = []
    recalls = []
    for i in range(0, num_folds):
        # Create classifier
        clf = LogisticRegressionClassifier()
        clf.fit(train_data[i].iloc[:, 0:num_cols].values,
                train_data[i].iloc[:, num_cols].values, lr=0.01, epochs=1000)

        # Test data is the meshgrid
        y_pred = clf.predict(test_data[i].iloc[:, 0:num_cols].values)

        # Calculate accuracy
        accuracy = np.sum(
            y_pred.astype(int) == test_data[i].iloc[:, num_cols].values.astype(int)) / len(y_pred)
        accuracies.append(accuracy)

        # Calculate precision
        true_positives = np.sum(np.logical_and(
            y_pred.astype(int) == 1, test_data[i].iloc[:, num_cols].values.astype(int) == 1))
        false_positives = np.sum(np.logical_and(
            y_pred.astype(int) == 1, test_data[i].iloc[:, num_cols].values.astype(int) == 0))
        precision = true_positives / (true_positives + false_positives)
        precisions.append(precision)

        # Calculate recall
        false_negatives = np.sum(np.logical_and(
            y_pred.astype(int) == 0, test_data[i].iloc[:, num_cols].values.astype(int) == 1))
        recall = true_positives / (true_positives + false_negatives)
        recalls.append(recall)

        print('Fold ', i+1, ' accuracy: ', accuracy)
        print('Fold ', i+1, ' precision: ', precision)
        print('Fold ', i+1, ' recall: ', recall)

    print('Accuracy: ', accuracies)
    print('Precision: ', precisions)
    print('Recall: ', recalls)

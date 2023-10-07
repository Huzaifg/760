import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from knn import NearestNeighborClassifier

if __name__ == '__main__':
    # Load the emails data file into pandas dataframe
    emails = pd.read_csv('./data/emails.csv', header='infer')
    # Drop the first column
    emails = emails.drop('Email No.', axis=1)

    num_cols = len(emails.columns) - 1
    num_folds = 5
    test_data = [None]*num_folds
    train_data = [None]*num_folds

    # split data into 5 folds each has training and testing sets

    test_start = 0
    test_end = 1000

    for i in range(1, num_folds+1):
        test_data[i-1] = emails[test_start:test_end]
        train_data[i-1] = emails.drop(emails.index[test_start:test_end])
        test_start = test_end
        test_end = test_end + 1000

    # Train on each and test for accuracy, precision and recall
    k_arr = [1, 3, 5, 7, 10]
    avg_accuracies = []
    avg_precisions = []
    avg_recalls = []

    for k in k_arr:
        accuracies = []
        precisions = []
        recalls = []
        for i in range(0, num_folds):
            # Create classifier
            clf = NearestNeighborClassifier()
            clf.fit(train_data[i].iloc[:, 0:num_cols].values,
                    train_data[i].iloc[:, num_cols].values, k=k)

            # Test data
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
        print('Average accuracy: ', np.mean(accuracies))
        print('Average precision: ', np.mean(precisions))
        print('Average recall: ', np.mean(recalls))
        avg_accuracies.append(np.mean(accuracies))
        avg_precisions.append(np.mean(precisions))
        avg_recalls.append(np.mean(recalls))

    # Plot the accuracies, precisions and recalls
    plt.plot(k_arr, avg_accuracies, label='Accuracy')
    plt.savefig('./images/2_4.png')
    plt.show()

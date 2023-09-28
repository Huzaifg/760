from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Add matplotlib rc parameters to beautify plots
SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 16

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


if __name__ == "__main__":

    dataset = 'Dbig'
    D = np.loadtxt("data/" + dataset + ".txt", delimiter=" ")

    # Shuffle the array randomly
    np.random.shuffle(D)

    # Split the array into two sets of 8192 items each
    D32 = D[:32]
    D128 = D[:128]
    D512 = D[:512]
    D2048 = D[:2048]
    D8192 = D[:8192]

    test_set = D[8192:]

    dataList = [D32, D128, D512, D2048, D8192]

    dataDict = {0: 'D32', 1: 'D128',
                2: 'D512', 3: 'D2048', 4: 'D8192'}

    n = 5
    errors = [None]*5
    num_nodes = [None]*5
    for idx, train_data in enumerate(dataList):
        # Train a DecisionTreeClassifier
        clf = DecisionTreeClassifier()
        y_train = np.squeeze(train_data[:, 2:])

        clf.fit(train_data[:, :2], y_train)
        num_nodes[idx] = clf.tree_.node_count

        # Calculate the classification error
        y_pred = clf.predict(test_set[:, :2])
        y_true = np.squeeze(test_set[:, 2:])
        error = 1 - accuracy_score(y_true, y_pred)
        errors[idx] = error
        print(f"Number of nodes is: {num_nodes[idx]} \n Error on the test set is: " +
              str(error))

    # Plot test error vs number of training examples
    plt.figure()
    plt.plot(num_nodes, errors)
    plt.xlabel('Number of nodes')
    plt.ylabel('Test error')
    plt.title('Test error vs number of nodes')
    plt.savefig('./images/' + dataDict[idx] + '_3.png')
    plt.show()

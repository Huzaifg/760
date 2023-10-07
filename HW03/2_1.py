import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from knn import NearestNeighborClassifier


def plotD2z(data):
    # grid for plotting x -> -2 to 2 in 0,1 steps
    x_1 = np.arange(-2, 2.1, 0.1)
    x_2 = np.arange(-2, 2.1, 0.1)
    # Create meshgrid
    xx, yy = np.meshgrid(x_1, x_2)

    # Create classifier
    clf = NearestNeighborClassifier()
    clf.fit(data[:, 0:2], data[:, 2], k=1)

    # Test data is the meshgrid
    test_data = np.column_stack((xx.ravel(), yy.ravel()))
    y_pred = clf.predict(test_data)

    # Plot the decision boundary by allocating a color to each point in the mesh
    y_pred = y_pred.reshape(xx.shape)
    plt.scatter(xx, yy, s=1, c=y_pred, cmap='Set1')
    plt.scatter(data[:, 0], data[:, 1], c='gray')
    plt.savefig('./images/D2z.png')
    plt.show()


if __name__ == "__main__":
    # Read the data
    data = np.genfromtxt('./data/D2z.txt', delimiter=' ')

    ### Q1 ###
    # Plot the data
    plotD2z(data)

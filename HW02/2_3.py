from decisionTree import DecisionTreeNode, train_tree, evaluate_tree
import matplotlib.pyplot as plt
import numpy as np

# training data
D = np.loadtxt("data/D1.txt", delimiter=" ")
root_node = train_tree(D)

Dtest = np.loadtxt("data/Druns.txt", delimiter=" ")
# Loop over training data and evaluate the tree
error = 0
for row in range(Dtest.shape[0]):
    # print("True label: ", D[row, 2])
    # print("Predicted label: ", evaluate_tree(root_node, D[row, :2]))
    # Accumulate the error
    if Dtest[row, 2] != evaluate_tree(root_node, Dtest[row, :2]):
        error += 1
    # print("")

print(error)
print(error/Dtest.shape[0])

from decisionTree import DecisionTreeNode, train_tree, evaluate_tree
import matplotlib.pyplot as plt
import numpy as np

# Small training data
D = np.array([[0, 0, 1],
              [1, 0, 1],
              [1, 1, 1],
              [0, 1, 0]])


# Train tree
rootnode = train_tree(D)


# Show that tree has no children
print(rootnode.children)


# Plot D

plt.scatter(D[:, 0], D[:, 1], c=D[:, 2])
plt.savefig('./images/2.png')
plt.show()




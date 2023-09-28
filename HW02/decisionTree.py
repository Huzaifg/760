from PIL import Image
import pygraphviz as pgv
import numpy as np
import sys
import pandas as pd


class DecisionTreeNode:
    def __init__(self):
        self.is_leaf = False
        self.split_feature = None
        self.split_threshold = None
        self.children = {}  # Dictionary to store child nodes
        self.class_label = None

        # -----------------------------------------------
        # Printing stuff required just for the assignment
        # -----------------------------------------------
        self.candidate_splits_ = {0: [], 1: []}


def DetermineCandidateSplits(D):
    # Check if data is a nonempty numpy array
    assert isinstance(D, np.ndarray) and D.size > 0

    # Check if the data has 3 columns (2 features and 1 label)
    assert D.shape[1] == 3

    # For each column in the data (for each feature), determine the candidate split values
    candidate_splits = [None] * (D.shape[1] - 1)
    for column in range(D.shape[1] - 1):
        candidate_splits[column] = []
        # Sort entire data by current column
        D_sort = D[D[:, column].argsort()]
        for row in range(D_sort.shape[0] - 1):
            # Add to candidate split if the class label is different
            if D_sort[row, 2] != D_sort[row + 1, 2]:
                # Split based on current point to ensure c \in D
                candidate_splits[column].append(D_sort[row, column])

    return candidate_splits


def GainRatio(D, split, whichFeature, splitEntropy):

    # Make sure whichFeature is valid (only two features)
    assert whichFeature == 0 or whichFeature == 1

    # Split the data based on the split value
    D_split_then = D[D[:, whichFeature] >= split]
    D_split_else = D[D[:, whichFeature] < split]

    # ---------------------------------------------------------------------------------------------------
    # In each split, get the entropy of the class labels given the splits happened (H_D(Y|SPLIT = split))
    # ---------------------------------------------------------------------------------------------------

    # Count the number of labels in the "then" branch
    y0Labels_then = D_split_then[D_split_then[:, 2] == 0].shape[0]
    y1Labels_then = D_split_then[D_split_then[:, 2] == 1].shape[0]

    # Calculate the emperical probability of each class label in the "then" branch
    prob_y0labels_then = y0Labels_then / D_split_then.shape[0]
    prob_y1labels_then = y1Labels_then / D_split_then.shape[0]

    # Count the number of labels in the "else" branch
    y0Labels_else = D_split_else[D_split_else[:, 2] == 0].shape[0]
    y1Labels_else = D_split_else[D_split_else[:, 2] == 1].shape[0]

    # Calculate the emperical probability of each class label in the "else" branch
    prob_y0labels_else = y0Labels_else / D_split_else.shape[0]
    prob_y1labels_else = y1Labels_else / D_split_else.shape[0]

    # Calculate the entropy of the class labels in each branch
    if prob_y0labels_then != 0 and prob_y1labels_then != 0:
        entropy_then = -prob_y0labels_then * \
            np.log2(prob_y0labels_then) - prob_y1labels_then * \
            np.log2(prob_y1labels_then)
    else:
        entropy_then = 0
    if prob_y0labels_else != 0 and prob_y1labels_else != 0:
        entropy_else = -prob_y0labels_else * \
            np.log2(prob_y0labels_else) - prob_y1labels_else * \
            np.log2(prob_y1labels_else)
    else:
        entropy_else = 0

    # ------------------------------------------------------------------------
    # Calculate the entropy of the class labels given the split (H_D(Y|SPLIT))
    # ------------------------------------------------------------------------

    # Ratio of data set with feature value >= split
    prob_then = D_split_then.shape[0] / D.shape[0]

    # Ratio of data set with feature value < split
    prob_else = D_split_else.shape[0] / D.shape[0]

    entropy = prob_then * entropy_then + prob_else * entropy_else

    # ---------------------------------------------
    # Calculate the entropy of the labels (H_D(Y))
    # ---------------------------------------------

    # Probablity of labels = 0
    prob_y0labels = D[D[:, 2] == 0].shape[0] / D.shape[0]

    # Probablity of labels = 1
    prob_y1labels = D[D[:, 2] == 1].shape[0] / D.shape[0]

    entropy_labels = -prob_y0labels * \
        np.log2(prob_y0labels) - prob_y1labels * np.log2(prob_y1labels)

    # ------------------------
    # Calculate the gain ratio
    # ------------------------

    gain_ratio = (entropy_labels - entropy) / splitEntropy

    return (gain_ratio, entropy_labels - entropy)


def GetSplitEntropy(D, split, whichFeature):

    # Make sure whichFeature is valid (only two features)
    assert whichFeature == 0 or whichFeature == 1

    # Split the data based on the split value
    D_split_then = D[D[:, whichFeature] >= split]
    D_split_else = D[D[:, whichFeature] < split]

    # Get size of each split
    size_ratio_then = D_split_then.shape[0] / D.shape[0]
    size_ratio_else = D_split_else.shape[0] / D.shape[0]

    if (size_ratio_then == 0 or size_ratio_else == 0):
        return 0

    # Get entropy of each split
    entropy = -size_ratio_else * \
        np.log2(size_ratio_else) - size_ratio_then * np.log2(size_ratio_then)

    return entropy


def StoppingCriteria(D, C):

    # Stop if node is empty
    if D.size == 0:
        return True

    # -----------------------------------------------------------------------
    # Stop if all splits have zero gain ratio or all splits have zero entropy
    # -----------------------------------------------------------------------
    for whichFeature, feature in enumerate(C):
        for split in feature:
            splitEntropy = GetSplitEntropy(D, split, whichFeature)
            # If we have any split that has a non-zero entropy and non-zero gain ratio, then we can't stop
            if (splitEntropy != 0):
                gainRatio, infoGain = GainRatio(
                    D, split, whichFeature, splitEntropy)
                if (gainRatio != 0):
                    return False

    return True


def FindBestSplit(D, C):

    # Loop over all splits in C where C is a dictionary of candidate splits
    best_split = None
    corresponding_feature = None
    best_gain_ratio = 0
    for whichFeature, feature in enumerate(C):
        for split in feature:
            splitEntropy = GetSplitEntropy(D, split, whichFeature)
            # Ensure we don't have a 0 or nan split entropy
            if (splitEntropy == 0 or np.isnan(splitEntropy)):
                continue
            # Calculate gain ratio for split
            gain_ratio, info_gain = GainRatio(
                D, split, whichFeature, splitEntropy)
            # Update best split if necessary
            if gain_ratio > best_gain_ratio:
                best_split = split
                corresponding_feature = whichFeature
                best_gain_ratio = gain_ratio

    return (corresponding_feature, best_split)


def MakeSubtree(D, printIt, level=1):
    node = DecisionTreeNode()
    # Get candidate splits and use it everywhere
    C = DetermineCandidateSplits(D)
    if StoppingCriteria(D, C):
        node.is_leaf = True
        # Determine class label or probabilities for leaf node
        node.class_label = DetermineClassLabel(D)
    else:
        # Only needed for assignment
        if (printIt):
            # Only for root node
            if level == 1:
                for whichFeature, feature in enumerate(C):
                    for split in feature:
                        splitEntropy = GetSplitEntropy(D, split, whichFeature)
                        # Ensure we don't have a 0 or nan split entropy
                        if (splitEntropy == 0 or np.isnan(splitEntropy)):
                            node.candidate_splits_[whichFeature].append((
                                split, 0, 0, 0))
                            continue
                        gain_ratio, info_gain = GainRatio(
                            D, split, whichFeature, splitEntropy)
                        node.candidate_splits_[whichFeature].append((
                            split, info_gain, gain_ratio, splitEntropy))

        node.is_leaf = False
        split = FindBestSplit(D, C)
        node.split_feature, node.split_threshold = split
        for outcome in [0, 1]:  # Binary classification
            Dk = SubsetInstances(D, node.split_feature,
                                 node.split_threshold, outcome)
            node.children[outcome] = MakeSubtree(Dk, level+1)

    return node


def DetermineClassLabel(D):
    # Determine the class label for the leaf node using majority voting

    # Count the number of labels
    y0Labels = D[D[:, 2] == 0].shape[0]
    y1Labels = D[D[:, 2] == 1].shape[0]

    # Return the majority label - when there is no majority, return 1
    if y0Labels > y1Labels:
        return 0
    else:
        return 1


def SubsetInstances(D, feature, threshold, outcome):
    # Subset the data based on the feature and threshold
    if outcome == 0:
        D_subset = D[D[:, feature] >= threshold]
    else:
        D_subset = D[D[:, feature] < threshold]

    return D_subset


def train_tree(D, printIt):
    """
    Train a decision tree

    Args:
    - D: A numpy array (n x 3) representing the training data. The first two columns
         are the features and the last column is the class label.

    Returns:
    - A DecisionTreeNode representing the root of the decision tree.
    """
    root_node = MakeSubtree(D, printIt)
    return root_node


def evaluate_tree(node, new_data_point):
    """
    Evaluate a decision tree for a new data point.

    Args:
    - node: The current node to evaluate from (start with the root node).
    - new_data_point: A list or array containing feature values for the new data point.

    Returns:
    - The predicted class label for the new data point.
    """
    if node.is_leaf:
        return node.class_label  # Leaf node reached, return the predicted class label
    else:
        # Check if the new data point is a pandas series
        if isinstance(new_data_point, pd.Series):
            # Convert to numpy array
            new_data_point = new_data_point.to_numpy()
        # Check the feature condition and move to the appropriate child node
        feature_value = new_data_point[node.split_feature]
        if feature_value >= node.split_threshold:
            # "then" branch
            return evaluate_tree(node.children[0], new_data_point)
        else:
            return evaluate_tree(node.children[1], new_data_point)


# Count the nodes for debugging
def CountNodes(node):
    if node is None:
        return 0

    # Initialize count with 1 (for the current node)
    count = 1

    if not node.is_leaf:
        # If it's an internal node, recursively count child nodes
        for outcome in node.children:
            count += CountNodes(node.children[outcome])

    return count


# Print all candidate split infromation for a node
def PrintNodeInfo(node):
    if node is None:
        return

    if not node.is_leaf:
        # loop over features
        for feature in node.candidate_splits_:
            print("Feature: ", feature)
            # loop over candidate splits
            for whichSplit, info in enumerate(node.candidate_splits_[feature]):
                print("\tSplit: ", info[0])
                print("\tGain ratio: ", node.candidate_splits_[
                      feature][whichSplit][2])
                print("\tInformation gain: ", node.candidate_splits_[
                      feature][whichSplit][1])
                print("\tSplit entropy: ", node.candidate_splits_[
                      feature][whichSplit][3])
                print("")


# Print the tree
def printTree(node, depth=0):
    # Check if node is a leaf
    if node.is_leaf:
        print("\t"*depth, "Label: ", node.class_label)
    else:
        # Print the decision label
        print("\t"*depth, "x0" if node.split_feature ==
              0 else "x1", ">= ", node.split_threshold)
        # Go down the "then" branch
        printTree(node.children[0], depth+1)
        # Go down the "else" branch
        printTree(node.children[1], depth+1)


def plot_decision_tree(root_node, filename):
    # Create a new Graphviz graph
    graph = pgv.AGraph(strict=True, directed=True)

    def add_node(node, parent=None):
        # Create a unique node ID based on the object's memory address
        node_id = str(id(node))

        # Determine label for the node
        if node.is_leaf:
            label = f"Class Label: {node.class_label}"
        else:
            label = f"Split Feature: {node.split_feature}\nSplit Threshold: {node.split_threshold}"

        # Add the node to the graph
        graph.add_node(node_id, label=label)

        # Connect the node to its parent (if specified)
        if parent is not None:
            graph.add_edge(parent, node_id)

        # Recursively add child nodes
        for outcome, child_node in node.children.items():
            add_node(child_node, node_id)

    # Start adding nodes from the root
    add_node(root_node)

    # Render the graph to a file (e.g., in PNG format)
    graph.draw(filename, format='png', prog='dot')


if __name__ == "__main__":

    dataset = sys.argv[1]
    print(f"-------------------- Training on {dataset} --------------------")
    D = np.loadtxt("data/" + dataset + ".txt", delimiter=" ")

    # Flag of whether we want to print node information
    printIt = int(sys.argv[3])
    root_node = train_tree(D, printIt)
    print(f"Tree has {CountNodes(root_node)} nodes")

    dataset_test = sys.argv[2]
    print(
        f"-------------------- Testing on {dataset_test} --------------------")
    Dtest = np.loadtxt("data/" + dataset_test + ".txt", delimiter=" ")

    # Loop over training data and evaluate the tree
    error = 0
    for row in range(Dtest.shape[0]):
        if Dtest[row, 2] != evaluate_tree(root_node, Dtest[row, :2]):
            error += 1

    print(f"{error} misclassified points out of {Dtest.shape[0]}")
    print(f"Accuracy {(1 - error/Dtest.shape[0])*100} %")

    output_filename = './images/' + dataset + '.png'
    plot_decision_tree(root_node, output_filename)

    # Display the generated image
    img = Image.open(output_filename)
    img.show()

    if (printIt):
        PrintNodeInfo(root_node)

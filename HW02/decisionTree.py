import numpy as np

class DecisionTreeNode:
    def __init__(self):
        self.is_leaf = False
        self.split_feature = None
        self.split_threshold = None
        self.children = {}  # Dictionary to store child nodes
        self.class_label = None


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


def GainRatio(D, split, whichFeature):

    # Evaluate the denominator which I call "SplitEntropy"
    splitEntropy = GetSplitEntropy(D, split, whichFeature)

    # Ensure we don't have a 0 or nan split entropy
    if(splitEntropy == 0 or np.isnan(splitEntropy)): return 0

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

    # Check if any of the probabilities are 0 - if so, then we can't calculate the entropy
    if prob_y0labels_then == 0 or prob_y1labels_then == 0 or prob_y0labels_else == 0 or prob_y1labels_else == 0:
        return 0

    # Calculate the entropy of the class labels in each branch
    entropy_then = -prob_y0labels_then * np.log2(prob_y0labels_then) - prob_y1labels_then * np.log2(prob_y1labels_then)
    entropy_else = -prob_y0labels_else * np.log2(prob_y0labels_else) - prob_y1labels_else * np.log2(prob_y1labels_else)

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

    entropy_labels = -prob_y0labels * np.log2(prob_y0labels) - prob_y1labels * np.log2(prob_y1labels)

    # ------------------------
    # Calculate the gain ratio
    # ------------------------

    gain_ratio = (entropy_labels - entropy) / splitEntropy

    return gain_ratio

def GetSplitEntropy(D, split, whichFeature):

    # Make sure whichFeature is valid (only two features)
    assert whichFeature == 0 or whichFeature == 1

    # Split the data based on the split value
    D_split_then = D[D[:, whichFeature] >= split]
    D_split_else = D[D[:, whichFeature] < split]

    # Get size of each split
    size_ratio_then = D_split_then.shape[0] / D.shape[0]
    size_ratio_else = D_split_else.shape[0] / D.shape[0]

    if(size_ratio_then == 0 or size_ratio_else == 0): return 0

    # Get entropy of each split
    entropy = -size_ratio_else * np.log2(size_ratio_else) - size_ratio_then * np.log2(size_ratio_then)

    return entropy

def StoppingCriteria(D, C):

    # Stop if node is empty
    if D.size == 0:
        return True
    
    # -----------------------------------------------------------------------
    # Stop if all splits have zero gain ratio or all splits have zero entropy
    # -----------------------------------------------------------------------

    # all_splits_zeroGain = True
    # all_splits_zeroEntropy = True
    # Loop over all candidate splits
    for whichFeature,feature in enumerate(C):
        for split in feature:
            # If we have any split that has a non-zero entropy and non-zero gain ratio, then we can't stop
            if(GetSplitEntropy(D, split, whichFeature) != 0):
                if(GainRatio(D, split, whichFeature) != 0):
                    return False
                # all_splits_zeroEntropy = False

            # If we have any split that has non-zero gain ratio, then we can't stop
            # if(GainRatio(D, split, whichFeature) != 0):
                # all_splits_zeroGain = False
                # return False

    return True
    # return (all_splits_zeroGain or all_splits_zeroEntropy)
    


def FindBestSplit(D, C):

    # Loop over all splits in C where C is a dictionary of candidate splits
    best_split = None
    corresponding_feature = None
    best_gain_ratio = 0
    for whichFeature,feature in enumerate(C):
        for split in feature:
            # Calculate gain ratio for split
            gain_ratio = GainRatio(D, split,whichFeature)
            # Update best split if necessary
            if gain_ratio > best_gain_ratio:
                best_split = split
                corresponding_feature = whichFeature
                best_gain_ratio = gain_ratio


    return (corresponding_feature, best_split)

def MakeSubtree(D):
    node = DecisionTreeNode()
    # Get candidate splits and use it everywhere
    C = DetermineCandidateSplits(D)
    if StoppingCriteria(D,C):
        node.is_leaf = True
        # Determine class label or probabilities for leaf node
        node.class_label = DetermineClassLabel(D)
    else:
        node.is_leaf = False
        split = FindBestSplit(D, C)
        node.split_feature, node.split_threshold = split
        for outcome in [0, 1]:  # Binary classification
            Dk = SubsetInstances(D, node.split_feature, node.split_threshold, outcome)
            node.children[outcome] = MakeSubtree(Dk)
    
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

def train_tree(D):
    """
    Train a decision tree

    Args:
    - D: A numpy array (n x 3) representing the training data. The first two columns
         are the features and the last column is the class label.

    Returns:
    - A DecisionTreeNode representing the root of the decision tree.
    """
    root_node = MakeSubtree(D)
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
        # Check the feature condition and move to the appropriate child node
        feature_value = new_data_point[node.split_feature]
        if feature_value >= node.split_threshold:
            return evaluate_tree(node.children[0], new_data_point)  # "then" branch
        else:
            return evaluate_tree(node.children[1], new_data_point)

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


if __name__ == "__main__":
    # Load your training data here as D from file data/D1.txt
    D = np.loadtxt("data/D1.txt", delimiter=" ")
    root_node = train_tree(D)

    print(CountNodes(root_node))

    print(root_node.split_feature,root_node.split_threshold)

    Dtest = np.loadtxt("data/D2.txt", delimiter=" ")
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
    print(error/D.shape[0])


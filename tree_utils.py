#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 19:20:00 2020

@author: francesco
"""
import numpy as np
from sklearn.tree._tree import TREE_LEAF


class NewTree(object):
    def __init__(self, estimator):

        self.n_nodes = np.array(estimator.tree_.node_count)
        self.children_left = np.array(estimator.tree_.children_left)
        self.children_right = np.array(estimator.tree_.children_right)
        self.features = np.array(estimator.tree_.feature)
        self.thresholds = np.array(estimator.tree_.threshold)
        # self.labels = np.array(estimator.tree_.value.argmax(axis=2).ravel())
        labels_idxs = np.array(estimator.tree_.value.argmax(axis=2).ravel())
        self.labels = []
        for idx in labels_idxs:
            self.labels.append(estimator.classes_[idx])
        self.nodes = None

    def build_tree(self):
        nodes = []
        for node in range(self.n_nodes):
            if (len(np.argwhere(self.children_right == node)) == 0) and (
                    len(np.argwhere(self.children_left == node)) == 0):
                idxancestor = None
            else:
                if len(np.argwhere(self.children_right == node)) != 0:
                    idxancestor = np.argwhere(self.children_right == node).ravel()[0]
                else:
                    idxancestor = np.argwhere(self.children_left == node).ravel()[0]
            new_node = Node(node,
                            self.children_left[node],
                            self.children_right[node],
                            idxancestor,
                            self.features[node],
                            self.thresholds[node],
                            self.labels[node])
            nodes.append(new_node)
        for node in nodes:
            node.left = nodes[node.idxleft] if node.idxleft != -1 else None
            node.right = nodes[node.idxright] if node.idxright != -1 else None
            node.ancestor = nodes[node.idxancestor] if node.idxancestor is not None else None
        self.nodes = nodes
        return self


class Node:
    def __init__(self, idx, idxleft, idxright, idxancestor, feature, threshold, label):
        self.idx = idx
        self.idxleft = idxleft
        self.idxright = idxright
        self.idxancestor = idxancestor
        self.feature = feature
        self.threshold = threshold
        self.label = label
        self.left = None
        self.right = None
        self.ancestor = None


# Find closest leaf to the given 
# node x in a tree 

# Utility class to create a new node 

# This function finds closest leaf to 
# root. This distance is stored at *minDist. 
def findLeafDown(root, lev, minDist, minidx, x):
    # base case 
    if (root == None):
        return

    # If this is a leaf node, then check if 
    # it is closer than the closest so far 
    if (root.left == None and
        root.right == None) and root.label != x.label:
        if (lev < (minDist[0])) and lev > 0:
            minDist[0] = lev
            minidx[0] = root.idx
        return

    # Recur for left and right subtrees 
    findLeafDown(root.left, lev + 1, minDist, minidx, x)
    findLeafDown(root.right, lev + 1, minDist, minidx, x)


# This function finds if there is 
# closer leaf to x through parent node. 
def findThroughParent(root, x, minDist, minidx):
    # Base cases 
    if (root == None):
        return -1
    if (root == x):
        return 0

    # Search x in left subtree of root 
    l = findThroughParent(root.left, x,
                          minDist, minidx)

    # If left subtree has x 
    if (l != -1):
        # Find closest leaf in right subtree 
        findLeafDown(root.right, l + 2, minDist, minidx, x)
        return l + 1

    # Search x in right subtree of root 
    r = findThroughParent(root.right, x, minDist, minidx)

    # If right subtree has x 
    if (r != -1):
        # Find closest leaf in left subtree 
        findLeafDown(root.left, r + 2, minDist, minidx, x)
        return r + 1

    return -1


# Returns minimum distance of a leaf
# from given node x 
def minimumDistance(root, x):
    # Initialize result (minimum 
    # distance from a leaf) 
    minDist = [np.inf]

    minidx = [None]

    # Find closest leaf down to x 
    findLeafDown(x, 0, minDist, minidx, x)

    # See if there is a closer leaf 
    # through parent 
    findThroughParent(root, x, minDist, minidx)

    return minDist[0], minidx[0]


def get_root_leaf_path(node):
    path = []
    features = []
    labels = []
    thresholds = []
    while node is not None:
        path.append(node.idx)
        features.append(node.feature)
        labels.append(node.label)
        thresholds.append(node.threshold)
        node = node.ancestor

    return {"path": path[::-1],
            "features": features[::-1],
            "labels": labels[::-1],
            "thresholds": thresholds[::-1],
            "thresholds_signs": None
            }


def get_branch_length(node):
    count = -1
    while node is not None:
        count += 1
        node = node.ancestor
    return count


def get_thresholds_signs(dtree, root_leaf_path):
    thresholds_signs = []
    for i, node_idx in enumerate(root_leaf_path["path"][:-1]):
        node = dtree.nodes[node_idx]
        if node.left.idx == root_leaf_path["path"][i + 1]:
            thresholds_signs.append("not-contained")
        else:
            thresholds_signs.append("contained")
    root_leaf_path["thresholds_signs"] = thresholds_signs
    return root_leaf_path


def prune_index(inner_tree, decisions, index=0):
    # Start pruning from the bottom - if we start from the top, we might miss
    # nodes that become leaves during pruning.
    # Do not use this directly - use prune_duplicate_leaves instead.
    if not is_leaf(inner_tree, inner_tree.children_left[index]):
        prune_index(inner_tree, decisions, inner_tree.children_left[index])
    if not is_leaf(inner_tree, inner_tree.children_right[index]):
        prune_index(inner_tree, decisions, inner_tree.children_right[index])

    # Prune children if both children are leaves now and make the same decision:
    if (is_leaf(inner_tree, inner_tree.children_left[index]) and
            is_leaf(inner_tree, inner_tree.children_right[index]) and
            (decisions[index] == decisions[inner_tree.children_left[index]]) and
            (decisions[index] == decisions[inner_tree.children_right[index]])):
        # turn node into a leaf by "unlinking" its children
        inner_tree.children_left[index] = TREE_LEAF
        inner_tree.children_right[index] = TREE_LEAF
        # print("Pruned {}".format(index))


def prune_duplicate_leaves(dt):
    # Remove leaves if both
    decisions = dt.tree_.value.argmax(axis=2).flatten().tolist()  # Decision for each node
    prune_index(dt.tree_, decisions)


def is_leaf(inner_tree, index):
    # Check whether node is leaf node
    return (inner_tree.children_left[index] == TREE_LEAF and
            inner_tree.children_right[index] == TREE_LEAF)

def coverage_score(n_lhs, n):
    return n_lhs / n


def coverage_score_tree_old(dt, X, x):
    x_leave_id = dt.apply(x)[0]
    X_leave_ids = dt.apply(X)
    n_lhs = (X_leave_ids == x_leave_id).sum()
    n = len(X_leave_ids)
    return coverage_score(n_lhs, n)


def coverage_score_tree(dt, leaf_id):
    n = dt.tree_.value[0].sum()
    n_lhs = dt.tree_.value[leaf_id].sum()
    return coverage_score(n_lhs, n)


def precision_score(n_y, n_lhs):
    return n_y / n_lhs


def precision_score_tree(dt, X, y, leaf_id):
    y_surrogate = dt.predict(X)
    X_leave_ids = dt.apply(X)
    idxs = np.argwhere(X_leave_ids == leaf_id)  # indexes of records of X in the leaf leaf_id
    n_y = (y[idxs] == y_surrogate[idxs]).sum()  # n of records of the leaf leaf_id that are correctly classified
    n_lhs = len(idxs)
    return precision_score(n_y, n_lhs)


def precision_score_tree2(dt, x):
    x_leave_id = dt.apply(x)[0]


if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from sklearn import tree

    X, y = load_iris(return_X_y=True)
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X, y)
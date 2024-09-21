import numpy as np
import copy
from tree_utils import check_same_label, find_most_common_label, data_separate, best_split
from feature_def import Feature, Column, Numeric_Attributes

def ID3(S_idx, listA, depth, max_depth, train_data, info_gain):
    S = copy.deepcopy(S_idx)
    A = copy.deepcopy(listA)
    if S.shape[0] == 0:
        return -1
    else:
        first_label, same_label = check_same_label(S, train_data)
        if same_label:
            return first_label
        elif not A or depth == max_depth:
            return find_most_common_label(S, train_data)
        else:
            root = {}
            subtree = {}
            attr = best_split(S, A, train_data, info_gain)
            V = data_separate(S, attr, train_data)
            A.remove(attr)
            if attr in Numeric_Attributes:
                for i, v in enumerate(V):
                    if v.shape[0] != 0:
                        result = ID3(v, A, depth+1, max_depth, train_data, info_gain)
                        subtree[i] = result if result != -1 else find_most_common_label(S, train_data)
                    else:
                        subtree[i] = find_most_common_label(S, train_data)
            else:
                for i, v in enumerate(V):
                    if v.shape[0] != 0:
                        result = ID3(v, A, depth+1, max_depth, train_data, info_gain)
                        subtree[Feature[attr][i]] = result if result != -1 else find_most_common_label(S, train_data)
                    else:
                        subtree[Feature[attr][i]] = find_most_common_label(S, train_data)
            root[attr] = subtree
            return root

def predict(data, Tree):
    if isinstance(Tree, dict):
        f = list(Tree.keys())[0]
        ind = Column.index(f)
        if f in Numeric_Attributes:
            T = Tree[f][int(data[ind])]
        else:
            T = Tree[f][data[ind]]
        return predict(data, T)
    else:
        return Tree

def compute_error(data, tree):
    num = len(data)
    count = sum(1 for i in range(num) if predict(data[i], tree) != data[i][-1])
    return count / num
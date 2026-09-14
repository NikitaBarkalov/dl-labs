import numpy as np
from sklearn.datasets import load_iris

def load_data():
    iris_data = load_iris()
    data = iris_data['data']
    target = iris_data['target']

    return data, target

def split_data(data, target, train_proportion, rng):
    classes = np.unique(target)
    train_by_class = int(len(target) / len(classes) * train_proportion)
    train_indices_by_classes, test_indices_by_classes = [], []

    for target_class in np.unique(target):
        indices = np.where(target == target_class)[0]
        permutated_indices = rng.permutation(indices)
        train_indices_by_classes.append(permutated_indices[:train_by_class])
        test_indices_by_classes.append(permutated_indices[train_by_class:])

    train_indices = np.concatenate(train_indices_by_classes)
    test_indices = np.concatenate(test_indices_by_classes)

    train_data, train_target = data[train_indices, :], target[train_indices]
    test_data, test_target = data[test_indices, :], target[test_indices]

    return train_data, train_target, test_data, test_target

def standardize_data(train_data, test_data):
    train_mean = train_data.mean(axis = 0)
    train_std = train_data.std(axis = 0, ddof = 0)

    standardized_train = (train_data - train_mean) / train_std
    standardized_test = (test_data - train_mean) / train_std

    return standardized_train, standardized_test
    
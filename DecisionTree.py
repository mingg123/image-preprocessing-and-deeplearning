
from sklearn.tree import DecisionTreeClassifier
import numpy as np


import tensorflow as tf  #내가추가한것
def DecisionTree(csv_path):
    xy = np.loadtxt(csv_path, delimiter=',', dtype=np.float32)

    x_train = xy[:, 0:-1]
    y_train = xy[:, [-1]]

    x_test = []
    x_test.append(x_train[len(x_train) - 1])
    x_test = np.array(x_test)

    x_test = x_test[:, 0:20]

    estimator = DecisionTreeClassifier(criterion='gini', max_depth=None, max_leaf_nodes=None, min_samples_split=2, min_samples_leaf=1, max_features=None)

    estimator.fit(x_train, y_train)

    y_predict = estimator.predict(x_test)

    return y_predict
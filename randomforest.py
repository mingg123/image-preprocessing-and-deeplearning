import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv
import random
import math
import operator
import numpy as np

# Importing the datasets

datasets = pd.read_csv(r'D:\final\data10.csv')
X = datasets.iloc[:, [0,19]].values
Y = datasets.iloc[:, 20].values

# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split
X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size = 0.1, random_state = 0)

# Feature Scaling

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_Train = sc_X.fit_transform(X_Train)
X_Test = sc_X.transform(X_Test)

# Fitting the classifier into the Training set

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 200, criterion = 'entropy', random_state = 0)
classifier.fit(X_Train,Y_Train)

# Predicting the test set results

Y_Pred = classifier.predict(X_Test)

# Making the Confusion Matrix

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_Test, Y_Pred)

# Visualising the Training set results


def getAccuracy(testSet, predictions):
    for i in range(len(testSet)):
        print(testSet[i], predictions[i])

    # 암인경우(1)를 True, 암이 아닌경우(0)를 False
    TP = 0
    FP = 0
    FN = 0
    TN = 0

    for x in range(len(testSet)):
        if(predictions[x] == 0 and testSet[x] == 0):
            TN += 1

        if(predictions[x] == 0 and testSet[x] == 1):
            FN += 1

        if(predictions[x] == 1 and testSet[x] == 0):

            FP += 1

        if(predictions[x] == 1 and testSet[x] == 1):

            TP += 1

    print("TP:", TP, "   FP:", FP, "   FN:", FN, "   TN:", TN)
    accuracy = ((TP+TN)/(TP+TN+FP+FN)) * 100.0
    sensitivity = TP/(TP+FN) * 100.0
    specificity = TN/(TN+FP) * 100.0
    precision = TP/(TP+FP) * 100.0


    return accuracy, sensitivity, specificity, precision

accuracy, sensitivity, specificity, precision = getAccuracy(Y_Test, Y_Pred)

print("accuracy:", accuracy)
print("recall:", sensitivity)
print("specificity:", specificity)
print("precision:", precision)
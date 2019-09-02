import csv
import random
import math
import operator
import numpy as np

def loadDataset(ehd_path):
    trainingSet = []
    testSet = []

    with open(ehd_path, 'r') as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        for x in range(len(dataset) - 1):
            for y in range(20):          # 4개의 vector => 우리는 20개의 vector
                #print(dataset[x][y])
                dataset[x][y] = float(dataset[x][y])

        for z in range(len(dataset)-2):
            trainingSet.append(dataset[z])

        testSet.append(dataset[len(dataset)-1])

    return trainingSet, testSet

def euclideanDistance(instance1, instance2, length):
    distance = 0
    for x in range(length-1):
        #print(type(instance1[x]), type(instance2[x]))
        distance += pow((int(instance1[x]) - int(instance2[x])), 2)

    return math.sqrt(distance)


def getNeighbors(trainingSet, testInstance, k):
    distances = []
    length = len(testInstance) - 1
    for x in range(len(trainingSet)):
        dist = euclideanDistance(testInstance, trainingSet[x], length)
        distances.append((trainingSet[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors


def getResponse(neighbors):
    classVotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1] #마지막열에 '라벨'만을 뽑아내어 response 변수에 저장
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]

def knn(ehd):
    # prepare data
    trainingSet, testSet = loadDataset(ehd)  # cancer가 0
    #print('Train set: ' + repr(len(trainingSet)))
    #print('Test set: ' + repr(len(testSet)))
    # generate predictions

    #num = 0
    #for x in range(len(trainingSet)):
    #    for y in range(len(testSet)):
    #        if (np.array_equal(testSet[y], trainingSet[x]) == True):
    #            num = num + 1

    #print("Train이 Test로 써진 경우의 수:", num)

    k = 5
    neighbors = getNeighbors(trainingSet, testSet[0], k)
    prediction = getResponse(neighbors)

    cancer_per = 0
    benign_per = 0

    for y in range(k):
        neighbor_list = neighbors[y]

        type = neighbor_list[len(neighbor_list)-1]

        if(type == '1'):
            cancer_per += 1
        else:
            benign_per += 1

    return int(cancer_per), int(benign_per)

        #print('> predicted=' + repr(result) + ', actual=' + repr(testSet[x][-1]))

        #if(result != testSet[x][-1] and testSet[x][-1] == '1'):
        #    print("Show neighbors of below:")
        #    print(testSet[x])
        #    for y in range(k):
        #        print("Neighbor" + str(y+1) + "'s")
        #        print(neighbors[y])

    #accuracy, sensitivity, specificity, precision = getAccuracy(testSet, predictions)

    #print('Accuracy: ' + repr(accuracy) + '%')
    #print('Sensitivity: ' + repr(sensitivity) + '%')
    #print('Specificity: ' + repr(specificity) + '%')
    #print('Precision: ' + repr(precision) + '%')

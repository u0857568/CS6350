import csv
from dataclasses import dataclass
import numpy as np

@dataclass
class GammaSchedule:
    gamma0: float
    d: float

def parseCSV(csvFilePath, zero2neg):
    x = []
    y = []

    with open(csvFilePath, 'r') as csvFile:
        csvReader = csv.reader(csvFile, delimiter=',')

        for row in csvReader:
            thisExample = [float(row[i]) for i in range(len(row)-1)]
            thisExample += [1.0]
            x.append(thisExample)
            if zero2neg:
                y.append(2*float(row[-1]) - 1)
            else:
                y.append(float(row[-1]))

    x = np.matrix(x)
    y = np.array(y)
    return x, y


def sigmoid(x):
    from math import exp
    if x >= 700:
        return 1
    elif x <= -700:
        return 0
    return 1 / (1 + exp(-1*x))

def LogisticRegression_SGD_MAP(x, y, T, v, GammaSchedule, checkConverge):
    import math
    wghts = np.zeros((1,x.shape[1]))
    idxs = np.arange(x.shape[0])
    gamma = GammaSchedule.gamma0
    iterations = 1

    lossList = []
    for epoch in range(T):
        
        np.random.shuffle(idxs)

        for i in idxs:
            gamma = GammaSchedule.gamma0 / (1 + (GammaSchedule.gamma0 * iterations / GammaSchedule.d))
            dL = np.subtract((1/v)*wghts, (1 - sigmoid(y[i]*np.dot(wghts, x[i].T))) * x.shape[0] * y[i] * x[i])
            wghts = np.subtract(wghts, gamma*dL)

            iterations += 1

        if checkConverge == True:
            lossSum = (1 / (2*v)) * np.dot(wghts, wghts.T)
            for i in idxs:
                lossSum += math.log(1 + math.exp(-y[i] * np.dot(wghts, x[i].T)))
            lossList.append(np.asscalar(lossSum))

    return wghts, lossList


def LogisticRegression_SGD_MAP_predict(x, w):
    predictions = []
    for ex in x:
        p = np.dot(w, ex.T)
        if p < 0:
            predictions.append(-1)
        else:
            predictions.append(1)
    return np.array(predictions)


def LogisticRegression_SGD_ML(x, y, T, GammaSchedule, checkConverge):
    import math
    wghts = np.zeros((1,x.shape[1]))
    idxs = np.arange(x.shape[0])
    gamma = GammaSchedule.gamma0
    iterations = 1

    lossList = []
    for epoch in range(T):
        np.random.shuffle(idxs)

        for i in idxs:
            gamma = GammaSchedule.gamma0 / (1 + (GammaSchedule.gamma0 * iterations / GammaSchedule.d))

            dL = -(1 - sigmoid(y[i]*np.dot(wghts, x[i].T))) * x.shape[0] * y[i] * x[i]
            wghts = np.subtract(wghts, gamma*dL)

            iterations += 1

        if checkConverge == True:
            lossSum = 0
            for i in idxs:
                power = np.asscalar(-y[i] * np.dot(wghts, x[i].T))
                if power > 700: # overflow
                    lossSum += power
                    break
                lossSum += math.log(1 + math.exp(power))
            lossList.append(lossSum)

    return wghts, lossList


def LogisticRegression_SGD_ML_predict(x, w):
    predictions = []
    for ex in x:
        p = np.dot(w, ex.T)
        if p < 0:
            predictions.append(-1)
        else:
            predictions.append(1)
    return np.array(predictions)
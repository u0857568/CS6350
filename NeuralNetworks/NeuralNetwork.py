import csv
from dataclasses import dataclass
import numpy as np

@dataclass
class GammaSchedule:
    gamma0: float
    d: float

class NeuralNet:
    def __init__(self, layers, numInputs, hiddenNodeCount, randInit):
        self.layerCount = layers
        self.layerNodeCounts = np.concatenate([np.array([numInputs]), np.array(hiddenNodeCount)+1, np.array([2])])
        self.nodes = np.zeros((layers, np.amax(self.layerNodeCounts)))
        self.nodes[:,0] = np.ones(layers) 
        self.weights = np.zeros((layers, np.amax(self.layerNodeCounts), np.amax(self.layerNodeCounts)))
        if randInit == True:
            self.weights = np.random.normal(size=(layers,np.amax(self.layerNodeCounts),np.amax(self.layerNodeCounts)))
        self.dweights = np.zeros((layers,np.amax(self.layerNodeCounts),np.amax(self.layerNodeCounts)))
        self.y = None

def parseCSV(csvFilePath, zero2neg):
    x = []
    y = []

    with open(csvFilePath, 'r') as csvFile:
        csvReader = csv.reader(csvFile, delimiter=',')

        for row in csvReader:
            thisExample = [1.0]
            thisExample += [float(row[i]) for i in range(len(row)-1)]
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
    return 1 / (1 + exp(-1*x))

def sigmoid_deriv(x):
    return x * (1 - x)


def NeuralNetwork_Backpropagation(y, nn):
    dLdy = nn.y - y
    cache = np.zeros((len(nn.layerNodeCounts), np.amax(nn.layerNodeCounts)))

    for target in reversed(range(1, len(nn.layerNodeCounts))):
        if target != 0 and target == nn.layerCount: 
            for to in range(1, nn.layerNodeCounts[target]):
                cache[target, to] = dLdy
                for fromNode in range(nn.layerNodeCounts[target-1]):
                    nn.dweights[target-1,to,fromNode] = cache[target, to] * nn.nodes[target-1, fromNode]
        else: 
            for to in range(1, nn.layerNodeCounts[target]):
                cache[target, to] = 0
                for connected in range(1, nn.layerNodeCounts[target+1]):
                    cache[target, to] += cache[target+1, connected] * nn.weights[target, connected, to] * sigmoid_deriv(nn.nodes[target, to])

            for to in range(nn.layerNodeCounts[target]):
                for fromNode in range(nn.layerNodeCounts[target-1]):
                    nn.dweights[target-1,to,fromNode] = cache[target, to] * nn.nodes[target-1, fromNode]


def NeuralNetwork_Forwardpass(x, nn):
    nn.nodes[0,:x.shape[1]] = np.copy(x)
    for layer in range(1, len(nn.layerNodeCounts)): 
        for node in range(1, nn.layerNodeCounts[layer]): 
            layerSum = np.sum(np.multiply(nn.nodes[layer-1,:], nn.weights[layer-1,node,:]))
            if layer == nn.layerCount: 
                nn.y = layerSum
            else: 
                nn.nodes[layer, node] = sigmoid(layerSum)

def NeuralNetwork_SGD(x, y, nn, GammaSchedule, T, checkConverge):
    from copy import deepcopy
    idxs = np.arange(x.shape[0])
    gamma = GammaSchedule.gamma0
    iterations = 1
    
    lossList = []
    for epoch in range(T):
        np.random.shuffle(idxs)

        for i in idxs:
            gamma = GammaSchedule.gamma0 / (1 + (GammaSchedule.gamma0 * iterations / GammaSchedule.d))

            NeuralNetwork_Forwardpass(x[i], nn)       
            NeuralNetwork_Backpropagation(y[i], nn)
            nn.weights = np.subtract(nn.weights, x.shape[0]*gamma*nn.dweights)

            iterations += 1

        if checkConverge == True:
            lossSum = 0
            for i in idxs:
                NeuralNetwork_Forwardpass(x[i], nn)
                lossSum += 0.5 * (nn.y - y[i])**2
            lossList.append(lossSum)

    return deepcopy(nn), lossList

def NeuralNetwork_SGD_predict(x, nn):
    predictions = []
    for ex in x:
        NeuralNetwork_Forwardpass(ex, nn)
        p = nn.y
        if p < 0:
            predictions.append(-1)
        else:
            predictions.append(1)
    return np.array(predictions)
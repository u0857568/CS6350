from copy import deepcopy
import csv
from enum import Enum

class GainMethods(Enum):
    ENTROPY = 1
    MAJORITY = 2
    GINI = 3

class Tree:
    def __init__(self, val):
        self.children = []
        self.parent = None
        self.attrSplit = None
        self.attrValue = val
        self.label = None
        self.common = None
        self.depth = None

def parseCSV(csvFilePath, cols):
    data = []

    with open(csvFilePath, 'r') as csvFile:
        csvReader = csv.reader(csvFile, delimiter=',')

        for row in csvReader:
            thisDict = {}
            i = 0
            for attr in cols:
                thisDict[attr] = row[i]
                i += 1
            data.append(thisDict)

    return data

def sameLabel(data, labelCol):
    lbl = data[0][labelCol]
    for example in data:
        if example[labelCol] != lbl:
            return False
    return True

def common(data, attr, labelCol, D):
    countDict = {}
    for lbl in attr[labelCol]:
        countDict[lbl] = 0
    
    for idx, example in enumerate(data):
        countDict[example[labelCol]] = countDict[example[labelCol]] + D[idx]

    vals = list(countDict.values())
    keys = list(countDict.keys())
    return keys[vals.index(max(vals))]

def splitData(data, attr, v, D):
    newData = []
    newD = []
    for idx, example in enumerate(data):
        if example[attr] == v:
            newData.append(example)
            newD.append(D[idx])

    return newData, newD

def purity(val, attr, data, labelList, labelCol, gainMethod, D):
    import math


    labelCount = {}
    for lbl in labelList:
        labelCount[lbl] = 0

    
    totalOccurences = 0
    for idx, example in enumerate(data):
        if val == "all": 
            labelCount[example[labelCol]] = labelCount[example[labelCol]] + D[idx]
            totalOccurences += D[idx]
        elif example[attr] == val:
            labelCount[example[labelCol]] = labelCount[example[labelCol]] + D[idx]
            totalOccurences += D[idx]

    if gainMethod == GainMethods.ENTROPY:
        entropy = 0
        for lbl in labelList:
            if totalOccurences > 0:
                p_i = labelCount[lbl]/totalOccurences
                if p_i > 0:
                    entropy += (p_i * math.log2(p_i))
        return (-1*entropy), totalOccurences
    elif gainMethod == GainMethods.GINI:
        giniSum = 0
        for lbl in labelList:
            if totalOccurences > 0:
                p_i = labelCount[lbl]/totalOccurences
                giniSum += (p_i * p_i)
        return (1-giniSum), totalOccurences
    elif gainMethod == GainMethods.MAJORITY:
        vals = list(labelCount.values())
        keys = list(labelCount.keys())
        maxIndx = vals.index(max(vals))
        errorSum = 0
        for i in range(0,len(vals)):
            if i != maxIndx and totalOccurences > 0:
                errorSum += vals[i]
        if totalOccurences > 0:
            return (errorSum/totalOccurences), totalOccurences
        else:
            return 0, totalOccurences

def best(data, attrList, labelCol, gainMethod, D):
    setPurity, totalCount = purity("all", None, data, attrList[labelCol], labelCol, gainMethod, D)

    attrGains = {}
    for attr in attrList:
        if attr == labelCol:
            continue
        attrPuritySum = 0
        for val in attrList[attr]: 
            attrValPurity, occur= purity(val, attr, data, attrList[labelCol], labelCol, gainMethod, D)
            attrPuritySum += (occur/totalCount) * attrValPurity

        attrGains[attr] = setPurity - attrPuritySum
    

    vals = list(attrGains.values())
    keys = list(attrGains.keys())
    if not vals:
        return None
    else:
        return keys[vals.index(max(vals))]


def ID3(data, attrDict, labelCol, node, maxDepth, gainMethod, D):
    import copy
    
    if D == None:
        D = [1 for i in range(len(data))]

    if not attrDict: 
        node.label = node.parent.common
        return
    if sameLabel(data, labelCol): 
        node.label = data[0][labelCol]
        return

    node.common = common(data, attrDict, labelCol, D) 

    if node.depth == maxDepth:
        node.label = node.common
        return

    node.attrSplit = best(data, attrDict, labelCol, gainMethod, D)
    
    if node.attrSplit == None:
        node.label = node.common
        return

    for v in attrDict[node.attrSplit]:
        if v == labelCol:
            continue
        
        child = Tree(v)
        child.parent = node
        child.depth = child.parent.depth + 1
        node.children.append(child)

        dataSplit, splitD = splitData(data, node.attrSplit, v, D)

        if not dataSplit:
            child.label = child.parent.common
        else:
            newAttrDict = copy.deepcopy(attrDict)
            del newAttrDict[node.attrSplit] 
            ID3(dataSplit, newAttrDict, labelCol, child, maxDepth, gainMethod, splitD)

    return node

def ID3_RandTree(data, attrDict, labelCol, node, maxDepth, gainMethod, D, featureSetSize):
    import copy, random
    
    if D == None:
        D = [1 for i in range(len(data))]

    if not attrDict:
        node.label = node.parent.common
        return
    if sameLabel(data, labelCol):
        node.label = data[0][labelCol]
        return

    node.common = common(data, attrDict, labelCol, D)

    if node.depth == maxDepth:
        node.label = node.common
        return

    attrSamples = {}
    try:
        newAttrDict = copy.deepcopy(attrDict)
        del newAttrDict[labelCol]
        attrSamplesList = random.sample(list(newAttrDict), k=featureSetSize)
        for attr in attrSamplesList:
            attrSamples[attr] = attrDict[attr]
        attrSamples[labelCol] = attrDict[labelCol]
    except ValueError:
        attrSamples = attrDict
   
    node.attrSplit = best(data, attrSamples, labelCol, gainMethod, D)
    
    if node.attrSplit == None:
        node.label = node.common
        return

    for v in attrDict[node.attrSplit]:
        if v == labelCol:
            continue

        child = Tree(v)
        child.parent = node
        child.depth = child.parent.depth + 1
        node.children.append(child)

        dataSplit, splitD = splitData(data, node.attrSplit, v, D)

        if not dataSplit:
            child.label = child.parent.common
        else:
            newAttrDict = copy.deepcopy(attrDict)
            del newAttrDict[node.attrSplit] 
            ID3(dataSplit, newAttrDict, labelCol, child, maxDepth, gainMethod, splitD)

    return node

def predict(data, predictCol, root):
    import copy
    predictData = copy.deepcopy(data)
    for example in predictData:
        node = root
        while node.label == None:
            for child in node.children:
                if child.attrValue == example[node.attrSplit]:
                    node = child
                    break
        example[predictCol] = node.label
    return predictData

def predict_example(example, predictCol, root):
    node = root
    while node.label == None:
        for child in node.children:
            if child.attrValue == example[node.attrSplit]:
                node = child
                break
    return node.label

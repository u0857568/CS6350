from math import inf
import sys
sys.path.append("../DecisionTree")
import DecisionTree

def RandomForests(data, attrDict, labelCol, gainMethod, T, featureSetSize, samplesize):
    import random, math
    forest = []

    for i in range(T):
        samples = random.choices(data, k=math.ceil(samplesize*len(data)))

        root = DecisionTree.Tree(None)
        root.depth = 0
        DecisionTree.ID3_RandTree(samples, attrDict, labelCol, root, inf, gainMethod, None, featureSetSize)
        forest.append(root)

    return forest

def predict(data, predictCol, forest):
    import copy
    predictData = copy.deepcopy(data)

    for example in predictData:
        example[predictCol] = predict_example(example, predictCol, forest)

    return predictData

def predict_example(example, predictCol, forest):
    labelVotes = {}
    for root in forest:
        thisPredict = DecisionTree.predict_example(example, predictCol, root)
        if thisPredict not in labelVotes:
            labelVotes[thisPredict] = 1
        else:
            labelVotes[thisPredict] += 1
    return max(labelVotes, key=labelVotes.get)

from math import inf
import sys
sys.path.append("../DecisionTree")
import DecisionTree

def BaggedDecisionTrees(data, attrDict, labelCol, gainMethod, T, samplesize):
    import random, math
    trees = []

    for i in range(T):
        samples = random.choices(data, k=math.ceil(samplesize*len(data)))

        root = DecisionTree.Tree(None)
        root.depth = 0
        DecisionTree.ID3(samples, attrDict, labelCol, root, inf, gainMethod, None)
        trees.append(root)

    return trees

def predict(data, predictCol, trees):
    import copy
    predictData = copy.deepcopy(data)

    for example in predictData:
        example[predictCol] = predict_example(example, predictCol, trees)

    return predictData

def predict_example(example, predictCol, trees):
    labelVotes = {}
    for root in trees:
        thisPredict = DecisionTree.predict_example(example, predictCol, root)
        if thisPredict not in labelVotes:
            labelVotes[thisPredict] = 1
        else:
            labelVotes[thisPredict] += 1
    return max(labelVotes, key=labelVotes.get)

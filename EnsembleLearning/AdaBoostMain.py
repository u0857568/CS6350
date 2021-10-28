import sys
sys.path.append("./DecisionTree")
import DecisionTree
import math

def stringBinaryLabel2numerical(data, attrDict, labelCol, negVal, posVal):
    import copy
    newData = copy.deepcopy(data)
    newAttrDict = copy.deepcopy(attrDict)
    for example in newData:
        if example[labelCol] == negVal:
            example[labelCol] = -1
        elif example[labelCol] == posVal:
            example[labelCol] = 1
    newAttrDict[labelCol] = [-1, 1]
    return newData, newAttrDict

def numericalLabel2string(data, attrDict, labelCol, negVal, posVal):
    import copy
    newData = copy.deepcopy(data)
    newAttrDict = copy.deepcopy(attrDict)
    for example in newData:
        if example[labelCol] == -1:
            example[labelCol] = negVal
        elif example[labelCol] == 1:
            example[labelCol] = posVal
    newAttrDict[labelCol] = [negVal, posVal]
    return newData, newAttrDict

def AdaBoost(data, attrDict, labelCol, gainMethod, T):
    d_weights = [1/len(data) for i in range(len(data))]
    a_list = []
    hyp_list = []

    for i in range(0, T):
        root = DecisionTree.Tree(None)
        root.depth = 0
        DecisionTree.ID3(data, attrDict, labelCol, root, 1, gainMethod, d_weights)
        hyp_list.append(root)

        stumpPredict = DecisionTree.predict(data, 'prediction', root)
        
        e = 0.0
        for idx, example in enumerate(data):
            if example[labelCol] != stumpPredict[idx]['prediction']:
                e += d_weights[idx]
        
        a = 0.5 * math.log2((1-e) / e)
        a_list.append(a)

        for idx, example in enumerate(data):
            d_weights[idx] = (d_weights[idx]) * math.exp(-a * example[labelCol] * stumpPredict[idx]['prediction'])

        Z = sum(d_weights)
        d_weights[:] = [d / Z for d in d_weights]

    return a_list, hyp_list

def predict(data, predictCol, a_list, hyp_list):
    import copy
    predictData = copy.deepcopy(data)
    for example in predictData:
        hyp_sum = 0
        for idx, hyp in enumerate(hyp_list):
            hyp_sum += a_list[idx]*DecisionTree.predict_example(example, predictCol, hyp)
        if hyp_sum < 0:
            example[predictCol] = -1
        else:
            example[predictCol] = 1
    return predictData


def stumpErrors(data, labelCol, predictCol, hyp_list):
    import copy
    stumpErrorList = []

    for idx, hyp in enumerate(hyp_list):
        predictdata = DecisionTree.predict(data, predictCol, hyp)
        total = 0
        wrong = 0
        for example in predictdata:
            if example[labelCol] != example[predictCol]:
                wrong += 1
            total += 1
        stumpErrorList.append(wrong/total)
    return stumpErrorList

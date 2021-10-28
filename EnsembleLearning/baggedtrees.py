import random
import statistics
import sys
sys.path.append("../DecisionTree")
sys.path.append("../Unknown")

import DecisionTree
import Unknown
import BaggedTreesMain


print("********** Part 2b **********")
print("Bagged Decision Tree experiment")

train_data = "bank/train.csv"
test_data = "bank/test.csv"

maxDepth = 1

cols = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing',
        'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays',
        'previous', 'poutcome', 'y']

attrDict = {}
attrDict["age"] = []
attrDict['job'] = ["admin.", "unknown", "unemployed", "management", "housemaid",
                   "entrepreneur", "student", "blue-collar", "self-employed", 
                   "retired", "technician", "services"]
attrDict['marital'] = ["married", "divorced", "single"]
attrDict['education'] = ["unknown", "secondary", "primary", "tertiary"]
attrDict['default'] = ["yes", "no"]
attrDict['balance'] = []
attrDict['housing'] = ["yes", "no"]
attrDict['loan'] = ["yes", "no"]
attrDict['contact'] = ["unknown", "telephone", "cellular"]
attrDict['day'] = []
attrDict['month'] = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", 
                     "oct", "nov", "dec"]
attrDict['duration'] = []
attrDict['campaign'] = []
attrDict['pdays'] = []
attrDict['previous'] = []
attrDict['poutcome'] = ["unknown", "other", "failure", "success"]
attrDict['y'] = ["yes", "no"]

examples_train = DecisionTree.parseCSV(train_data, cols)
examples_test = DecisionTree.parseCSV(test_data, cols)

medianList = Unknown.numerical2binary_MedianThreshold(examples_train, attrDict)

temp, examples_train = Unknown.numerical2binary_MedianThreshold_Replace(examples_train, attrDict, medianList)
attrDict, examples_test = Unknown.numerical2binary_MedianThreshold_Replace(examples_test, attrDict, medianList)

print("T\tTraining Data\tTest Data")
for depth in range(1, maxDepth+1):
    tree_list = BaggedTreesMain.BaggedDecisionTrees(examples_train, attrDict, 'y', DecisionTree.GainMethods.ENTROPY, depth, 0.4)

    predictdata_train = BaggedTreesMain.predict(examples_train, 'prediction', tree_list)
    predictdata_test = BaggedTreesMain.predict(examples_test, 'prediction', tree_list)
    
    total_train = 0
    wrong_train = 0
    for example in predictdata_train:
        if example['y'] != example["prediction"]:
            wrong_train += 1
        total_train += 1
    total_test = 0
    wrong_test = 0
    for example in predictdata_test:
        if example['y'] != example["prediction"]:
            wrong_test += 1
        total_test += 1

    print(f"{depth}\t{wrong_train/total_train:.7f}\t{wrong_test/total_test:.7f}")
    with open("bank_errors_bagged.csv", 'a') as errorFile:
        errorFile.write(f"{depth},{wrong_train/total_train:.7f},{wrong_test/total_test:.7f}\n")

print("data written")

print()
print("********** Part 2c **********")
print("Bagged Decision Tree bias and variance experiments")

train_data = "bank/train.csv"
test_data = "bank/test.csv"

cols = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing',
        'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays',
        'previous', 'poutcome', 'y']

attrDict = {}
attrDict["age"] = []
attrDict['job'] = ["admin.", "unknown", "unemployed", "management", "housemaid",
                   "entrepreneur", "student", "blue-collar", "self-employed", 
                   "retired", "technician", "services"]
attrDict['marital'] = ["married", "divorced", "single"]
attrDict['education'] = ["unknown", "secondary", "primary", "tertiary"]
attrDict['default'] = ["yes", "no"]
attrDict['balance'] = []
attrDict['housing'] = ["yes", "no"]
attrDict['loan'] = ["yes", "no"]
attrDict['contact'] = ["unknown", "telephone", "cellular"]
attrDict['day'] = []
attrDict['month'] = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", 
                     "oct", "nov", "dec"]
attrDict['duration'] = []
attrDict['campaign'] = []
attrDict['pdays'] = []
attrDict['previous'] = []
attrDict['poutcome'] = ["unknown", "other", "failure", "success"]
attrDict['y'] = ["yes", "no"]

examples_train = DecisionTree.parseCSV(train_data, cols)
examples_test = DecisionTree.parseCSV(test_data, cols)

medianList = Unknown.numerical2binary_MedianThreshold(examples_train, attrDict)

temp, examples_train = Unknown.numerical2binary_MedianThreshold_Replace(examples_train, attrDict, medianList)
attrDict, examples_test = Unknown.numerical2binary_MedianThreshold_Replace(examples_test, attrDict, medianList)

trees = []
for i in range(2):
    samples = random.sample(examples_train, 1000)

    tree_list = BaggedTreesMain.BaggedDecisionTrees(samples, attrDict, 'y', DecisionTree.GainMethods.ENTROPY, 500, 1)
    trees.append(tree_list)

singleTreeBiases = []
singleTreeVariance = []
for example in examples_test:
    predictionList = []
    for tree_list in trees:
        prediction = DecisionTree.predict_example(example, 'prediction', tree_list[0])
        if prediction == 'yes':
            predictionList.append(1)
        else:
            predictionList.append(-1)
    bias = 0
    if example['y'] == 'yes':
        bias = 1
    else:
        bias = -1
    predictionAvg = statistics.mean(predictionList)
    bias -= predictionAvg
    bias *= bias
    singleTreeBiases.append(bias)

    variance = statistics.variance(predictionList)
    singleTreeVariance.append(variance)

biasEst = statistics.mean(singleTreeBiases)
varEst = statistics.mean(singleTreeVariance)

genSquareErrEst = biasEst + varEst
print(f"Single Decision Tree: General Bias = {biasEst:.7f}, General Variance = {varEst:.7f}, General Squared Error = {genSquareErrEst:.7f}")

baggedTreesBiases = []
baggedTreesVariance = []
for example in examples_test:
    predictionList = []
    for tree_list in trees:
        prediction = BaggedTreesMain.predict_example(example, "prediction", tree_list)
        if prediction == 'yes':
            predictionList.append(1)
        else:
            predictionList.append(-1)
    bias = 0
    if example['y'] == 'yes':
        bias = 1
    else:
        bias = -1
    predictionAvg = statistics.mean(predictionList)
    bias -= predictionAvg
    bias *= bias
    baggedTreesBiases.append(bias)

    variance = statistics.variance(predictionList)
    baggedTreesVariance.append(variance)

biasEst = statistics.mean(baggedTreesBiases)
varEst = statistics.mean(baggedTreesVariance)

genSquareErrEst = biasEst + varEst
print(f"Bagged Decision Trees: General Bias = {biasEst:.7f}, General Variance = {varEst:.7f}, General Squared Error = {genSquareErrEst:.7f}")

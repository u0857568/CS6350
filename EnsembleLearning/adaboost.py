import sys
sys.path.append("../DecisionTree")
sys.path.append("../Unknown")

import DecisionTree
import Unknown
import AdaBoostMain
import matplotlib.pyplot as plt 

print("********** Part 2a **********")
print("AdaBoost")

train_data = "bank/train.csv"
test_data = "bank/test.csv"

maxDepth = 100

col = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing',
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

examples_train = DecisionTree.parseCSV(train_data, col)
examples_test = DecisionTree.parseCSV(test_data, col)

medianList = Unknown.numerical2binary_MedianThreshold(examples_train, attrDict)
temp, examples_train = Unknown.numerical2binary_MedianThreshold_Replace(examples_train, attrDict, medianList)
attrDict, examples_test = Unknown.numerical2binary_MedianThreshold_Replace(examples_test, attrDict, medianList)

stumpErrs_train = []
stumpErrs_test = []

errorFile = open("bank_errors_adaboost.csv", 'w')

print("T\tTraining Data\tTest Data")

train_errors, test_errors = [0 for x in range(maxDepth)], [0 for x in range(maxDepth)]
train_errorsT, test_errorsT = [0 for x in range(maxDepth)], [0 for x in range(maxDepth)]
for depth in range(1, maxDepth+1):
    examples_train, AdaBoostAttrDict = AdaBoostMain.stringBinaryLabel2numerical(examples_train, attrDict, 'y', 'no', 'yes')
    examples_test, AdaBoostAttrDict = AdaBoostMain.stringBinaryLabel2numerical(examples_test, attrDict, 'y', 'no', 'yes')

    a_list, hyp_list = AdaBoostMain.AdaBoost(examples_train, AdaBoostAttrDict, 'y', DecisionTree.GainMethods.ENTROPY, depth)

    predictdata_train = AdaBoostMain.predict(examples_train, 'prediction', a_list, hyp_list)
    predictdata_test = AdaBoostMain.predict(examples_test, 'prediction', a_list, hyp_list)
    
    if depth == maxDepth:
        stumpErrs_train = AdaBoostMain.stumpErrors(examples_train, 'y', 'prediction', hyp_list)
        stumpErrs_test = AdaBoostMain.stumpErrors(examples_test, 'y', 'prediction', hyp_list)

    predictdata_train, oldAttrDict = AdaBoostMain.numericalLabel2string(predictdata_train, AdaBoostAttrDict, 'y', 'no', 'yes')
    predictdata_train, oldAttrDict = AdaBoostMain.numericalLabel2string(predictdata_train, AdaBoostAttrDict, 'prediction', 'no', 'yes')
    predictdata_test, oldAttrDict = AdaBoostMain.numericalLabel2string(predictdata_test, AdaBoostAttrDict, 'y', 'no', 'yes')
    predictdata_test, oldAttrDict = AdaBoostMain.numericalLabel2string(predictdata_test, AdaBoostAttrDict, 'prediction', 'no', 'yes')

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
    errorFile.write(f"{depth},{wrong_train/total_train:.7f},{wrong_test/total_test:.7f}\n")
    train_errors.append(wrong_train/total_train)
    test_errors.append(wrong_test/total_test)

errorFile.close()

errorFile = open("bank_stumperrors_adaboost.csv", 'w')
print()
print()
print("Stump errors at each iteration")
print("T\tTraining Data\tTest Data")

for i in range(len(stumpErrs_train)):
    print(f"{i+1}\t{stumpErrs_train[i]:.7f}\t{stumpErrs_test[i]:.7f}")
    errorFile.write(f"{i+1},{stumpErrs_train[i]:.7f},{stumpErrs_test[i]:.7f}\n")
    train_errorsT.append(wrong_train/total_train)
    test_errorsT.append(wrong_test/total_test)

print("data written")
fig, (ax1, ax2) = plt.subplots(2, 1)
fig.tight_layout(h_pad=2)

ax1.plot(train_errors[::-1],  color='blue', label='train_data')
ax1.set_ylabel('error')
ax1.plot(test_errors[::-1],  color='red', label='test_data')

ax1.set_title("individual prediction")
ax1.legend()

ax2.plot(train_errorsT[::-1],  color='blue', label='train_data')
ax2.set_ylabel('error rate')
ax2.set_xlabel('T')
ax2.plot(test_errorsT[::-1],  color='red', label='test_data')

ax2.set_title("all prediction")
ax2.legend()

fig.savefig('adaboost.png', dpi=300, bbox_inches='tight')
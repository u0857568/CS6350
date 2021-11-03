import numpy as np

import PerceptronMain

train_data = "bank-note/train.csv"
test_data = "bank-note/test.csv"


print("2a")
print("Standard Perceptron")

x, y = PerceptronMain.parseCSV(train_data, True)
x_test, y_test = PerceptronMain.parseCSV(test_data, True)

r = 0.1
T = 10
w = PerceptronMain.StandardPerceptron(x, y, r, T)
print(f"Learned weight vector: {w}")

test_predictions = PerceptronMain.predict_StandardPerceptron(x_test, w)
numWrong = sum(abs(test_predictions-y_test) / 2)
print(f"Test Error after {T} epochs: {numWrong/len(y_test)}")



print()
print("2b")
print("Voted Perceptron")

r = 0.1
T = 10
wghts = PerceptronMain.VotedPerceptron(x, y, r, T)
with open("2b.csv", 'w') as f:
    f.write(f"Weight Vector,Count\n")
    for wc in wghts:
        f.write(f"{wc[0]},{wc[1]}\n")
print("check data in 2b.csv")

test_predictions = PerceptronMain.predict_VotedPerceptron(x_test, wghts)
# if prediction is different, difference will be +-2, if same, will be 0
numWrong = sum(abs(test_predictions-y_test) / 2)
print(f"Test Error after {T} epochs: {numWrong/len(y_test)}")




print()
print("2c")
print("Average Perceptron")

r = 0.1
T = 10
a = PerceptronMain.AveragedPerceptron(x, y, r, T)
print(f"Learned weight vector: {a}")

test_predictions = PerceptronMain.predict_AveragedPerceptron(x_test, a)

numWrong = sum(abs(test_predictions-y_test) / 2)
print(f"Test Error after {T} epochs: {numWrong/len(y_test)}")
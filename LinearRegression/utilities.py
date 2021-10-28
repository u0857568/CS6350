def createInputMatrices(data, labelCol):
    x = []
    y = []
    for example in data:
        exampleList = [1] 
        for attr in example:
            if attr == labelCol:
                y.append(float(example[attr]))
            else:
                exampleList.append(float(example[attr]))
        x.append(exampleList)
    return x, y

def dot(w, x):
    return sum([wk*xk for wk,xk in zip(w,x)])

def costValue(w,x,y):
    costSum = 0
    for i in range(len(y)):
        costSum += (y[i] - dot(w,x[i]))**2
    return 0.5*costSum


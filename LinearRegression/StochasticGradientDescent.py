import utilities

def StochasticGradientDescent(x, y, r, iterations):
    import random

    wghts = [0 for i in range(len(x[0]))]
    costs = [utilities.costValue(wghts, x, y)]
    converge = False

    for i in range(iterations):
        index = random.randrange(len(x))
        newWghts = []
        for j in range(len(wghts)):
            newWghts.append(wghts[j] + r*x[index][j]*(y[index] - utilities.dot(wghts,x[index])))
        wghts = newWghts
        
        costVal = utilities.costValue(wghts, x, y)
        if abs(costVal - costs[-1]) < 10e-6:
            converge = True
        costs.append(costVal)

    return wghts, costs, converge

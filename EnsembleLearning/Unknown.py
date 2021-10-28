from statistics import quantiles

def numerical2binary_MedianThreshold(data, attrDict):
    import statistics

    medianList = {}
    for attr in attrDict:
        
        if not attrDict[attr]:
            
            valuesList = []
            for example in data:
                valuesList.append(float(example[attr]))
            attrMedian = statistics.median(valuesList)
            medianList[attr] = attrMedian

    return medianList

def numerical2binary_MedianThreshold_Replace3Categories(data, attrDict, medianList):
    import copy
    attrDictCopy = copy.deepcopy(attrDict)

    for attr in medianList:
        for example in data:
            if float(example[attr]) > medianList[attr]:
                example[attr] = "larger"
            elif float(example[attr]) == medianList[attr]:
                example[attr] = "equal"
            else:
                example[attr] = "smaller"
        attrDictCopy[attr] = ["smaller", "equal", "larger"]

    return attrDictCopy, data

def numerical2binary_MedianThreshold_Replace(data, attrDict, medianList):
    import copy
    attrDictCopy = copy.deepcopy(attrDict)

    for attr in medianList:
        
        for example in data:
            if float(example[attr]) > medianList[attr]:
                example[attr] = "larger"
            else:
                example[attr] = "smaller"
        attrDictCopy[attr] = ["smaller", "larger"]

    return attrDictCopy, data

def findMajorityAttribute(data, attrDict, attrCol, unknown):
    countDict = {}
    for lbl in attrDict[attrCol]:
        countDict[lbl] = 0
    
    for example in data:
        if countDict[example[attrCol]] != unknown:
            countDict[example[attrCol]] = countDict[example[attrCol]] + 1

    vals = list(countDict.values())
    keys = list(countDict.keys())
    return keys[vals.index(max(vals))]

def replaceUnknown_MajorityAttribute(data, attrDict, unknown):
    majorityAttrs = {}
    for attr in attrDict:
        majorityAttrs[attr] = findMajorityAttribute(data, attrDict, attr, unknown)

    return majorityAttrs


def replaceUnknown_MajorityAttribute_Replace(data, majorityAttrs, unknown):
    for attr in majorityAttrs:
        for example in data:
            if example[attr] == unknown:
                example[attr] = majorityAttrs[attr]

    return data


def replaceContinuous_Quartiles(data, attrDict):
    import statistics

    quartilesList = {}
    for attr in attrDict:
        
        if not attrDict[attr]:
            valuesList = []

            for example in data:
                valuesList.append(float(example[attr]))
            quartiles = statistics.quantiles(valuesList, n=4)
            quartilesList[attr] = quartiles

    return quartilesList

def replaceContinuous_Quartiles_Replace(data, attrDict, quartilesList):
    import copy
    attrDictCopy = copy.deepcopy(attrDict)

    for attr in quartilesList:
        
        for example in data:
            if float(example[attr]) < quartilesList[attr][0]:
                example[attr] = 'Q1'
            elif float(example[attr]) < quartilesList[attr][1]:
                example[attr] = 'Q2'
            elif float(example[attr]) < quartilesList[attr][2]:
                example[attr] = 'Q3'
            else:
                example[attr] = 'Q4'
        attrDictCopy[attr] = ["Q1", "Q2", 'Q3', 'Q4']

    return attrDictCopy, data

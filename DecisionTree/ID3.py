import math
import argparse
import sys


parser = argparse.ArgumentParser()
parser.add_argument( "-train")
parser.add_argument( "-test")

parser.add_argument( "-car", action="store_true" )
parser.add_argument( "-ME", action="store_true" )
parser.add_argument( "-GI", action="store_true" )
parser.add_argument( "-depth", type=int )

parser.add_argument( "-bank", action="store_true" )

parser.add_argument( "-missing", action="store_true" )

args = parser.parse_args()


class Node(object):
    def __init__(self, data, depth, label, entropy):
        
        self.children = []
        
        self.data = data

        self.depth = depth

        self.label = label
        
        self.entropy = entropy
        

class ID3Tree(object):
    def __init__(self, data, depth, labels, value, cAttribute, attributes):
        self.data = data

        self.depth = depth

        self.labels = labels

        self.value = value

        self.cAttribute = cAttribute

        self.attributes = attributes

        self.root = None

        self.isRoot = False


    def getAttribute(self, attribute):
        if args.car:
            values ={
                'buying':['vhigh','high','med', 'low'],
                'maint':['vhigh','high','med', 'low'],
                'doors':['2','3','4','5more'],
                'persons':['2','4','more'],
                'lug_boot':['small','med','big'],
                'safety':['low','med','high']
            }
        if args.bank and self.isRoot:
            values = {
                'age':[],
                'job':['admin.','unknown','unemployed','management', 'housemaid','entreprenuer','student','blue-collar','self-employeed','retired','technician','services'],
                'marital':['married', 'divorced', 'single'],
                'education':['unknown','secondary','primary','tertiary'],
                'default':['yes','no'],
                'balance':[],
                'housing':['yes','no'],
                'loan':['yes','no'],
                'contact':['unknown','telephone','cellular'],
                'day':[],
                'month':['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'],
                'duration':[],
                'campaign':[],
                'pdays':[],
                'previous':[],
                'poutcome':['unknown','other','failure','success']
            }

        if args.bank and not self.isRoot:
            values = {
                'age':['low','high'],
                'job':['admin.','unknown','unemployed','management', 'housemaid','entreprenuer','student','blue-collar','self-employeed','retired','technician','services'],
                'marital':['married', 'divorced', 'single'],
                'education':['unknown','secondary','primary','tertiary'],
                'default':['yes','no'],
                'balance':['low', 'high'],
                'housing':['yes','no'],
                'loan':['yes','no'],
                'contact':['unknown','telephone','cellular'],
                'day':['low', 'high'],
                'month':['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'],
                'duration':['low', 'high'],
                'campaign':['low', 'high'],
                'pdays':['low', 'high'],
                'previous':['low', 'high'],
                'poutcome':['unknown','other','failure','success']
            }
        return values.get(attribute,[])


    def labelCount(self, label, data):
        count = 0

        for x in data:
            if x[-1] == label:
                count += 1

        return count


    def commonLabel(self,data):
        count = 0
        clabel = self.labels[0]

        for i in self.labels:
            labelCount = self.labelCount(i, data)

            if labelCount > count:
                count = labelCount
                clabel = i

        return clabel



    def labelMatching(self):
        matching = True
        label = self.data[0][-1]

        for i in self.data:
            if i[-1] != label:
                matching=False
                break

        return (matching, label)


    def commonAttribute(self,attr):
        attr_index=self.attributes.index(attr)
        attribute_values=self.getAttribute(attr)
        attr_count=[0 for x in range(0,len(attribute_values))]
        for x in self.data:
            idx=0
            for attr_val in attribute_values:
                if x[attr_index]==attr_val:
                    attr_count[idx]+=1
                idx+=1

        return attribute_values[max_index]


    def bestSplit(self):
        count = 0
        splitAttributesList = [0 for attr in range(0, len(self.attributes))]
        

        for attribute in self.attributes:
            splitAttributesList[count] = self.gain(attribute)
            count += 1

        best = max(splitAttributesList)
        count = 0

        for gain in splitAttributesList:
            if gain == best:
                break
            count += 1

        return self.attributes[count]


    def gain(self, attribute):

        entropy = self.entropy(self.data)
        gain = entropy

        index = self.attributes.index(attribute)
        valueA = self.getAttribute(attribute)
        score=[0 for i in range(0,len(valueA))]
        

        for i in valueA:
            subset=[]

            for j in self.data:
                if j[index] == i:
                    score[valueA.index(i)] += 1
                    subset.append(j)

            entropySub = self.entropy(subset)
            gain = gain - score[valueA.index(i)] * entropySub / len(self.data)

        return gain


    def entropy(self, data):
        if args.ME:
            ME = 0

            if len(data)==0:
                return ME

            majority = self.commonLabel(data)
            count = self.labelCount(majority, data)
            ME = 1- (count / float(len(data)))

            return ME

        if args.GI:
            GI = 1
            if len(data) == 0:
                return GI

            for x in self.labels:
                p = self.labelCount(x,data) / float(len(data))
                GI -= p*p
            return GI

        entropy = 0
        if dataLength == 0:
            return entropy

        for x in self.labels:

            p=self.labelCount(x,data) / float(dataLength)
            if p!= 0:
                entropy-=p*math.log(p,2)

        return entropy


    def buildTree(self):

        medianList=[]
        commonAttributes=[]

        if self.depth == args.depth:
            self.isRoot = True

        if self.isRoot:

            for i in self.attributes:
                attribute = self.getAttribute(i)
                index =self.attributes.index(i)

                if len(attribute) == 0:
                    medianVector = []

                    for j in self.data:
                        medianVector.append(float(j[index]))

                    medianValue =calculateMedian(medianVector)
                    medianList.append(medianValue)

                    for j in self.data:
                        if float(j[index])>medianValue:
                            j[index] ='high'

                        else:
                            j[index] = 'low'

                else:
                    if args.missing:
                        commonAttribute = self.commonAttribute(attr)
                        commonAttributes.append(commonAttribute)

                        for j in self.data:
                            if j[index]=='unknown':
                                j[index]=commonAttribute

            self.isRoot=False

        cAttribute = self.bestSplit()
        (matching, label) = self.labelMatching()

        entropy = self.entropy(self.data)
        label = self.commonLabel(self.data)
        rootNode= Node(self.data,self.depth, label, entropy)

        if (self.depth > 0 and not matching ):
            attribute = self.getAttribute(cAttribute)
            
            for i in attribute:
                temp=[]
                
                for data in self.data:
                    if data[self.attributes.index(cAttribute)] == i:
                        temp.append(data)

                if len(temp)!=0:
                    subtree = ID3Tree(temp, self.depth-1, self.labels, i, cAttribute, self.attributes)
                    rootNode.children.append(subtree)
                    subtree.buildTree()


        self.root =rootNode
        self.root.value =self.value
        self.root.cAttribute =self.cAttribute
        
        return (rootNode, medianList, commonAttributes)



    def predictLabel(self, x):
        label = self.root.label

        for child in self.root.children:
            count = 0
            cAttribute = child.root.cAttribute
            
            for attribute in self.attributes:
                if cAttribute == attribute:
                    index = count
                count += 1

            attribute = child.root.value

            if x[index] == attribute:
                label=child.predictLabel(x)

        return label


def load_file(fileName):
    data=[]
    with open(fileName, 'r' ) as f:
        for line in f:
            terms = line.strip().split(',')
            data.append(terms)
    return data


def calculateMedian(data):
    n = len(data)

    if n < 1:
        return None

    if n % 2 == 1:
        return sorted(data)[n//2]

    else:
        return sum(sorted(data)[n//2-1:n//2+1])/2.0


def trainData(data):
    if args.car:
        labels = ['unacc','acc','good', 'vgood']
        attributes = ['buying', 'maint', 'doors','persons','lug_boot','safety']
        

    if args.bank:
        labels = ['yes','no']
        attributes = ['age','job','marital','education','default','balance','housing','loan','contact','day','month','duration','campaign','pdays','previous','poutcome']
        

    tree=ID3Tree(data, args.depth, labels, "all", "base", attributes)
    (rootOfTree, medianList, commonAttributes) = tree.buildTree()

    return (tree, medianList, commonAttributes)

def testData(data, tree, medianList, commonAttributes):
    prediction = 0
    length=len(data)

    if args.car:
        dataSet='cars'

    if args.bank:
        dataSet ='bank'
        numericalAttributes = [0, 5, 9, 11, 12, 13, 14]
        fixNumerical(data, numericalAttributes, medianList)

    if args.missing:
        for x in data:
            for i in range(len(commonAttributes)):
                if x[i] == 'unknown':
                    x[i] = commonAttributes[i]

    for x in data:
        label=tree.predictLabel(x)

        if label == x[-1]:
            prediction += 1
    accuracy =prediction / float(length)

    print(dataSet+ " in depth " + str(args.depth) + " has accuracy: "+ str(accuracy))



def fixNumerical(data, numericalAttributes, medianValues):
    for i in range(len(numericalAttributes)):

        for x in data:
            if float(x[numericalAttributes[i]]) > medianValues[i]:
                x[numericalAttributes[i]] = 'high'
            else:
                x[numericalAttributes[i]] = 'low'


if __name__ == "__main__":

        data = load_file(args.train)
        (tree, medianList, commonAttributes) = trainData(data)
        data = load_file(args.test)
        testData(data, tree, medianList, commonAttributes)




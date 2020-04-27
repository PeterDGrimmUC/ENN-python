import math
import numpy as np
import random as rand
from itertools import groupby
from enum import Enum
import pdb
from sklearn import preprocessing
import threading
import multiprocessing
randomVal = lambda : rand.uniform(0,1)

class activationFuncs(Enum):
    SIGMOID=lambda x: 1/(1+math.exp(-x))
    TANH = lambda x: (math.exp(x)-math.exp(-x))/(math.exp(x)+math.exp(-x))

class nodeTypes(Enum):
    INPUT=0
    OUTPUT=1
    HIDDEN =2
    SIGMOID=3
    TANH=4

class connectionTypes(Enum):
    STANDARD=0
    RECURRRENT=1

# Class node: Node in network
class masterNode:
    def __init__(self,nodeNum,inputInnovations = [],outputInnovations=[], depth=-1,bias=-1,activationFunc=activationFuncs.TANH, nodeType=nodeTypes.SIGMOID):
        self.nodeNum = nodeNum
        self.depth = depth
        self.nodeType = nodeType
        self.inputInnovations = []
        self.outputInnovations = []
        self.bias = bias
        self.activationFunction = activationFunc
        self.delta = float("inf")
    def __repr__(self):
        return 'NodeNum: %i, depth = %f' %(self.nodeNum, self.depth)

    def copy(self, newNodeNum=None):
        if newNodeNum is None:
            newNodeNum = self.nodeNum
        newInps = []
        newOutps = []
        for inputInnovation in self.inputInnovations:
            newInps.append(inputInnovation.copy())
        for outputInnovation in self.outputInnovations:
            newOutps.append(outputInnovation.copy())
        return masterNode(newNodeNum, newInps, newOutps ,self.depth,self.bias,nodeType=self.nodeType)

    def stat(self):
        print(self.__str__())

# Class Connection: connections in network
class masterConnection:
    def __init__(self, innovNum, inputNode, outputNode, connectionType=connectionTypes.STANDARD):
        self.innovNum = innovNum
        self.inputNode = inputNode
        self.outputNode = outputNode
        self.enable = True
        self.connectionType = connectionType
    def copy(self,newInnovNum=None):
        if newInnovNum is None:
            newInnovNum = self.innovNum
        return masterConnection(newInnovNum, self.IOTuple,self.weight,self.depth)
    def __repr__(self):
        return "ID: %i, IO (%i,%i)" % (self.innovNum, self.inputNode.nodeNum,self.outputNode.nodeNum)

class node:
    def __init__(self,masterRef,bias):
        self.masterRef = masterRef
        self.bias = bias
        self.enabled = True
        self.nodeNum = self.masterRef.nodeNum
    def disable(self):
        self.enabled = False
    def enable(self):
        self.enabled = True
    def copy(self):
        return node(self.masterRef, randomVal())
    def __repr__(self):
        return "ID: %i,Depth= %f, bias: %f, enabled: %r" % (self.nodeNum,self.masterRef.depth, self.bias, self.enabled)

class connection:
    def __init__(self, masterRef, weight):
        self.masterRef = masterRef
        self.innovNum = self.masterRef.innovNum
        self.weight = weight
        self.enabled = True
    def disable(self):
        self.enabled = False
    def enable(self):
        self.enabled = True
    def copy(self):
        return connection(self.masterRef, randomVal())
    def __repr__(self):
        return "ID: %i, IO: (%i,%i), enabled: %r, weight: %f " % (self.innovNum, self.masterRef.inputNode.nodeNum, self.masterRef.outputNode.nodeNum,self.enabled,self.weight)

class genome:
    def __init__(self, ID,nodeGenes, connectionGenes):
        self.ID = ID
        self.nodeGenes = nodeGenes
        self.connectionGenes = connectionGenes
        self.fitness = -1
    def copy(self, ID):
        nodeCopies = []
        connectionCopies = []
        for nodeGene in self.nodeGenes:
            nodeCopies.append(nodeGene.copy())
        for connectionGene in self.connectionGenes:
            connectionCopies.append(connectionGene.copy())
        return genome(ID, nodeCopies,connectionCopies)

    def addConnection(self,connectionIn,weight):
        newConnection = connection(connectionIn,weight)
        self.connectionGenes.append(newConnection)

    def addNode(self, nodeIn, bias):
        newNode = node(nodeIn, bias)
        self.nodeGenes.append(newNode)

    def transcodeNetwork(self,inputs,outputs):
        self.net = network(self,inputs,outputs)
        self.net.parseGenome()

    def train(self,inputData,outputData,epochs,learningRate):
        self.net.train(inputData,outputData,epochs,learningRate)

    def getFitness(self,inputData,outputData):
        self.net.evaluate(inputData,outputData)
        self.fitness = self.net.evalutateFitness()

    def verifyStructure(self):
        inputNodesInConnections = [n.masterRef.inputNode for n in self.connectionGenes if n.enabled ==True]
        outputNodesInConnections = [n.masterRef.outputNode for n in self.connectionGenes if n.enabled ==True]
        totalEnabledNodesInConnections = list(set(inputNodesInConnections).union(set(outputNodesInConnections)))
        enabledNodes = [n.masterRef for n in self.nodeGenes if n.enabled == True]
        for currNode in enabledNodes:
            if currNode not in totalEnabledNodesInConnections:
                pdb.set_trace()


    def printTopology(self):
        outStr = "Genome" + str(self.ID) + "topology: \n"
        sortedByDepth = sorted([n for n in self.nodeGenes if n.enabled is True], key = lambda x: x.masterRef.depth)
        for currNode in sortedByDepth:
            outStr += "\t Node: " + str(currNode.nodeNum) + "\n\t Depth:"+str(currNode.masterRef.depth)+  "\n\t\t Connections: ["
            for innov in currNode.masterRef.outputInnovations:
                if innov.innovNum in [n.innovNum for n in self.connectionGenes if n.enabled ==True]:
                    outStr+= str(innov.outputNode.nodeNum) + ", "
            outStr += "] \n"
        print(outStr)

    def __repr__(self):
        return "GENOME: %i, Node Genes: %i, connection Genes: %i" %(self.ID, len(self.nodeGenes),len(self.connectionGenes))


from ENN.EXAMMV2.dataStructs import *
from itertools import groupby
from dataclasses import dataclass
import numpy as np
import pdb
import math

class networkParallel:
    def __init__(self,inputs,outputs):
        self.inputs = inputs
        self.outputs = outputs
        self.activationFunction = lambda x: 1/(1+math.exp(-x)) # logistic
        self.dL_Do = lambda o,t: o-t # MSE derivative
        #self.dPhi_dnet = lambda o: 1-np.square(o) #htan deriv
        self.dPhi_dnet = lambda o: np.multiply(o,(1-o)) # logistic deriv
        self.complexityDependence = .2

    def parseGenome(self, genomeIn):
        # get enabled nodes and connections
        # build dependance map
        genomeConnMap = []
        genomeNodeMap = []
        self.connectionMap = []
        self.backPropConnectionMap = []
        for ng in genomeIn.nodeGenes:
            self.nodes.append(miniNode(ng.nodeNum,ng.bias))
        for cg in genomeIn.connectionGenes:
            self.connections.append(miniConn(ng.innovNum, ng.inputNode.nodeNum, ng.outputNode.nodeNum, ng.weight))
        # group connections by output node
        sortedConns = sorted(genomeIn.connectionGenes, key = lambda x : x.outputNode.nodeNum)
        for _, g in groupby(sortedConns, lambda x : x.outputNode.nodeNum):
            genomeConnMap.append(list(g))
        # sort by output node depth
        genomeConnMap.sort(key = lambda x : x[0].outputNode.depth)
        sortedNodes = sorted(genomeIn.nodeGenes, key = lambda x : x.nodeNum)
        sortedNodes.sort(key = lambda x : x.depth)
        for connSet in genomeConnMap:
            self.connectionMap.append([n.inputNode.nodeNum for n in connSet])
        pdb.set_trace()
        # create a lookup table for nodes and their index in delta/net arrays
        self.structureLUT = dict([(n[0],i) for i,n in enumerate(self.nodes)])
        # create list for backprop
        self.connections.sort(key = lambda x : x[0][0])
        for _,g in groupby(self.connections,lambda x: x[0][0]):
            self.backPropConnectionMap.append(list(g))
        self.backPropConnectionMap.sort(key = lambda x : x[0][0][1],reverse=True)

    def updateGenome(self,genomeIn):
        for ng in self.nodes:
            genomeIn.nodeGenes[ng[3]] = ng[1]
        for cg in self.connections:
            genomeIn.connectionGenes[cg[3]] = cg[2]

    def feedForward(self,inputData):
        net = np.zeros(len(self.nodes))
        net[0:self.inputs] = inputData
        for ind,currNode in enumerate(self.connectionMap):
            for conn in currNode:
                net[self.structureLUT[conn[1][0]]] += conn[2] * net[self.structureLUT[conn[0][0]]]
            net[self.structureLUT[conn[1][0]]] = self.activationFunction(net[self.structureLUT[conn[1][0]]] + self.nodes[self.structureLUT[currNode[0][1][0]]][1])
            #print(a)
        return net[-self.outputs:]
    def backProp(self,inputData,inputLabel,learningRate):
        # feed forward
        net = np.zeros(len(self.nodes))
        net[0:self.inputs] = inputData
        for ind,currNode in enumerate(self.connectionMap):
            for conn in currNode:
                net[self.structureLUT[conn[1][0]]] += conn[2] * net[self.structureLUT[conn[0][0]]]
            net[self.structureLUT[conn[1][0]]] = self.activationFunction(net[self.structureLUT[conn[1][0]]] + self.nodes[self.structureLUT[currNode[0][1][0]]][1])
        delta = np.zeros(len(net))
        nodeLen = len(net)-1
        delta[-self.outputs:] = np.multiply(self.dPhi_dnet(net[-self.outputs:]),self.dL_Do(net[-self.outputs:],inputLabel))
        for ind, currNode in enumerate((self.backPropConnectionMap)):
            for conn in currNode:
                delta[self.structureLUT[conn[0][0]]] += conn[2] * delta[self.structureLUT[conn[1][0]]]
            delta[self.structureLUT[conn[0][0]]] *= self.dPhi_dnet(net[self.structureLUT[conn[0][0]]])
        for conn in self.connections:
            conn[2] += -learningRate * net[self.structureLUT[conn[0][0]]] * delta[self.structureLUT[conn[1][0]]]
        for currNode in self.nodes:
            currNode[1] += -learningRate * delta[self.structureLUT[currNode[0]]]

    def train(self,inputVec, outputVec,epochs,learningRate):
        for epoch in range(0,epochs):
            for ind,data in enumerate(inputVec):
                self.backProp(data,outputVec[ind],learningRate)

    def evaluate(self,inputVec,outputVec):
        MSE = np.zeros(len(inputVec))
        for ind,data in enumerate(inputVec):
            output = self.feedForward(data)
            MSE[ind] = np.mean(np.square(output - np.array(outputVec[ind])))
        self.MSE = np.mean(MSE)
        return self.MSE

    def evalutateFitness(self):
        return 1/self.MSE - self.complexityDependence*(1/self.MSE * 1/self.getComplexity())
    def getComplexity(self):
        c =  len(self.connections)/(self.inputs * self.outputs)
        return c

@dataclass
class miniNode:
    nodeNum: int
    bias: float

@dataclass
class miniConn:
    innovNum: int
    inputNodeNum: int
    outputNodeNum: int
    weight: float

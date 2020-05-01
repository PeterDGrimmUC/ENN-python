from ENN.EXAMMV2.dataStructs import *
from itertools import groupby
import numpy as np
import pdb
import math
class network:
    def __init__(self,genomeIn,inputs,outputs):
        self.genomeIn = genomeIn
        self.inputs = inputs
        self.outputs = outputs
        self.activationFunction = lambda x: 1/(1+math.exp(-x)) # logistic
        self.dL_Do = lambda o,t: o-t # MSE derivative
        #self.dPhi_dnet = lambda o: 1-np.square(o) #htan deriv
        self.dPhi_dnet = lambda o: np.multiply(o,(1-o)) # logistic deriv
        self.complexityDependence = .2

    def parseGenome(self):
        # get enabled nodes and connections
        self.nodes = [n for n in self.genomeIn.nodeGenes if n.enabled == True]
        self.connections = [n for n in self.genomeIn.connectionGenes if n.enabled == True]

        # build dependance map
        self.connectionMap = []
        self.backPropConnectionMap = []

        # group connections by output node
        self.connections.sort(key = lambda x : x.outputNode.nodeNum)
        for _, g in groupby(self.connections, lambda x : x.outputNode.nodeNum):
            self.connectionMap.append(list(g))
        # sort by output node depth
        self.connectionMap.sort(key = lambda x : x[0].outputNode.depth)
        self.nodes.sort(key = lambda x : x.nodeNum)
        self.nodes.sort(key = lambda x : x.depth)
        # create a lookup table for nodes and their index in delta/net arrays
        self.structureLUT = dict([(n.nodeNum,i) for i,n in enumerate(self.nodes)])
        # create list for backprop
        self.connections.sort(key = lambda x : x.inputNode.nodeNum)
        for _,g in groupby(self.connections,lambda x: x.inputNode.nodeNum):
            self.backPropConnectionMap.append(list(g))
        self.backPropConnectionMap.sort(key = lambda x : x[0].inputNode.depth,reverse=True)

    def feedForward(self,inputData):
        net = np.zeros(len(self.nodes))
        net[0:self.inputs] = inputData
        for ind,currNode in enumerate(self.connectionMap):
            for conn in currNode:
                net[self.structureLUT[conn.outputNode.nodeNum]] += conn.weight * net[self.structureLUT[conn.inputNode.nodeNum]]
            net[self.structureLUT[conn.outputNode.nodeNum]] = self.activationFunction(net[self.structureLUT[conn.outputNode.nodeNum]] + self.nodes[self.structureLUT[currNode[0].outputNode.nodeNum]].bias)
            #print(a)
        return net[-self.outputs:]
    def backProp(self,inputData,inputLabel,learningRate):
        # feed forward
        net = np.zeros(len(self.nodes))
        net[0:self.inputs] = inputData
        for ind,currNode in enumerate(self.connectionMap):
            for conn in currNode:
                net[self.structureLUT[conn.outputNode.nodeNum]] += conn.weight * net[self.structureLUT[conn.inputNode.nodeNum]]
            net[self.structureLUT[conn.outputNode.nodeNum]] = self.activationFunction(net[self.structureLUT[conn.outputNode.nodeNum]] + self.nodes[self.structureLUT[currNode[0].outputNode.nodeNum]].bias)
        delta = np.zeros(len(net))
        nodeLen = len(net)-1
        delta[-self.outputs:] = np.multiply(self.dPhi_dnet(net[-self.outputs:]),self.dL_Do(net[-self.outputs:],inputLabel))
        for ind, currNode in enumerate((self.backPropConnectionMap)):
            for conn in currNode:
                delta[self.structureLUT[conn.inputNode.nodeNum]] += conn.weight * delta[self.structureLUT[conn.outputNode.nodeNum]]
            delta[self.structureLUT[conn.inputNode.nodeNum]] *= self.dPhi_dnet(net[self.structureLUT[conn.inputNode.nodeNum]])
        for conn in self.connections:
            conn.weight += -learningRate * net[self.structureLUT[conn.inputNode.nodeNum]] * delta[self.structureLUT[conn.outputNode.nodeNum]]
        for currNode in self.nodes:
            currNode.bias += -learningRate * delta[self.structureLUT[currNode.nodeNum]]

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
    def evalCorrect(self, inputVec,outputVec):
        MSE = np.zeros(len(inputVec))
        corr = 0
        dPt = 0
        for ind,data in enumerate(inputVec):
            output = self.feedForward(data)
            corr = 0
            for ind2,bit in enumerate(output):
                if bit > .5 and outputVec[ind2] > .5 or bit < .5 and outputVec[ind2] < .5:
                    corr += 1
            if corr == len(outputVec[ind]):
                dPt += 1
        pdb.set_trace()
        if dPt == len(outputVec):
            return True
        else:
            return False

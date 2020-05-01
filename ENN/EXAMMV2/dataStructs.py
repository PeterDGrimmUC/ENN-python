from ENN.EXAMMV2.network import *
import math
import numpy as np
import random as rand
from itertools import groupby
from enum import Enum
import pdb
import threading
import multiprocessing
import dill

randVal = lambda : rand.uniform(-.5,.5)
# Enums
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
class node:
    def __init__(self,nodeNum, depth, bias, nodeType):
        self.nodeNum = nodeNum
        self.depth = depth
        self.nodeType = nodeType
        self.inputConnections = []
        self.outputConnections = []
        self.bias = bias
        self.activationFunction = activationFuncs.SIGMOID
        self.enabled = True

    def __repr__(self):
        return 'Node(%i,depth=%f,type=%s,en=%r)' %(self.nodeNum, self.depth,str(self.nodeType),self.enabled)
    def enable(self):
        self.enabled = True
    def disable(self):
        self.enabled = False
    def addConnection(self,connectionIn,connectionType):
        if connectionType == 'Input':
            self.inputConnections.append(connectionIn)
        elif connectionType == 'Output':
            self.outputConnections.append(connectionIn)


class masterNode:
    def __init__(self,nodeNum, depth, nodeType):
        self.nodeNum = nodeNum
        self.depth = depth
        self.nodeType = nodeType
        self.inputConnections = []
        self.outputConnections = []
        self.activationFunction = activationFuncs.SIGMOID

    def __repr__(self):
        return 'NodeNum: %i, depth = %f' %(self.nodeNum, self.depth)

    def addConnection(self,connectionIn,connectionType):
        if connectionType == 'Input':
            self.inputConnections.append(connectionIn)
        elif connectionType == 'Output':
            self.outputConnections.append(connectionIn)

    def stat(self):
        print(self.__str__())

    def createChildCopy(self):
        inpCopy = []
        outpCopy = []
        for inp in self.inputConnections:
            inpCopy.append(connection(self.nodeNum,inp.createChildCopy(),weight=randVal()))
        for outp in self.outputConnections:
            outpCopy.append(connection(self.nodeNum,outp.createChildCopy(),weight=randVal()))
        childNode =  node(self.nodeNum, depth = self.depth, bias=randVal(),nodeType=self.nodeType)
        childNode.inputConnections = inpCopy
        childNode.outputConnections = outpCopy
        return childNode

# Class Connection: connections in network
class connection:
    def __init__(self, innovNum, inputNode, outputNode,weight, connectionType=connectionTypes.STANDARD):
        self.innovNum = innovNum
        self.inputNode = inputNode
        self.outputNode = outputNode
        self.connectionType = connectionType
        self.weight = weight
        self.enabled = True

    def __repr__(self):
        return "ID: %i, IO (%i,%i),en=%r" % (self.innovNum, self.inputNode.nodeNum,self.outputNode.nodeNum,self.enabled)

    def enable(self):
        self.enabled = True

    def disable(self):
        self.enabled = False

class masterConnection:
    def __init__(self, innovNum, inputNode, outputNode, connectionType=connectionTypes.STANDARD):
        self.innovNum = innovNum
        self.inputNode = inputNode
        self.outputNode = outputNode
        self.connectionType = connectionType

    def createChildCopy(self):
        return connection(self.innovNum, self.inputNode.createChildCopy(), self.outputNode.createChildCopy(),weight = randVal())
    def __repr__(self):
        return "ID: %i, IO (%i,%i)" % (self.innovNum, self.inputNode.nodeNum,self.outputNode.nodeNum)

class masterGenome:
    def __init__(self,inputs,outputs):
        self.nodeGenes = []
        self.connectionGenes = []
        self.numNodes = 0
        self.numConnections = 0
        self.inputs = inputs
        self.outputs = outputs
        self.buildInitList(inputs,outputs)

    def buildInitList(self,inputs,outputs):
        tempInputList = []
        tempOutputList = []
        for _ in range(0,inputs):
            newMasterNode = masterNode(self.get_nodeNum(), depth=0,nodeType=nodeTypes.INPUT)
            self.nodeGenes.append(newMasterNode)
            tempInputList.append(newMasterNode)
        for _ in range(0,outputs):
            newMasterNode = masterNode(self.get_nodeNum(), depth=1,nodeType=nodeTypes.OUTPUT)
            self.nodeGenes.append(newMasterNode)
            tempOutputList.append(newMasterNode)
        for outputNode in tempOutputList:
            for inputNode in tempInputList:
                newMasterConnection = masterConnection(self.get_innovNum(),inputNode,outputNode)
                self.connectionGenes.append(newMasterConnection)
                inputNode.outputConnections.append(newMasterConnection)
                outputNode.inputConnections.append(newMasterConnection)

    def newNodeInnovation(self,depth,inputConnections=None,outputConnections=None,nodeType=nodeTypes.HIDDEN):
        # create new node innovation and assign connections
        # get new node num
        newNodeNum = self.get_nodeNum()
        # create child copy to send back to requsting genome
        #pdb.set_trace()
        newChildNode = node(newNodeNum,depth=depth,nodeType=nodeType,bias= randVal()) # TODO put this in a variable
        newMasterNode = masterNode(newNodeNum, depth=depth,nodeType=nodeTypes.HIDDEN)
        # create array of new connection innovations based on recieved GENOME nodes
        connectionInnovations = []
        if inputConnections is not None:
            for currNode in inputConnections:
                masterInputNode = self.nodeGenes[currNode.nodeNum]
                connectionInnovations.append(self.newConnectionFromNewNodeInnovation(currNode,newChildNode,masterInputNode,newMasterNode))
        if outputConnections is not None:
            for currNode in outputConnections:
                masterOutputNode = self.nodeGenes[currNode.nodeNum]
                connectionInnovations.append(self.newConnectionFromNewNodeInnovation(newChildNode,currNode,newMasterNode,masterOutputNode))
        self.nodeGenes.append(newMasterNode)
        return (newChildNode, connectionInnovations)

    def newConnectionFromNewNodeInnovation(self,inputChildNode,outputChildNode,inputMasterNode,outputMasterNode):
        # create new master connections
        newInnovNum = self.get_innovNum()
        newMasterConnection = masterConnection(newInnovNum, inputMasterNode,outputMasterNode)
        self.connectionGenes.append(newMasterConnection)
        inputMasterNode.addConnection(newMasterConnection,'Output')
        outputMasterNode.addConnection(newMasterConnection,'Input')
        # return child connection
        newChildConnection = connection(newInnovNum, inputChildNode, outputChildNode, weight = randVal())
        # add connection to child's input, output list
        inputChildNode.addConnection(newChildConnection, 'Output')
        outputChildNode.addConnection(newChildConnection, 'Input')
        return newChildConnection

    def newConnectionInnovation(self,childInputNode,childOutputNode):
        # create new connection innovation to EXISTING nodes
        newInnovNum = self.get_innovNum()
        inputNodeNum = childInputNode.nodeNum
        outputNodeNum = childOutputNode.nodeNum
        masterInputNode = self.nodeGenes[inputNodeNum]
        masterOutputNode = self.nodeGenes[outputNodeNum]
        newMasterConnection = masterConnection(newInnovNum, masterInputNode,masterOutputNode)
        newChildConnection = connection(newInnovNum, childInputNode,childOutputNode, weight = randVal())
        masterInputNode.addConnection(newMasterConnection,'Output')
        masterOutputNode.addConnection(newMasterConnection,'Input')
        self.connectionGenes.append(newMasterConnection)
        childInputNode.outputConnections.append(newChildConnection)
        childOutputNode.inputConnections.append(newChildConnection)
        return newChildConnection

    def get_nodeNum(self):
        self.numNodes += 1
        return self.numNodes - 1

    def get_innovNum(self):
        self.numConnections += 1
        return self.numConnections -1

    def checkConnectionExists(self,childInputNode,childOutputNode):
        masterInputNode = self.nodeGenes[childInputNode.nodeNum]
        masterOutputNode = self.nodeGenes[childOutputNode.nodeNum]
        for conn in masterInputNode.outputConnections:
            if conn.outputNode.nodeNum == masterOutputNode.nodeNum:
                return connection(conn.innovNum, childInputNode,childOutputNode,weight=randVal())
        return None

    def copy(self, ID):
        nodeGenes = []
        connectionGenes = []
        for ng in self.nodeGenes:
            nodeGenes.append(node(ng.nodeNum, ng.depth, bias=randVal(),nodeType=ng.nodeType))
        nodeNums = [n.nodeNum for n in nodeGenes]
        for cg in self.connectionGenes:
            inputNode = nodeGenes[nodeNums.index(cg.inputNode.nodeNum)]
            outputNode = nodeGenes[nodeNums.index(cg.outputNode.nodeNum)]
            newConn = connection(cg.innovNum,inputNode,outputNode, weight=randVal())
            connectionGenes.append(newConn)
            inputNode.outputConnections.append(newConn)
            outputNode.inputConnections.append(newConn)
        return genome(ID, self.inputs,self.outputs,nodeGenes = nodeGenes, connectionGenes = connectionGenes)

class genome:
    def __init__(self,ID,inputs,outputs,nodeGenes=[], connectionGenes=[]):
        self.ID = ID
        self.nodeGenes = nodeGenes
        self.connectionGenes = connectionGenes
        self.fitness = -1
        self.inputs = inputs
        self.outputs = outputs
        self.epochMult = 1
    def copy(self, ID):
        nodeCopies = []
        connectionCopies = []
        for nodeGene in self.nodeGenes:
            nodeCopies.append(node(nodeGene.nodeNum,depth=nodeGene.depth))
        refDict = dict([(i.nodeNum,j) for j,i in enumerate(nodeCopies)])
        for conn in self.connectionGenes:
            newConn = connection(conn.innovNum, nodeCopies[refDict[conn.inputNode.nodeNum]],nodeCopies[refDict[conn.outputNode.nodeNum]],weight=conn.weight)
            connectionCopies.append(newConn)
            nodeCopies[refDict[conn.inputNode.nodeNum]].outputConnections.append(conn)
            nodeCopies[refDict[conn.outputNode.nodeNum]].inputConnections.append(conn)
        return genome(ID,self.inputs,self.outputs, nodeGenes=nodeCopies,connectionGenes=connectionCopies)

    def addConnection(self,connectionIn,weight=-1):
        newConnection = connectionIn.copy(newWeight=weight)
        connectionIn.inputNode.addConnection(connectionIn,'Output')
        connectionIn.outputNode.addConnection(connectionIn,'Input')
        self.connectionGenes.append(newConnection)

    def transcodeNetwork(self):
        self.net = network(self,self.inputs,self.outputs)
        self.net.parseGenome()

    def transcodeNetworkParallel(self):
        net = network(self.inputs,self.outputs)
        net.parseGenome(self)
        return net

    def train(self,inputData,outputData,epochs,learningRate):
        self.net.train(inputData,outputData,epochs,learningRate)

    def getFitness(self,inputData,outputData) -> None:
        self.net.evaluate(inputData,outputData)
        self.fitness = self.net.evalutateFitness()


    def verifyStructure(self) -> bool:
        inputNodesInConnections = [n.inputNode for n in self.connectionGenes if n.enabled ==True]
        outputNodesInConnections = [n.outputNode for n in self.connectionGenes if n.enabled ==True]
        totalEnabledNodesInConnections = list(set(inputNodesInConnections).union(set(outputNodesInConnections)))
        enabledNodes = [n for n in self.nodeGenes if n.enabled == True]
        for currNode in enabledNodes:
            if currNode not in totalEnabledNodesInConnections:
                return False
        return True


    def printTopology(self) -> None:
        outStr = "Genome" + str(self.ID) + "topology: \n"
        sortedByDepth = sorted([n for n in self.nodeGenes if n.enabled is True], key = lambda x: x.depth)
        for currNode in sortedByDepth:
            outStr += "\t Node: " + str(currNode.nodeNum) + "\n\t Depth:"+str(currNode.depth)+  "\n\t\t Connections: ["
            for innov in currNode.outputConnections:
                if innov.innovNum in [n.innovNum for n in self.connectionGenes if n.enabled ==True]:
                    outStr+= str(innov.outputNode.nodeNum) + ", "
            outStr += "] \n"
        print(outStr)

    def printReverseTopology(self) -> None:
        outStr = "Genome" + str(self.ID) + "reverse topology: \n"
        sortedByDepth = sorted([n for n in self.nodeGenes if n.enabled is True], key = lambda x: x.depth,reverse=True)
        for currNode in sortedByDepth:
            outStr += "\t Node: " + str(currNode.nodeNum) + "\n\t Depth:"+str(currNode.depth)+  "\n\t\t Connections: ["
            for innov in currNode.inputConnections:
                if innov.innovNum in [n.innovNum for n in self.connectionGenes if n.enabled ==True]:
                    outStr+= str(innov.inputNode.nodeNum) + ", "
            outStr += "] \n"
        print(outStr)

    def __repr__(self):
        return "GENOME: %i, Node Genes: %i, connection Genes: %i" %(self.ID, len(self.nodeGenes),len(self.connectionGenes))



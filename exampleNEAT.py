from ENN.EXAMMV2 import *
import matplotlib.pyplot as plt
import numpy as np
# Set parameters
inputs = 2
outputs = 1
master = masterProcess(inputs,outputs)
# set data
trainingData = [[0,0],[0,1],[1,0],[1,1]]
#trainingData = [[0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1]]
trainingLabels = [[0],[1],[1],[0]]
#trainingLabels = [[0,0],[0,1],[0,1],[1,0],[0,1],[1,0],[1,0],[1,1]]
#trainingData = [[0,0,0,0],[0,0,0,1],[0,0,1,0],[0,0,1,1],[0,1,0,0],[0,1,0,1],[0,1,1,0],[0,1,1,1],[1,0,0,0],[1,0,0,1],[1,0,1,0],[1,0,1,1],[1,1,0,0],[1,1,0,1],[1,1,1,0],[1,1,1,1]]
#trainingLabels = [[0,0,0],[0,0,1],[0,1,0],[0,1,1],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[0,1,0],[0,1,1],[0,1,0],[1,0,1],[0,1,1],[1,0,0],[1,0,1],[1,1,0]]

master.set_trainingData(trainingData,trainingLabels)
master.set_testData(trainingData,trainingLabels)
# set model parameters
initpop = 500
c1 = 1
c2 = 1
c3 = 1
cutoff = 2
maxGens = 100
topNrat = .3
topNSpecRat = .8
(maxO,meanO,g) = master.evolveNEAT(initpop,c1 ,c2,c3,cutoff,maxGens,topNrat,topNSpecRat)

#initpop = 20
#c1 = 1
#c2 = 1
#c3 = .3
#cutoff = 2
#maxGens = 30
#e0 = 150
#topNrat = .3
#topNSpecRat = .6
#l0 = .3
#feedBackGenToFinish = []
#feedBackGenToFinish = []
#feedBackNode = []
#feedBackConn = []
#max0Tot = np.zeros(maxGens)
#mean0Tot = np.zeros(maxGens)
#for _ in range(0,10):
#    master = masterProcess(inputs,outputs)
#    # set data
#    #trainingData = [[0,0],[0,1],[1,0],[1,1]]
#    #trainingLabels = [[0],[1],[1],[0]]
#    master.set_trainingData(trainingData,trainingLabels)
#    master.set_testData(trainingData,trainingLabels)
#    (maxO,meanO,g,n,c) = master.evolveFeedbackParallel(initpop,c1 ,c2,c3,cutoff,maxGens,e0,l0,.3,.01,topNrat,topNSpecRat)
#    feedBackGenToFinish.append(g)
#    max0Tot += maxO
#    mean0Tot += meanO
#    feedBackNode.append(n)
#    feedBackConn.append(c)
#max0Tot = max0Tot/10
#mean0Tot = mean0Tot/10
#print(g)
#
#feedBackNodeN = []
#feedBackConnN = []
#max0TotN = np.zeros(maxGens)
#mean0TotN = np.zeros(maxGens)
#feedBackGenToFinishN = []
#for _ in range(0,10):
#    master = masterProcess(inputs,outputs)
#    # set data
#    master.set_trainingData(trainingData,trainingLabels)
#    master.set_testData(trainingData,trainingLabels)
#    (maxO,meanO,g,n,c) = master.evolveFeedbackParallel(initpop,c1 ,c2,c3,cutoff,maxGens,e0,l0,.2,.001,topNrat,topNSpecRat)
#    feedBackGenToFinishN.append(g)
#    max0TotN += maxO
#    mean0TotN += meanO
#    feedBackNodeN.append(n)
#    feedBackConnN.append(c)
#max0TotN = max0TotN/10
#mean0TotN = mean0TotN/10
#

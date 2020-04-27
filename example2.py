from ENN.EXAMMV2 import *
master = masterProcess(3,2)
inputData = [[0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1]]
labelData = [[0,0],[0,1],[0,1],[1,0],[0,1],[1,0],[1,0],[1,1]]
initPopulation = 500
cutoff = .9
c1 = .6
c2 = .3
c3 = .1
epochs = 200
learningRate = .2
maxGens = 10
master.set_trainingData(inputData, labelData)
master.evolveFeedbackParallel(initPopulation,c1,c2,c3,cutoff,maxGens,epochs,learningRate,.2,.005,.4,.6)

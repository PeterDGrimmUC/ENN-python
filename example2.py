from ENN.EXAMMV2 import *
import marketData
inputs = 30
outputs = 1
dt_outputs = 1
dt_inputs = 1
datSrc = marketData.marketData("spy")
(inputData,labelData) = datSrc.parsePriceData(inputs,outputs,dt_inputs,dt_outputs)
master = masterProcess(inputs,outputs)
initPopulation = 20
cutoff = .9
c1 = .6
c2 = .3
c3 = .1
epochs = 1
learningRate = .02
maxGens = 10
master.set_trainingData(inputData, labelData)
master.evolveFeedbackParallel(initPopulation,c1,c2,c3,cutoff,maxGens,epochs,learningRate,.2,.005,.4,.5)

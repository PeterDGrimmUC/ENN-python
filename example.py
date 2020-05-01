from ENN.EXAMMV2 import *

# Set parameters
inputs = 2
outputs = 1
master = masterProcess(inputs,outputs)
# set data
trainingData = [[0,0],[0,1],[1,0],[1,1]]
trainingLabels = [[0],[1],[1],[0]]
master.set_trainingData(trainingData,trainingLabels)
master.set_testData(trainingData,trainingLabels)
# set model parameters
master.evolveFeedbackParallel2(20, .4,.4,.2,.5,20,10,.2,.5,.01,.8,.8)


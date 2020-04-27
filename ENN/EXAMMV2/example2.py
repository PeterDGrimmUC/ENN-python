#from ENN.EXAMMV2 import *
#import ENN.EXAMMV2
import dataStructs
import network
import masterProcess
import importlib
import pdb
import threading
from multiprocessing import Process
import cProfile
#importlib.reload(dataStructs)
#importlib.reload(masterProcess)
#importlib.reload(network)
master = masterProcess.masterProcess(3,2)
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
master.spawnProcessess()
#master.mutationStressTest(initPopulation,maxGens,  epochs, learningRate, .5)
masterProcess.tic()
master.evolveFeedbackParallel(initPopulation,c1,c2,c3,cutoff,maxGens,epochs,learningRate,.2,.005,.4,.6)
cProfile,run('master.evolveFeedbackParallel(initPopulation,c1,c2,c3,cutoff,maxGens,epochs,learningRate,.2,.005,.4,.6)')
masterProcess.toc()
#masterProcess.tic()
#master.evolveFeedback(initPopulation,c1,c2,c3,cutoff,maxGens,epochs,learningRate,.2,.005,.4,.6)
#masterProcess.toc()
#cProfile.run('master.evolveFeedback(initPopulation,c1,c2,c3,cutoff,maxGens,epochs,learningRate,.5,.01,.5,.7)')
#g1 = master.newInitGenome()
#g2 = master.newInitGenome()
#master.addRandomNode(g1)
#master.addRandomNode(g1)
#master.addRandomNode(g1)
#master.addRandomNode(g1)
#for _ in range(0,20000):
#    master.addRandomConnection(g1)
#g1.transcodeNetwork()
#g1.printTopology()
#pdb.set_trace()
#g1.train(inputData,labelData,epochs,learningRate)
##g3 = master.newInitGenome()
##for i in range(0,5):
##    master.randomMutationSpecial(g1,10)
#    g1.transcodeNetwork()
#    g1.train(inputData,labelData,epochs,learningRate)
#    master.randomMutationSpecial(g2,10)
#    g2.transcodeNetwork()
#    g2.train(inputData,labelData,epochs,learningRate)
#    master.randomMutationSpecial(g3,10)
#    g3.transcodeNetwork()
#    g3.train(inputData,labelData,epochs,learningRate)
#    print(i)
#    g1 = master.mergeMethod2(g1,g2,1)
#    g2 = master.mergeMethod2(g1,g3,1)
#    g3 = master.mergeMethod2(g2,g3,1)
#
##g2 = master.newInitGenome()
##g1.printTopology()
##g2.printTopology()
##master.addRandomNode(g1)
##master.addRandomNode(g1)
##master.addRandomNode(g1)
##master.addRandomNode(g1)
##master.addRandomNode(g1)
##master.addRandomNode(g1)
##master.addRandomNode(g1)
##
##master.addRandomNode(g2)
##master.addRandomNode(g2)
##master.addRandomNode(g2)
##master.addRandomNode(g2)
##master.addRandomNode(g2)
##master.addRandomNode(g2)
##master.addRandomNode(g2)
##g1.printTopology()
##g2.printTopology()
##g3 = master.mergeMethod2(g1,g2,.5)
##g3.printTopology()
##g3.printReverseTopology()
##g3.transcodeNetwork()
##g3.train(inputData,labelData,10000,.1)
##
#

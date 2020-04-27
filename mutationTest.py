from ENN.EXAMM import *
testGenomes = 2000
testMutations = 300
master = masterProcess(5,3)
master.generatePopulation(testGenomes)
master.verbose = True
for gen in master.genomes:
    for _ in range(0,testMutations):
        master.randomMutation(gen)
    gen.verifyStructure()

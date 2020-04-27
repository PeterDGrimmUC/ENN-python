from .EXAMM import *
testGenomes = 20
testMutations = 50
master = masterProcess(5,3)
master.generatePopulation(testGenomes)
master.verbose = True
for gen in master.genomes:
    master.randomMutation(gen)

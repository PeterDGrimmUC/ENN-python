from ENN.EXAMM.dataStructs import *
from ENN.EXAMM.network import *
class masterProcess:
    def __init__(self, inputs,outputs):
        # inputs and outputs
        self.inputs = inputs
        self.outputs = outputs
        # Innovation numbers
        self.innovGlobal = 0
        self.nodeNumGlobal = 0
        # genome statistics
        self.genomeIDGlobal = 0
        self.masterNodeList = []
        self.masterConnectionList = []
        self.genomes = []
        self.randomVal = randomVal
        self.initMasterList()
        # parameters
        self.proportionConnectionsForNewNode = .3
        # other
        self.verbose = False

    def initMasterList(self):
        # create initial lists
        inputNodes = []
        outputNodes = []
        genomeNodes = []
        genomeConnections = []
        # create new input nodes
        for inputNode in range(0,self.inputs):
            inputNodes.append(self.newNodeInnovation(0, nodeType = nodeTypes.INPUT))
        # create new output nodes
        for outputNode in range(self.inputs,self.inputs+self.outputs):
            outputNodes.append(self.newNodeInnovation(1, nodeType = nodeTypes.OUTPUT))
        # create new connection innovations and associate with their nodes
        for outputNode in outputNodes:
            for inputNode in inputNodes:
                newConnection = self.newConnectionInnovation(inputNode,outputNode)
        # initalize startin genome with a list of nodes and connections
        for currNode in self.masterNodeList:
            genomeNodes.append(node(currNode,self.randomVal()))
        for currConnection in self.masterConnectionList:
            genomeConnections.append(connection(currConnection,self.randomVal()))
        self.templateGenome = self.newGenome(genomeNodes,genomeConnections)

    # Getters and setters for global vars
    def set_trainingData(self,trainingData,trainingLabels):
        self.trainingData = trainingData
        self.trainingLabels = trainingLabels
    def set_testData(self,testData,testLabels):
        self.testData = testData
        self.testLabels = testLabels
    def get_nodeNum(self):
        self.nodeNumGlobal += 1
        return self.nodeNumGlobal -1

    def get_innovationNum(self):
        self.innovGlobal += 1
        return self.innovGlobal -1

    def get_genomeID(self):
        self.genomeIDGlobal += 1
        return self.genomeIDGlobal -1

    # Helper funcs
    def newGenome(self, genomeNodes, genomeConnections):
        newID = self.get_genomeID
        return genome(newID, genomeNodes, genomeConnections)

    def newInitGenome(self):
        return self.templateGenome.copy(self.get_genomeID())

    def generatePopulation(self,pop):
        for _ in range(0,pop):
            self.genomes.append(self.newInitGenome())

    # new innovations
    def newConnectionInnovation(self, inputNode,outputNode):
        # takes a master node reference to inputNode and outputNode
        newConnection = masterConnection(self.get_innovationNum(), inputNode,outputNode)
        self.masterConnectionList.append(newConnection)
        inputNode.outputInnovations.append(newConnection)
        outputNode.inputInnovations.append(newConnection)
        #return a reference to connection innovation
        return newConnection

    def newNodeInnovation(self,depth, nodeType='Hidden'):
        newNode = masterNode(self.get_nodeNum(),nodeType=nodeType,depth=depth)
        self.masterNodeList.append(newNode)
        # return a reference to the master node
        return newNode

    ## Connection Mutation Operations
    def enableRandomConnection(self,genomeIn):
        validMasterNodes = [n.masterRef for n in genomeIn.nodeGenes if n.enabled == True]
        validConnections = [n for n in genomeIn.connectionGenes if n.enabled==False and n.masterRef.inputNode in validMasterNodes and n.masterRef.outputNode in validMasterNodes]
        if len(validConnections) > 0:
            rand.choice(validConnections).enable()
            #print("connection enabled for genome %i"%(genomeIn.ID))

    def disableRandomConnection(self,genomeIn):
        validMasterNodes = [n.masterRef for n in genomeIn.nodeGenes if n.enabled == True]
        validConnections = [n for n in genomeIn.connectionGenes if n.enabled==True and n.masterRef.inputNode in validMasterNodes and n.masterRef.outputNode in validMasterNodes]
        if len(validConnections) > 0:
            rand.choice(validConnections).disable()
            #print("connection disabled for genome %i"%(genomeIn.ID))

    def splitConnection(self,genomeIn,connectionIn):
        #genomein is genome to mutate, connectionIN is a genome connection
        inputNode = connectionIn.masterRef.inputNode
        outputNode = connectionIn.masterRef.outputNode
        newDepth = (inputNode.depth + outputNode.depth)/2
        newNode = self.newNodeInnovation(newDepth)
        newInputConnection = self.newConnectionInnovation(inputNode,newNode)
        newOutputConnection = self.newConnectionInnovation(newNode, outputNode)
        connectionIn.disable()
        genomeIn.addConnection(newInputConnection, weight=self.randomVal())
        genomeIn.addConnection(newOutputConnection, weight=self.randomVal())
        genomeIn.addNode(newNode, bias=self.randomVal())

    def splitRandomConnection(self,genomeIn):
        # split a connection randommly
        validConnections = [n for n in genomeIn.connectionGenes if n.enabled == True]
        if len(validConnections) > 0:
            connectionChoice = rand.choice(validConnections)
            self.splitConnection(genomeIn, connectionChoice)
            #print("connection split for genome %i"%(genomeIn.ID))

    def checkConnectionExists(self,inputNode,outputNode):
        # takes two masternodes and checks if there is a connection between them
        for outputInnovation in inputNode.outputInnovations:
            if outputInnovation.outputNode is outputNode:
                return outputInnovation
        return None

    def addConnection(self,genomeIn, inputNode,outputNode):
        # take two master nodes and create connection
        newConnection = self.newConnectionInnovation(inputNode,outputNode)
        genomeIn.addConnection(newConnection, self.randomVal())

    def addRandomConnection(self,genomeIn):
        viableInputNodes = [n.masterRef for n in genomeIn.nodeGenes if n.masterRef.nodeType is not nodeTypes.OUTPUT and n.enabled == True]
        if len(viableInputNodes) >1:
            inputNodeChoice = rand.choice(viableInputNodes)
            viableOutputNodes = [n.masterRef for n in genomeIn.nodeGenes if n.masterRef.depth > inputNodeChoice.depth and n.enabled == True]
            if len(viableOutputNodes) > 1:
                outputNodeChoice = rand.choice(viableOutputNodes)
                existingConnection = self.checkConnectionExists(inputNodeChoice,outputNodeChoice)
                if existingConnection is not None:
                    if existingConnection not in [n.masterRef for n in genomeIn.connectionGenes]:
                        genomeIn.addConnection(existingConnection,self.randomVal())
                        return existingConnection
                else:
                    newConnection = self.addConnection(genomeIn,inputNodeChoice, outputNodeChoice)
                    return newConnection
        return None

    def addRecurrentConnection(self):
        pass
    def addRandomRecurrentConnection(self):
        pass

    ## Node Mutation Operations
    def disableNode(self,genomeIn,nodeRef):
        # takes a genome node reference and disables the node
        # disable node in genome
        if nodeRef.enabled:
            # disable connection genes in genome
            assocInnovs = [n for n in nodeRef.masterRef.inputInnovations]  + [n for n in nodeRef.masterRef.outputInnovations]
            #[(n.disable(),print("disabled connection from %i to %i" % (n.masterRef.inputNode.nodeNum,n.masterRef.outputNode.nodeNum))) for n in genomeIn.connectionGenes if n.masterRef in assocInnovs]
            [n.disable() for n in genomeIn.connectionGenes if n.masterRef in assocInnovs]
            nodeRef.disable()
            #print("node disabled %i for genome %i"%(nodeRef.nodeNum,genomeIn.ID))

    def enableNode(self,genomeIn,nodeRef):
        # disable node in genome
        if not nodeRef.enabled:
            # disable connection genes in genome
            assocInnovs = [n for n in nodeRef.masterRef.inputInnovations]  + [n for n in nodeRef.masterRef.outputInnovations]
            enabledNodes = [n.masterRef for n in genomeIn.nodeGenes if n.enabled]
            [n.enable() for n in genomeIn.connectionGenes if n.masterRef in assocInnovs and n.masterRef.inputNode in enabledNodes and n.masterRef.outputNode in enabledNodes]
            nodeRef.enable()
            #print("node enabled %i for genome %i"%(nodeRef.masterRef.nodeNum, genomeIn.ID))

    def enableRandomNode(self, genomeIn):
        choices = [n for n in genomeIn.nodeGenes if n.enabled == False and n.masterRef.nodeType is nodeTypes.HIDDEN]
        if len(choices) > 0:
            nodeRef = rand.choice(choices)
            self.enableNode(genomeIn, nodeRef)

    def disableRandomNode(self,genomeIn):
        choices = [n for n in genomeIn.nodeGenes if n.enabled == True and n.masterRef.nodeType is nodeTypes.HIDDEN]
        if len(choices) > 0:
            nodeRef = rand.choice(choices)
            self.disableNode(genomeIn, nodeRef)
            nodeRef.disable()

    def addNode(self,genomeIn):
        # select random depth
        depth = rand.uniform(.000001,.99999)
        # determine nodes before and after current node
        genomeNodes = [n.masterRef for n in genomeIn.nodeGenes if n.enabled== True]
        lowerDepthNodes = [n for n in genomeNodes if n.depth < depth]
        higherDepthNodes = [n for n in genomeNodes if n.depth > depth]
        # determine how many nodes to add
        numNodesInGenome = len(genomeIn.nodeGenes)
        numConnectionsToAdd = math.floor(numNodesInGenome * self.proportionConnectionsForNewNode)
        numLowerDepthNodes = len(lowerDepthNodes)
        numHigherDepthNodes = len(higherDepthNodes)
        proportionLower = rand.random()
        if math.floor(proportionLower * numConnectionsToAdd) > numLowerDepthNodes:
            numLower = max(numLowerDepthNodes,1)
        else:
            numLower = max(math.floor(proportionLower * numConnectionsToAdd),1)
        if math.floor((1-proportionLower) * numConnectionsToAdd) > numHigherDepthNodes:
            numHigher = max(numHigherDepthNodes,1)
        else:
            numHigher = max(math.floor((1-proportionLower)*numConnectionsToAdd),1)

        newNode = self.newNodeInnovation(depth, nodeType = nodeTypes.HIDDEN)
        genomeIn.addNode(newNode,self.randomVal())
        lowerConnectionsToAdd = rand.sample(lowerDepthNodes, numLower)
        higherConnectionsToAdd = rand.sample(higherDepthNodes,numHigher)
        for lowerConnection in lowerConnectionsToAdd:
            newConnection = self.newConnectionInnovation(lowerConnection,newNode)
            genomeIn.addConnection(newConnection,self.randomVal())
        for higherConnection in higherDepthNodes:
            newConnection = self.newConnectionInnovation(newNode,higherConnection)
            genomeIn.addConnection(newConnection,self.randomVal())
        #print("added node for genome %i, node added: %i "%(genomeIn.ID, newNode.nodeNum))

    def splitNode(self,genomeIn):
        # get all viable nodes
        choices = [n for n in genomeIn.nodeGenes if n.enabled == True and n.masterRef.nodeType is nodeTypes.HIDDEN]
        #print(choices)
        if len(choices) > 0:
            # select a random node
            nodeToSplit = rand.choice(choices)
            masterNode = nodeToSplit.masterRef
            # get enabled connections
            enabledConnections = [n.masterRef for n in genomeIn.connectionGenes if n.enabled == True]
            masterNodeInputInnovations = [n for n in masterNode.inputInnovations if n in enabledConnections]
            numInputNodes = len(masterNodeInputInnovations)
            if numInputNodes < 1:
                print("Dead Node Found")
                self.disableNode(genomeIn, nodeToSplit)
                return
            inputsForFirstSplitNode = math.floor(numInputNodes * .5)
            newNode1 = self.newNodeInnovation(masterNode.depth, nodeType = nodeTypes.HIDDEN) # todo , allow for different depth nodes
            newNode2 = self.newNodeInnovation(masterNode.depth, nodeType = nodeTypes.HIDDEN) # todo , allow for different depth nodes
            genomeIn.addNode(newNode1,self.randomVal())
            genomeIn.addNode(newNode2,self.randomVal())
            if inputsForFirstSplitNode < 1:
                try:
                    singleConnection = masterNodeInputInnovations[0].inputNode
                except:
                    pdb.set_trace()
                newConn1 = self.newConnectionInnovation(singleConnection,newNode1)
                newConn2 = self.newConnectionInnovation(singleConnection,newNode2)
                genomeIn.addConnection(newConn1, self.randomVal())
                genomeIn.addConnection(newConn2, self.randomVal())
            else:
                firstHalf = masterNodeInputInnovations[0:inputsForFirstSplitNode]
                secondHalf = masterNodeInputInnovations[inputsForFirstSplitNode:]
                for conn in firstHalf:
                    newConn = self.newConnectionInnovation(conn.inputNode, newNode1)
                    genomeIn.addConnection(newConn,self.randomVal())
                for conn in secondHalf:
                    newConn = self.newConnectionInnovation(conn.inputNode, newNode2)
                    genomeIn.addConnection(newConn,self.randomVal())
            masterNodeOutputInnovations = [n for n in masterNode.outputInnovations if n in enabledConnections]
            for innov in masterNodeOutputInnovations:
                newConn1 = self.newConnectionInnovation(newNode1, innov.outputNode)
                newConn2 = self.newConnectionInnovation(newNode2, innov.outputNode)
                genomeIn.addConnection(newConn1,self.randomVal())
                genomeIn.addConnection(newConn2,self.randomVal())
            self.disableNode(genomeIn,nodeToSplit)
            #print("node split for genome %i,new nodes: %i and %i, disabled node : %i "%(genomeIn.ID,newNode1.nodeNum,newNode2.nodeNum, nodeToSplit.nodeNum))

    def mergeNode(self,genomeIn):
        choices = [n for n in genomeIn.nodeGenes if n.enabled == True and n.masterRef.nodeType is nodeTypes.HIDDEN]
        if len(choices) >= 2:
            randomNodes = rand.sample(choices,2)
            node1 = randomNodes[0]
            node2 = randomNodes[1]
            node1Master = node1.masterRef
            node2Master = node2.masterRef
            newNode = self.newNodeInnovation((node2Master.depth + node1Master.depth)/2, nodeType = nodeTypes.HIDDEN) # todo , allow for different depth nodes
            genomeIn.addNode(newNode,self.randomVal())
            (lowerDepthNode,higherDepthNode) = (node1Master,node2Master) if node1Master.depth < node2Master.depth else (node2Master,node1Master)
            enabledConnections = [n.masterRef for n in genomeIn.connectionGenes if n.enabled == True]
            inputsToReplicate = [n for n in lowerDepthNode.inputInnovations if n in enabledConnections]
            outputsToReplicate = [n for n in higherDepthNode.outputInnovations if n in enabledConnections]
            for inp in inputsToReplicate:
                newConn = self.newConnectionInnovation(inp.inputNode, newNode)
                genomeIn.addConnection(newConn,self.randomVal())
            for outp in outputsToReplicate:
                newConn = self.newConnectionInnovation(newNode,outp.outputNode)
                genomeIn.addConnection(newConn,self.randomVal())
            self.disableNode(genomeIn, node1)
            self.disableNode(genomeIn, node2)
            #print("node merged for genome %i,new node: %i, removed nodes: %i and %i "%(genomeIn.ID,newNode.nodeNum, node1.nodeNum,node2.nodeNum))
####################################################################################################
    def pertrubNetwork(self,genomeIn,propToPertrub,amountToPerturb):
        connectionsToPurturb = math.floor(len(genomeIn.connectionGenes) * propToPertrub)
        nodesToPurturb = math.floor(len(genomeIn.nodeGenes) * propToPertrub)
        if nodesToPurturb > 0:
            chosenNodes = rand.sample(genomeIn.nodeGenes,nodesToPurturb)
            for currNode in chosenNodes:
                currNode.bias += amountToPerturb * self.randomVal()
        if connectionsToPurturb > 0:
            chosenConns = rand.sample(genomeIn.connectionGenes, connectionsToPurturb)
            for currConn in chosenConns:
                currConn.weight += amountToPerturb * self.randomVal()

    def randomMutation(self,genomeIn):
        if rand.random() < .2:
            self.pertrubNetwork(genomeIn, .2, .3)
        if rand.random() < .1:
            self.addRandomConnection(genomeIn)
        if rand.random() < .05:
            self.addNode(genomeIn)
        if rand.random() < .05:
            self.splitNode(genomeIn)
        if rand.random() < .05:
            self.splitRandomConnection(genomeIn)
#        if rand.random() <.05:
#            self.mergeNode(genomeIn)
#        if rand.random() <.05:
#            self.disableRandomNode(genomeIn)
#        if rand.random() <.05:
#            self.enableRandomNode(genomeIn)
#        if rand.random() <.01:
#            self.enableRandomConnection(genomeIn)

    ## Genome Operations
    def fullGenomeCombination(self,genome1,genome2):
        combinedConnections = []
        (betterParent,worseParent) = (genome1,genome2) if genome1.fitness > genome2.fitness else (genome2,genome1)
        enabledGenome1Conns = [n for n in genome1.connectionGenes if n.enabled == True]
        enabledGenome2Conns = [n for n in genome2.connectionGenes if n.enabled == True]
        maxInnov1 = max([n.masterRef.innovNum for n in enabledGenome1Conns])
        maxInnov2 = max([n.masterRef.innovNum for n in enabledGenome2Conns])
        maxInnovNum = max(maxInnov1,maxInnov2)
        (longerList,shorterList) = (enabledGenome1Conns,enabledGenome2Conns) if maxInnov1 > maxInnov2 else (enabledGenome2Conns,enabledGenome1Conns)
        indexInShortList = 0
        ind = 0
        list1EndFound = 0
        list2EndFound = 0
        maxFound = 0
        #pdb.set_trace()
        shortListEndReached = 0
        longListEndReached = 0
        inputNodeFound = []
        outputNodeFound = []
        while longListEndReached is not True:
            while longerList[ind].masterRef.innovNum > shorterList[indexInShortList].masterRef.innovNum and not shortListEndReached:
                # only in list 2
                if indexInShortList+1 < len(shorterList):
                    combinedConnections.append(shorterList[indexInShortList].copy())
                    inputNodeFound.append(combinedConnections[-1].masterRef.inputNode) if combinedConnections[-1].masterRef.inputNode not in inputNodeFound else None
                    outputNodeFound.append(combinedConnections[-1].masterRef.outputNode) if combinedConnections[-1].masterRef.outputNode not in outputNodeFound else None
                    indexInShortList+=1
                    break
                else:
                    combinedConnections.append(shorterList[-1].copy())
                    inputNodeFound.append(combinedConnections[-1].masterRef.inputNode) if combinedConnections[-1].masterRef.inputNode not in inputNodeFound else None
                    outputNodeFound.append(combinedConnections[-1].masterRef.outputNode) if combinedConnections[-1].masterRef.outputNode not in outputNodeFound else None
                    shortListEndReached = True
                    break
            while longerList[ind].masterRef.innovNum < shorterList[indexInShortList].masterRef.innovNum:
                # only in list 1
                if ind<  len(longerList):
                    combinedConnections.append(longerList[ind].copy())
                    inputNodeFound.append(combinedConnections[-1].masterRef.inputNode) if combinedConnections[-1].masterRef.inputNode not in inputNodeFound else None
                    outputNodeFound.append(combinedConnections[-1].masterRef.outputNode) if combinedConnections[-1].masterRef.outputNode not in outputNodeFound else None
                    ind += 1
                else:
                    longListEndReached = True
                    break
            if longerList[ind].masterRef.innovNum == shorterList[indexInShortList].masterRef.innovNum:
                # mutual connection found
                if ind < len(longerList) and indexInShortList < len(shorterList):
                    r = rand.uniform(-.5,1.5)
                    combinedConnections.append(longerList[ind].copy())
                    combinedConnections[-1].weight = (longerList[ind].weight + shorterList[indexInShortList].weight)/2
                    inputNodeFound.append(combinedConnections[-1].masterRef.inputNode) if combinedConnections[-1].masterRef.inputNode not in inputNodeFound else None
                    outputNodeFound.append(combinedConnections[-1].masterRef.outputNode) if combinedConnections[-1].masterRef.outputNode not in outputNodeFound else None
                    ind += 1
                else:
                    longListEndReached = True
                    break
                if indexInShortList + 1< len(shorterList):
                    indexInShortList+=1
                else:
                    shortListEndReached = True
                    break
            while shortListEndReached:
                if ind< len(longerList):
                    combinedConnections.append(longerList[ind].copy())
                    inputNodeFound.append(combinedConnections[-1].masterRef.inputNode) if combinedConnections[-1].masterRef.inputNode not in inputNodeFound else None
                    outputNodeFound.append(combinedConnections[-1].masterRef.outputNode) if combinedConnections[-1].masterRef.outputNode not in outputNodeFound else None
                    ind += 1
                else:
                    longListEndReached = True
                    break
        connectedHiddenNodes = list(set(inputNodeFound).intersection(set(outputNodeFound)))
        genome1HiddenNodes = [n for n in genome1.nodeGenes]
        genome1HiddenNodesMaster = [n.masterRef for n in genome1.nodeGenes]
        genome2HiddenNodes = [n for n in genome2.nodeGenes]
        genome2HiddenNodesMaster = [n.masterRef for n in genome2.nodeGenes]
        finalNodes = [n.copy() for n in genome1.nodeGenes if n.masterRef.nodeType == nodeTypes.INPUT or n.masterRef.nodeType == nodeTypes.OUTPUT]
        for hiddenNode in connectedHiddenNodes:
            hiddenNodeIndex1 = genome1HiddenNodesMaster.index(hiddenNode) if hiddenNode in genome1HiddenNodesMaster else None
            hiddenNodeIndex2 = genome2HiddenNodesMaster.index(hiddenNode) if hiddenNode in genome2HiddenNodesMaster else None
            if hiddenNodeIndex1 is not None and hiddenNodeIndex2 is not None:
                finalNodes.append(genome1HiddenNodes[hiddenNodeIndex1].copy())
                finalNodes[-1].bias = (genome1HiddenNodes[hiddenNodeIndex1].bias + genome2HiddenNodes[hiddenNodeIndex2].bias)/2
            if hiddenNodeIndex2 is not None:
                finalNodes.append(genome2HiddenNodes[hiddenNodeIndex2].copy())
            if hiddenNodeIndex1 is not None:
                finalNodes.append(genome1HiddenNodes[hiddenNodeIndex1].copy())
        finalConnections = sorted(combinedConnections, key=lambda x: x.masterRef.innovNum)
        return genome(self.get_genomeID(), finalNodes,finalConnections)

    def cloneGenome(self,genomeIn):
        return genomeIn.copy()
    # evolution
    def simpleEvolve(self,pop,inputData,outputData):
        instances = []
        fitnessArr = np.zeros((pop))
        for _ in range(0,pop):
            instances.append(self.newInitGenome())
        for gen in range(0,10):
            for ind,inst in enumerate(instances):
                self.randomMutation(inst)
                inst.transcodeNetwork(self.inputs,self.outputs)
                inst.train(inputData,outputData,1000,.2)
                inst.getFitness(inputData,outputData)
                print(("Genome %i fitness:  %f" %(inst.ID,inst.fitness))+ " " + str(inst))
                fitnessArr[ind] = inst.fitness
            ranking = np.argsort(fitnessArr)[::-1]
            numToRemove = math.floor(len(instances)*.8)
            numToRemain = len(instances) - numToRemove
            oldinstances = [instances[n] for n in ranking[0:numToRemain]]
            instances = []
            for _ in range(0,pop):
                newClone = rand.choice(oldinstances)
                partner = rand.choice(oldinstances)
                newGenome = self.fullGenomeCombination(newClone,partner)
                instances.append(newGenome)
            print("Generation: %i"%(gen))
        return instances

    def checkCompat(self, genome1,genome2,c1,c2):
        genome1Nodes = [n.masterRef.nodeNum for n in genome1.nodeGenes]
        genome2Nodes = [n.masterRef.nodeNum for n in genome2.nodeGenes]
        genome1Conns= [n.masterRef.innovNum for n in genome1.connectionGenes]
        genome2Conns= [n.masterRef.innovNum for n in genome2.connectionGenes]
        numNodes = len(genome1Nodes) + len(genome2Nodes)
        numConns = len(genome1Conns) + len(genome2Conns)
        diffNodes = set(genome1Nodes).symmetric_difference(set(genome2Nodes))
        diffConns = set(genome1Conns).symmetric_difference(set(genome2Conns))
        numDiffNodes = len(diffNodes)
        numDiffConns = len(diffConns)
        return (c1 * numDiffConns/numConns) + (c2 * numDiffNodes/numNodes)

    def speciate(self, genomeList,cutoff,c1,c2, speciesStartList = []):
        if speciesStartList is []:
            speciesList = [[genomeList[0]]]
            del genomeList[0]
        else:
            speciesList = speciesStartList
        for currGenome in genomeList:
            suitableSpeciesFound = False
            for species in speciesList:
                for gen in species:
                    compat = self.checkCompat(currGenome,gen,c1,c2)
                    if compat <cutoff:
                        suitableSpeciesFound = True
                        species.append(currGenome)
                        break
            if not suitableSpeciesFound:
                speciesList.append([])
                speciesList[-1].append(currGenome)
        return speciesList

    def evolveSpeciated(self,pop,c1,c2,cutoff,inputData,outputData,maxGenerations,epochs,learningRate):
        instances = []
        speciesStartList = []
        fitnessArr = np.zeros((pop))
        for _ in range(0,pop):
            instances.append(self.newInitGenome())
        for gen in range(0,maxGenerations):
            #pdb.set_trace()
            print("There are %i total genomes"%(len(instances)))
            for inst in instances:
                for _ in range(0,3):
                    self.randomMutation(inst)
            speciesList = self.speciate(instances, cutoff,c1,c2, speciesStartList = speciesStartList)
            speciesStartList = []
            print("there are %i total species"%(len(speciesList)))
            for species in speciesList:
                (instances,best) = self.evaluateSpecies(species,.8,inputData,outputData,epochs,learningRate)
            print("current generation: %i"%(gen))

    def evaluateSpecies(self,speciesGenomes,proportionToElim,inputData,outputData,epochs,learningRate):
        numGenomes = len(speciesGenomes)
        outputGenomes = []
        fitnessArr = np.zeros(numGenomes)
        for ind,gen in enumerate(speciesGenomes):
            gen.transcodeNetwork(self.inputs,self.outputs)
            gen.train(inputData,outputData,epochs,learningRate)
            gen.getFitness(inputData,outputData)
        #pdb.set_trace()
        fitnessArr = np.array([n.fitness for n in speciesGenomes])/numGenomes
        #print(fitnessArr)
        numToRemove = math.floor(proportionToElim * numGenomes)
        genToRemove = fitnessArr.argsort()[:numToRemove][::-1]
        successfulGenomes = [i for j, i in enumerate(speciesGenomes) if j not in genToRemove]
        numToReplace = numGenomes - numToRemove
        bestFitness = max(fitnessArr.tolist())
        bestGenome = speciesGenomes[fitnessArr.tolist().index(bestFitness)]
        for n in range(0,numGenomes):
            choice1 = rand.choice(successfulGenomes)
            choice2 = rand.choice(successfulGenomes)
            outputGenomes.append(self.fullGenomeCombination(choice1,choice2))
        return (outputGenomes, bestGenome)
    def evaluateSpeciesThreaded(self,speciesGenomes,proportionToElim,inputData,outputData,epochs,learningRate):
        numGenomes = len(speciesGenomes)
        outputGenomes = []
        newSpeciesGenomes = []
        fitnessArr = np.zeros(numGenomes)
        for ind,gen in enumerate(speciesGenomes):
            gen.transcodeNetwork(self.inputs,self.outputs)
            gen.train(inputData,outputData,epochs,learningRate)
            gen.getFitness(inputData,outputData)
        #pdb.set_trace()
        fitnessArr = np.array([n.fitness for n in speciesGenomes])/numGenomes
        #print(fitnessArr)
        numToRemove = math.floor(proportionToElim * numGenomes)
        genToRemove = fitnessArr.argsort()[:numToRemove][::-1]
        successfulGenomes = [i for j, i in enumerate(speciesGenomes) if j not in genToRemove]
        numToReplace = numGenomes - numToRemove
        bestFitness = max(fitnessArr.tolist())
        bestGenome = speciesGenomes[fitnessArr.tolist().index(bestFitness)]
        for n in range(0,numGenomes):
            choice1 = rand.choice(successfulGenomes)
            choice2 = rand.choice(successfulGenomes)
            newSpeciesGenomes.append(self.fullGenomeCombination(choice1,choice2))
        speciesGenomes = newSpeciesGenomes

    def evolveSpeciatedThreaded(self,pop,c1,c2,cutoff,inputData,outputData,maxGenerations,epochs,learningRate):
        instances = []
        threads = []
        speciesStartList = []
        fitnessArr = np.zeros((pop))
        for _ in range(0,pop):
            instances.append(self.newInitGenome())
        for gen in range(0,maxGenerations):
            print("There are %i total genomes"%(len(instances)))
            for inst in instances:
                for _ in range(0,3):
                    self.randomMutation(inst)
            speciesList = self.speciate(instances, cutoff,c1,c2, speciesStartList = speciesStartList)
            speciesStartList = []
            threads = []
            print("There are %i total species"%(len(speciesList)))
            for species in speciesList:
                newThread = threading.Thread(target=self.evaluateSpeciesThreaded,args=(species,.8,inputData,outputData,epochs,learningRate))
                threads.append(newThread)
                threads[-1].start()
            for thread in threads:
                thread.join()
            print("current generations: %i"%(gen))
        return


from ENN.EXAMMV2.dataStructs import *
from ENN.EXAMMV2.networkParallel import *
from ENN.EXAMMV2.network import *
from multiprocessing import Process,Queue,cpu_count,Pipe
from pathos.multiprocessing import ProcessingPool as Pool
# This version trades computational complexity for space complexity, requires more space but less calculations
# each genome node and connection contains all information needed to carry out mutations
randomVal = lambda : rand.uniform(-.5,.5)
class masterProcess:
    def __init__(self, inputs,outputs):
        # inputs and outputs
        self.inputs = inputs
        self.outputs = outputs
        # genome statistics
        self.genomeIDGlobal = 0
        # master genome keeps track of innovations (god willing)
        self.masterGenome = masterGenome(inputs,outputs)
        self.genomes = []
        self.randomVal = randomVal
        # parameters
        self.proportionConnectionsForNewNode = .3
        # other
        self.verbose = False

    # Getters and setters for global vars
    def set_trainingData(self,trainingData,trainingLabels):
        self.trainingData = trainingData
        self.trainingLabels = trainingLabels

    def set_testData(self,testData,testLabels):
        self.testData = testData
        self.testLabels = testLabels

    def get_genomeID(self) -> int:
        self.genomeIDGlobal += 1
        return self.genomeIDGlobal -1

    # Helper funcs
    def newGenome(self, genomeNodes, genomeConnections) -> genome:
        newID = self.get_genomeID
        return genome(newID, genomeNodes, genomeConnections)

    def newInitGenome(self) -> genome:
        newGenomeNum = self.get_genomeID()
        return self.masterGenome.copy(newGenomeNum)

    def generatePopulation(self,pop) -> None:
        for _ in range(0,pop):
            self.genomes.append(self.newInitGenome())

    # new innovations
    def newConnectionInnovation(self, genomeIn, inputNode,outputNode) -> None:
        # takes a master node reference to inputNode and outputNode, adds the connection
        newConnection = self.masterGenome.newConnectionInnovation(inputNode,outputNode)
        # add to genome
        genomeIn.connectionGenes.append(newConnection)

    def newNodeInnovation(self,genomeIn,depth,inputConnections=[],outputConnections=[], nodeType=nodeTypes.HIDDEN) -> None:
        # inputConnecttions/outputConnections = list of node references to add
        (newNode,assocConnections) = self.masterGenome.newNodeInnovation(depth,inputConnections,outputConnections,nodeType=nodeType)
        # add to genome
        genomeIn.nodeGenes.append(newNode)
        # add connections to genome
        for conn in assocConnections:
            genomeIn.connectionGenes.append(conn)

    def newNodeAlt(self, genomeIn, depth, inputConnections,outputConnections,nodeType):
        newNodeNum = self.masterGenome.get_nodeNum()
        newNode = node(newNodeNum, depth, self.randomVal(), nodeType)
        newMasterNode = masterNode(newNodeNum, depth, nodeType)
        genomeIn.nodeGenes.append(newNode)
        self.masterGenome.nodeGenes.append(newMasterNode)
        for inp in inputConnections:
            # create new connection objects
            newInnovNum = self.masterGenome.get_innovNum()
            newConn = connection(newInnovNum, inp, newNode, self.randomVal())
            newMasterConn = masterConnection(newInnovNum, self.masterGenome.nodeGenes[inp.nodeNum], newMasterNode)
            masterInp = self.masterGenome.nodeGenes[inp.nodeNum]
            newNode.inputConnections.append(newConn)
            newMasterNode.inputConnections.append(newMasterConn)
            # append new connection to genomes
            genomeIn.connectionGenes.append(newConn)
            self.masterGenome.connectionGenes.append(newMasterConn)
            # append new connection to output of old node
            inp.outputConnections.append(newConn)
            masterInp.outputConnections.append(newMasterConn)
        for outp in outputConnections:
            newInnovNum = self.masterGenome.get_innovNum()
            newConn = connection(newInnovNum, newNode, outp, self.randomVal())
            newMasterConn = masterConnection(newInnovNum, newMasterNode, self.masterGenome.nodeGenes[outp.nodeNum])
            masteroutp = self.masterGenome.nodeGenes[outp.nodeNum]
            newNode.outputConnections.append(newConn)
            newMasterNode.outputConnections.append(newMasterConn)
            genomeIn.connectionGenes.append(newConn)
            outp.inputConnections.append(newConn)
            masteroutp.inputConnections.append(newMasterConn)
            self.masterGenome.connectionGenes.append(newMasterConn)

    def newConnectionAlt(self, genomeIn, inputNode, outputNode):
        # check if connection exists in master genome
        masterInputNode = self.masterGenome.nodeGenes[inputNode.nodeNum]
        masterOutputNode = self.masterGenome.nodeGenes[outputNode.nodeNum]
        existingInnovNum = None
        for conn in self.masterGenome.connectionGenes:
            if conn.inputNode is masterInputNode and conn.outputNode is masterOutputNode:
                existingInnovNum = conn.innovNum
                break
        if existingInnovNum is not None:
            # existing connection
            if existingInnovNum not in [n.innovNum for n in genomeIn.connectionGenes]:
                newConn = connection(existingInnovNum, inputNode, outputNode, self.randomVal())
                genomeIn.connectionGenes.append(newConn)
                inputNode.outputConnections.append(newConn)
                outputNode.inputConnections.append(newConn)
        else:
            newInnovNum = self.masterGenome.get_innovNum()
            newMasterConn = masterConnection(newInnovNum, masterInputNode,masterOutputNode)
            newConn = connection(newInnovNum, inputNode, outputNode, self.randomVal())
            genomeIn.connectionGenes.append(newConn)
            self.masterGenome.connectionGenes.append(newMasterConn)
            inputNode.outputConnections.append(newConn)
            outputNode.inputConnections.append(newConn)
            masterInputNode.outputConnections.append(newMasterConn)
            masterOutputNode.inputConnections.append(newMasterConn)

    ## Connection Mutation Operations
    def enableRandomConnection(self,genomeIn) -> None:
        choices = [n for n in genomeIn.connectionGenes if n.enabled == False and n.inputNode.enabled == True and n.outputNode.enabled==True]
        if len(choices) > 0:
            rand.choice(choices).enable()

    def disableRandomConnection(self,genomeIn) -> None:
        choices = [n for n in genomeIn.connectionGenes if n.enabled == True]
        if len(choices) > 0:
            rand.choice(choices).enable()

    def splitConnection(self,genomeIn,connectionIn) -> None:
        #genomein is genome to mutate, connectionIN is a genome connection
        inputNode = connectionIn.inputNode
        outputNode = connectionIn.outputNode
        newDepth = (inputNode.depth + outputNode.depth)/2
        self.newNodeAlt(genomeIn, newDepth, [inputNode],[outputNode], nodeTypes.HIDDEN)
        connectionIn.disable()

    def splitRandomConnection(self,genomeIn) -> None:
        # split a connection randommly
        validConnections = [n for n in genomeIn.connectionGenes if n.enabled == True]
        if len(validConnections) > 0:
            connectionChoice = rand.choice(validConnections)
            self.splitConnection(genomeIn, connectionChoice)

    def checkConnectionInnovationExists(self,inputNode,outputNode) -> connection:
        # takes two genome nodes and checks if there is a connection between them
        innov = self.masterGenome.checkConnectionExists(inputNode,outputNode)
        return innov

    def addConnection(self,genomeIn, inputNode,outputNode) -> None:
        # take two master nodes and create connection
        newConnection = self.newConnectionAlt(genomeIn, inputNode,outputNode)

    def addRandomConnection(self,genomeIn) -> None:
        # add a random connection between nodes
        enabledNodes = [n for n in genomeIn.nodeGenes if n.enabled == True]
        viableInputNodes = [n for n in enabledNodes if n.nodeType is not nodeTypes.OUTPUT]
        if len(viableInputNodes) >0:
            inputNodeChoice = rand.choice(viableInputNodes)
            viableOutputNodes = [n for n in enabledNodes if n.depth > inputNodeChoice.depth]
            if len(viableOutputNodes) > 0:
                outputNodeChoice = rand.choice(viableOutputNodes)
                self.addConnection(genomeIn, inputNodeChoice,outputNodeChoice)

    def addRecurrentConnection(self):
        pass
    def addRandomRecurrentConnection(self):
        pass

    ## Node Mutation Operations
    def disableNode(self,nodeRef) -> None:
        # takes a genome node reference and disables the node
        # disable node in genome
        [n.disable() for n in (nodeRef.inputConnections + nodeRef.outputConnections)]
        nodeRef.disable()

    def enableNode(self,nodeRef) -> None:
        # disable node in genome
        # disable connection genes in genome
        [n.enable() for n in nodeRef.inputConnections  if n.inputNode.enabled == True]
        [n.enable() for n in nodeRef.outputConnections  if n.outputNode.enabled == True]
        nodeRef.enable()

    def enableRandomNode(self, genomeIn) -> None:
        choices = [n for n in genomeIn.nodeGenes if n.enabled == False]
        if len(choices) > 0:
            nodeRef = rand.choice(choices)
            self.enableNode(nodeRef)

    def disableRandomNode(self,genomeIn) -> None:
        choices = [n for n in genomeIn.nodeGenes if n.enabled == True and n.nodeType is nodeTypes.HIDDEN]
        if len(choices) > 0:
            nodeRef = rand.choice(choices)
            self.disableNode(nodeRef)

    def addRandomNode(self,genomeIn) -> None:
        # select random depth
        depth = rand.uniform(.000001,.99999)
        # determine nodes before and after current node
        self.addNode(genomeIn,depth, .5)

    def addNode(self,genomeIn,depth,proportConns) -> None:
        # select random depth
        # determine nodes before and after current node
        genomeNodes = [n for n in genomeIn.nodeGenes if n.enabled== True]
        lowerDepthNodes = [n for n in genomeNodes if n.depth < depth]
        higherDepthNodes = [n for n in genomeNodes if n.depth > depth]
        # determine how many nodes to add
        numLowerNodesToAdd = math.ceil(max(1, len(lowerDepthNodes) * proportConns)) # TODO, possibly make this a variable
        numHigherNodesToAdd = math.ceil(max(1, len(higherDepthNodes) * proportConns))
        lowerNodesToAdd = rand.sample(lowerDepthNodes,numLowerNodesToAdd)
        higherNodesToAdd = rand.sample(higherDepthNodes,numHigherNodesToAdd)
        self.newNodeAlt(genomeIn,depth, lowerNodesToAdd, higherNodesToAdd, nodeTypes.HIDDEN)

    def addFullLayer(self,genomeIn,numNodes,depth) -> None:
        for _ in range(0,numNodes):
            self.addNode(genomeIn, depth, 1)
        #disable nodes that cross the new layer
        [n.disable() for n in genomeIn.connectionGenes if n.outputNode.depth > depth and n.inputNode.depth <depth]

    def splitNode(self,genomeIn) -> None:
        # get all viable nodes
        choices = [n for n in genomeIn.nodeGenes if n.enabled == True and n.nodeType is nodeTypes.HIDDEN]
        if len(choices) > 0:
            # select a random node
            nodeToSplit = rand.choice(choices)
            inputConns = [n for n in nodeToSplit.inputConnections if n.enabled == True]
            outputConns = [n for n in nodeToSplit.outputConnections if n.enabled == True]
            if len(inputConns) == 1:
                node1Conns = inputConns
                node2Conns = inputConns
            elif len(inputConns) > 1:
                numInputsForNode1 = math.floor(len(inputConns) * .5)
                node1Conns = inputConns[0:numInputsForNode1]
                node2Conns = inputConns[numInputsForNode1:]
            node1Inputs = [n.inputNode for n in node1Conns]
            node2Inputs = [n.inputNode for n in node2Conns]
            outputNodes = [n.outputNode for n in outputConns]
            self.newNodeAlt(genomeIn,nodeToSplit.depth, node1Inputs,outputNodes,nodeTypes.HIDDEN)
            self.newNodeAlt(genomeIn,nodeToSplit.depth, node2Inputs,outputNodes, nodeTypes.HIDDEN)
            self.disableNode(nodeToSplit)


    def mergeNode(self,genomeIn) -> None:
        choices = [n for n in genomeIn.nodeGenes if n.enabled == True and n.nodeType is nodeTypes.HIDDEN]
        if len(choices) >= 2:
            (node1,node2) = rand.sample(choices,2)
            (lowerDepthNode,higherDepthNode) = (node1,node2) if node1.depth < node2.depth else (node2,node1)
            inputsToReplicate = [n.inputNode for n in lowerDepthNode.inputConnections if n.enabled == True]
            outputsToReplicate = [n.outputNode for n in higherDepthNode.outputConnections if n.enabled == True]
            newDepth = (lowerDepthNode.depth + higherDepthNode.depth)/2
            self.newNodeAlt(genomeIn, newDepth, inputsToReplicate, outputsToReplicate, nodeTypes.HIDDEN)
            self.disableNode(node1)
            self.disableNode(node2)

    def pertrubNetwork(self,genomeIn,propToPertrub,amountToPerturb) -> None:
        connectionsToPurturb = math.floor(len(genomeIn.connectionGenes) * propToPertrub)
        nodesToPurturb = math.floor(len(genomeIn.nodeGenes) * propToPertrub)
        nodesToReset = math.floor(len(genomeIn.nodeGenes) * .2)
        connsToReset = math.floor(len(genomeIn.connectionGenes) * .2)
        if nodesToPurturb > 0:
            chosenNodes = rand.sample(genomeIn.nodeGenes,nodesToPurturb)
            for currNode in chosenNodes:
                currNode.bias += amountToPerturb * self.randomVal()
        if connectionsToPurturb > 0:
            chosenConns = rand.sample(genomeIn.connectionGenes, connectionsToPurturb)
            for currConn in chosenConns:
                currConn.weight += amountToPerturb * self.randomVal()
        if nodesToReset > 0:
            chosenNodes = rand.sample(genomeIn.nodeGenes,nodesToReset)
            for currNode in chosenNodes:
                currNode.bias = self.randomVal()
        if connsToReset > 0:
            chosenConns = rand.sample(genomeIn.connectionGenes, connsToReset)
            for currConn in chosenConns:
                currConn.weight = self.randomVal()
    def randomMutation(self,genomeIn) -> None:
        if rand.random() < .2:
            self.pertrubNetwork(genomeIn, .2, .3)
        if rand.random() < .1:
            self.addRandomConnection(genomeIn)
        if rand.random() < .05:
            self.addRandomNode(genomeIn)
        if rand.random() < .05:
            self.splitNode(genomeIn)
        if rand.random() < .05:
            self.splitRandomConnection(genomeIn)
        if rand.random() <.05:
            self.mergeNode(genomeIn)
        if rand.random() <.05:
            self.disableRandomNode(genomeIn)
        if rand.random() < .01:
            self.addFullLayer(genomeIn, 5, rand.uniform(.0001,.9999))
        if rand.random() <.05:
            self.enableRandomNode(genomeIn)
        if rand.random() <.01:
            self.enableRandomConnection(genomeIn)

    def randomMutationSpecial(self,genomeIn,mutationMuliplier) -> float:
        trainingMultiplier = 0
        if rand.random() < .2 * mutationMuliplier :
            self.pertrubNetwork(genomeIn, .2, .3)
        if rand.random() < .3 * mutationMuliplier:
            self.addRandomConnection(genomeIn)
            trainingMultiplier += .1
        if rand.random() < .1 * mutationMuliplier:
            self.addRandomNode(genomeIn)
            trainingMultiplier += 2
        if rand.random() < .05 * mutationMuliplier:
            self.splitNode(genomeIn)
            trainingMultiplier += 4
        if rand.random() < .05 * mutationMuliplier:
            self.splitRandomConnection(genomeIn)
            self.findDisconnectedGenome(genomeIn)
            trainingMultiplier += 2
        if rand.random() <.1 * mutationMuliplier:
            self.mergeNode(genomeIn)
            trainingMultiplier += 4
        if rand.random() < .2 * mutationMuliplier:
            self.disableRandomNode(genomeIn)
            trainingMultiplier += 4
        if rand.random() < .3 * mutationMuliplier:
            self.disableRandomConnection(genomeIn)
            trainingMultiplier += .4
        if rand.random() < .01 * mutationMuliplier:
            maxNodes = (self.inputs + self.outputs)
            nodesToAdd = math.floor(rand.uniform(1,maxNodes+1))
            self.addFullLayer(genomeIn, nodesToAdd, rand.uniform(.0001,.9999))
            trainingMultiplier += nodesToAdd*.5
        if rand.random() <.001 * mutationMuliplier:
            self.enableRandomNode(genomeIn)
            trainingMultiplier += 1
        if rand.random() <.001 * mutationMuliplier:
            self.enableRandomConnection(genomeIn)
            trainingMultiplier += .2
        return trainingMultiplier

    def randomMutationNEAT(self,genomeIn,mutationMuliplier):
        if rand.random() < 1:# * mutationMuliplier :
            self.pertrubNetwork(genomeIn, .8, .5)
        if rand.random() < .05:# * mutationMuliplier:
            self.addRandomConnection(genomeIn)
        if rand.random() < .03:# * mutationMuliplier:
            self.splitRandomConnection(genomeIn)
        if rand.random() <.03: # * mutationMuliplier:
            self.enableRandomConnection(genomeIn)
        if rand.random() <.03: # * mutationMuliplier:
            self.disableRandomConnection(genomeIn)

    def randomMutationVerify(self,genomeIn,mutationMuliplier):
        trainingMultiplier = .1
        print("Mutating genome %i"%(genomeIn.ID))
        if rand.random() < .2 * mutationMuliplier :
            self.pertrubNetwork(genomeIn, .2, .3)
        if rand.random() < .2 * mutationMuliplier:
            self.addRandomConnection(genomeIn)
            trainingMultiplier += .1
            print("\tadded connection")
        if rand.random() < .05 * mutationMuliplier:
            self.addRandomNode(genomeIn)
            trainingMultiplier += .3
            print("\tadded node")
        if rand.random() < .05 * mutationMuliplier:
            self.splitNode(genomeIn)
            trainingMultiplier += .3
            print("\tsplit node")
        if rand.random() < .05 * mutationMuliplier:
            self.splitRandomConnection(genomeIn)
            self.findDisconnectedGenome(genomeIn)
            trainingMultiplier += .3
            print("\tsplit connection")
        if rand.random() <.05 * mutationMuliplier:
            self.mergeNode(genomeIn)
            trainingMultiplier += 1
            print("\tsplit connection")
        if rand.random() <.05 * mutationMuliplier:
            self.disableRandomNode(genomeIn)
            trainingMultiplier += .3
            self.findDisconnectedGenome(genomeIn)
            print("\tDisabled node")
        if rand.random() < .01 * mutationMuliplier:
            maxNodes = (self.inputs + self.outputs)
            nodesToAdd = math.floor(rand.uniform(1,maxNodes+1))
            self.addFullLayer(genomeIn, nodesToAdd, rand.uniform(.0001,.9999))
            trainingMultiplier += nodesToAdd
            print("\tadded full layer")
        if rand.random() <.05 * mutationMuliplier:
            self.enableRandomNode(genomeIn)
            trainingMultiplier += 1
            print("\tenabled node ")
        if rand.random() <.01 * mutationMuliplier:
            self.enableRandomConnection(genomeIn)
            trainingMultiplier += .5
            print("\tenabled connection ")
        print("\n")
        return trainingMultiplier

    ## Genome Operations

    def cloneGenome(self,genomeIn):
        return genomeIn.copy()

    def geneCompatMethod2(self,genome1,genome2,c1,c2,c3):
        if genome1 is genome2:
            return c1 + c2 + c3
        (moreFitParent,lessFitParent) = (genome1,genome2) if genome1.fitness > genome2.fitness else (genome2,genome1)
        moreFitNodes = [(n, 0) for n in moreFitParent.nodeGenes if n.enabled == True]
        lessFitNodes = [(n, 1) for n in lessFitParent.nodeGenes if n.enabled == True]
        moreFitConns = [(n, 0) for n in moreFitParent.connectionGenes if n.enabled == True]
        lessFitConns = [(n, 1) for n in lessFitParent.connectionGenes if n.enabled == True]
        totalNodes = moreFitNodes + lessFitNodes
        totalConns = moreFitConns + lessFitConns
        totalNodes.sort(key= lambda x: x[0].nodeNum)
        totalConns.sort(key= lambda x: x[0].innovNum)
        outNodes = []
        outConns = []
        wtDiff = 0
        biasDiff = 0
        numSameNodes = 0
        numSameConns = 0
        numDiffNodes = 0
        numDiffConns = 0
        maxWDiff = float('-inf')
        minWDiff = float('inf')
        numWts = 0
        for _,g in groupby(totalNodes, key = lambda x:x[0].nodeNum):
            outNodes.append(list(g))
        for _,g in groupby(totalConns, key = lambda x:x[0].innovNum):
            outConns.append(list(g))
        for subGroup in outNodes:
            if len(subGroup) == 2:
                # morefit first
                bDiff = abs(subGroup[0][0].bias - subGroup[1][0].bias)
                wtDiff += bDiff
                maxWDiff = max(maxWDiff, bDiff)
                minWDiff = min(minWDiff, bDiff)
                numSameNodes +=1
                numWts += 1
            else:
                numDiffNodes +=1
        for subGroup in outConns:
            if len(subGroup) == 2:
                cDiff = abs(subGroup[0][0].weight - subGroup[1][0].weight)
                wtDiff += cDiff
                maxWDiff = max(maxWDiff,wtDiff)
                minWDiff = min(minWDiff, wtDiff)
                numWts += 1
                numSameConns += 1
            else:
                numDiffConns += 1
        nodeCoeff = (c1 * (numSameNodes)/(numSameNodes+numDiffNodes))
        connCoeff = (c2 * (numSameConns)/(numSameConns+numDiffConns))
        if (maxWDiff - minWDiff) != 0 and numWts != 0:
            #wtCoeff = (c3 * (1/(maxWDiff - minWDiff)*((wtDiff/numWts) - minWDiff)))
            wtCoeff = abs(c3 * wtDiff/numWts)
        else:
            wtCoeff = 1
        return nodeCoeff + connCoeff + wtCoeff

    def cloneGenome(self,genome1):
        ngCop = []
        cgCop = []
        for ng in genome1.nodeGenes:
            ngCop.append(node(ng.nodeNum,ng.depth,ng.bias, ng.nodeType))
        refDict = dict([(j.nodeNum,i) for i,j in enumerate(ngCop)])
        for cg in genome1.connectionGenes:
            newConn = connection(cg.innovNum, ngCop[refDict[cg.inputNode.nodeNum]],ngCop[refDict[cg.outputNode.nodeNum]], cg.weight)
            ngCop[refDict[cg.inputNode.nodeNum]].outputConnections.append(newConn)
            ngCop[refDict[cg.outputNode.nodeNum]].inputConnections.append(newConn)
            cgCop.append(newConn)
        o = genome(self.get_genomeID(), self.inputs,self.outputs, nodeGenes = ngCop, connectionGenes=cgCop)
        return o
    def mergeNEAT(self,genome1,genome2,pctWorseNodesToAdd):
        if genome1 is genome2:
            return self.cloneGenome(genome1)
        # find more fit parent
        (moreFitParent,lessFitParent) = (genome1,genome2) if genome1.fitness > genome2.fitness else (genome2,genome1)
        # get all nodes and connections in both parents, associate in a tuple with whether or not it is the more fit genome
        moreFitNodes = [(n, 0) for n in moreFitParent.nodeGenes if n.enabled == True]
        lessFitNodes = [(n, 1) for n in lessFitParent.nodeGenes if n.enabled == True]
        moreFitConns = [(n, 0) for n in moreFitParent.connectionGenes if n.enabled == True]
        lessFitConns = [(n, 1) for n in lessFitParent.connectionGenes if n.enabled == True]
        totalNodes = moreFitNodes + lessFitNodes
        totalConns = moreFitConns + lessFitConns
        # sort by node number and innovation number
        totalNodes.sort(key= lambda x: x[0].nodeNum)
        totalConns.sort(key= lambda x: x[0].innovNum)
        outNodes = []
        outConns = []
        newNodeGenes = []
        newConnGenes = []
        nodeNums = []
        # associate shared genes
        for _,g in groupby(totalNodes, key = lambda x:x[0].nodeNum):
            outNodes.append(list(g))
        for _,g in groupby(totalConns, key = lambda x:x[0].innovNum):
            outConns.append(list(g))
        # perform merge operation on nodes
        for subGroup in outNodes:
            if len(subGroup) == 2:
                # ndoe in both
                if subGroup[0][1] == 0:
                    if rand.random() > .1:
                        newWt = subGroup[0][0].bias
                    else:
                        newWt = subGroup[1][0].bias
                else:
                    if rand.random() > .1:
                        newWt = subGroup[1][0].bias
                    else:
                        newWt = subGroup[0][0].bias
                newNode = node(subGroup[1][0].nodeNum, depth=subGroup[1][0].depth,bias=newWt,nodeType = subGroup[1][0].nodeType)
                newNodeGenes.append(newNode)
                nodeNums.append(newNode.nodeNum)
            else:
                if subGroup[0][1] == 0:
                    newWt = subGroup[0][0].bias
                    newNode = node(subGroup[0][0].nodeNum, depth=subGroup[0][0].depth,bias=newWt,nodeType = subGroup[0][0].nodeType)
                    newNodeGenes.append(newNode)
                    nodeNums.append(newNode.nodeNum)
                elif rand.random() < pctWorseNodesToAdd:
                    newWt = subGroup[0][0].bias
                    newNode = node(subGroup[0][0].nodeNum, depth=subGroup[0][0].depth,bias=newWt,nodeType = subGroup[0][0].nodeType)
                    newNodeGenes.append(newNode)
                    nodeNums.append(newNode.nodeNum)
        refDict = dict([(j,i) for i,j in enumerate(nodeNums)])
        # perform merge operation on connections
        for subGroup in outConns:
            if len(subGroup) == 2:
                # connection in both
                if subGroup[0][1] == 0:
                    if rand.random() > .1:
                        newWt = subGroup[0][0].weight
                    else:
                        newWt = subGroup[1][0].weight
                else:
                    if rand.random() > .1:
                        newWt = subGroup[1][0].weight
                    else:
                        newWt = subGroup[0][0].weight
                inputNode = newNodeGenes[refDict[subGroup[0][0].inputNode.nodeNum]]
                outputNode = newNodeGenes[refDict[subGroup[0][0].outputNode.nodeNum]]
                newConn = connection(subGroup[0][0].innovNum,inputNode,outputNode,weight=newWt)
                inputNode.outputConnections.append(newConn)
                outputNode.inputConnections.append(newConn)
                newConnGenes.append(newConn)
            elif subGroup[0][1] == 0:
                # connection in better genome
                inputNode = newNodeGenes[refDict[subGroup[0][0].inputNode.nodeNum]]
                outputNode = newNodeGenes[refDict[subGroup[0][0].outputNode.nodeNum]]
                newConn = connection(subGroup[0][0].innovNum,inputNode,outputNode,weight=subGroup[0][0].weight)
                inputNode.outputConnections.append(newConn)
                outputNode.inputConnections.append(newConn)
                newConnGenes.append(newConn)
            else:
                #connection in worse genome
                # check if node was selected
                if subGroup[0][0].inputNode.nodeNum in nodeNums and subGroup[0][0].outputNode.nodeNum in nodeNums:
                    inputNode = newNodeGenes[refDict[subGroup[0][0].inputNode.nodeNum]]
                    outputNode = newNodeGenes[refDict[subGroup[0][0].outputNode.nodeNum]]
                    newConn = connection(subGroup[0][0].innovNum,inputNode,outputNode,weight=subGroup[0][0].weight)
                    inputNode.outputConnections.append(newConn)
                    outputNode.inputConnections.append(newConn)
                    newConnGenes.append(newConn)
        # find dead nodes
        # forward pass
        #genome(self.get_genomeID(),self.inputs,self.outputs, nodeGenes = newNodeGenes,connectionGenes = newConnGenes).printTopology()
        # find nodes with no input or output connections, recursively cycle through connected nodes and delete connections
        for ng in newNodeGenes:
            self.recursiveDelete(ng, newConnGenes)
        newNodeGenes = [n for n in newNodeGenes if (len(n.outputConnections) > 0 or n.nodeType == nodeTypes.OUTPUT) and (len(n.inputConnections) > 0 or n.nodeType == nodeTypes.INPUT)]
        newGenome = genome(self.get_genomeID(),self.inputs,self.outputs, nodeGenes = newNodeGenes,connectionGenes = newConnGenes)
        # fix connections
        newGenome.printTopology()
        return newGenome
    def mergeMethod2(self,genome1,genome2,pctWorseNodesToAdd):
        if genome1 is genome2:
            return self.cloneGenome(genome1)
        # find more fit parent
        (moreFitParent,lessFitParent) = (genome1,genome2) if genome1.fitness > genome2.fitness else (genome2,genome1)
        # get all nodes and connections in both parents, associate in a tuple with whether or not it is the more fit genome
        moreFitNodes = [(n, 0) for n in moreFitParent.nodeGenes if n.enabled == True]
        lessFitNodes = [(n, 1) for n in lessFitParent.nodeGenes if n.enabled == True]
        moreFitConns = [(n, 0) for n in moreFitParent.connectionGenes if n.enabled == True]
        lessFitConns = [(n, 1) for n in lessFitParent.connectionGenes if n.enabled == True]
        totalNodes = moreFitNodes + lessFitNodes
        totalConns = moreFitConns + lessFitConns
        # sort by node number and innovation number
        totalNodes.sort(key= lambda x: x[0].nodeNum)
        totalConns.sort(key= lambda x: x[0].innovNum)
        outNodes = []
        outConns = []
        newNodeGenes = []
        newConnGenes = []
        nodeNums = []
        # associate shared genes
        r = lambda : rand.uniform(-.5,1.5)
        for _,g in groupby(totalNodes, key = lambda x:x[0].nodeNum):
            outNodes.append(list(g))
        for _,g in groupby(totalConns, key = lambda x:x[0].innovNum):
            outConns.append(list(g))
        # perform merge operation on nodes
        for subGroup in outNodes:
            if len(subGroup) == 2:
                # ndoe in both
                if subGroup[0][1] == 0:
                    newWt = r()*(subGroup[1][0].bias-subGroup[0][0].bias) + subGroup[0][0].bias
                else:
                    newWt = r()*(subGroup[0][0].bias-subGroup[1][0].bias) + subGroup[1][0].bias
                newNode = node(subGroup[1][0].nodeNum, depth=subGroup[1][0].depth,bias=newWt,nodeType = subGroup[1][0].nodeType)
                newNodeGenes.append(newNode)
                nodeNums.append(newNode.nodeNum)
            else:
                if subGroup[0][1] == 0:
                    newWt = subGroup[0][0].bias
                    newNode = node(subGroup[0][0].nodeNum, depth=subGroup[0][0].depth,bias=newWt,nodeType = subGroup[0][0].nodeType)
                    newNodeGenes.append(newNode)
                    nodeNums.append(newNode.nodeNum)
                elif rand.random() < pctWorseNodesToAdd:
                    newWt = subGroup[0][0].bias
                    newNode = node(subGroup[0][0].nodeNum, depth=subGroup[0][0].depth,bias=newWt,nodeType = subGroup[0][0].nodeType)
                    newNodeGenes.append(newNode)
                    nodeNums.append(newNode.nodeNum)
        refDict = dict([(j,i) for i,j in enumerate(nodeNums)])
        # perform merge operation on connections
        for subGroup in outConns:
            if len(subGroup) == 2:
                # connection in both
                if subGroup[0][1] == 0:
                    newWt = r()*(subGroup[1][0].weight-subGroup[0][0].weight) + subGroup[0][0].weight
                else:
                    newWt = r()*(subGroup[0][0].weight-subGroup[1][0].weight) + subGroup[1][0].weight
                try:
                    inputNode = newNodeGenes[refDict[subGroup[0][0].inputNode.nodeNum]]
                except:
                    pdb.set_trace()
                outputNode = newNodeGenes[refDict[subGroup[0][0].outputNode.nodeNum]]
                newConn = connection(subGroup[0][0].innovNum,inputNode,outputNode,weight=newWt)
                inputNode.outputConnections.append(newConn)
                outputNode.inputConnections.append(newConn)
                newConnGenes.append(newConn)
            elif subGroup[0][1] == 0:
                # connection in better genome
                inputNode = newNodeGenes[refDict[subGroup[0][0].inputNode.nodeNum]]
                outputNode = newNodeGenes[refDict[subGroup[0][0].outputNode.nodeNum]]
                newConn = connection(subGroup[0][0].innovNum,inputNode,outputNode,weight=subGroup[0][0].weight)
                inputNode.outputConnections.append(newConn)
                outputNode.inputConnections.append(newConn)
                newConnGenes.append(newConn)
            else:
                #connection in worse genome
                # check if node was selected
                if subGroup[0][0].inputNode.nodeNum in nodeNums and subGroup[0][0].outputNode.nodeNum in nodeNums:
                    inputNode = newNodeGenes[refDict[subGroup[0][0].inputNode.nodeNum]]
                    outputNode = newNodeGenes[refDict[subGroup[0][0].outputNode.nodeNum]]
                    newConn = connection(subGroup[0][0].innovNum,inputNode,outputNode,weight=subGroup[0][0].weight)
                    inputNode.outputConnections.append(newConn)
                    outputNode.inputConnections.append(newConn)
                    newConnGenes.append(newConn)
        # find dead nodes
        # forward pass
        #genome(self.get_genomeID(),self.inputs,self.outputs, nodeGenes = newNodeGenes,connectionGenes = newConnGenes).printTopology()
        # find nodes with no input or output connections, recursively cycle through connected nodes and delete connections
        for ng in newNodeGenes:
            self.recursiveDelete(ng, newConnGenes)
        newNodeGenes = [n for n in newNodeGenes if (len(n.outputConnections) > 0 or n.nodeType == nodeTypes.OUTPUT) and (len(n.inputConnections) > 0 or n.nodeType == nodeTypes.INPUT)]
        newGenome = genome(self.get_genomeID(),self.inputs,self.outputs, nodeGenes = newNodeGenes,connectionGenes = newConnGenes)
        # fix connections
        newGenome.printTopology()
        return newGenome

    def parallelMerge(self,inputTuple):
        genome1 = inputTuple[0]
        genome2 = inputTuple[1]
        pctWorseNodesToAdd = inputTuple[2]
        if genome1 is genome2:
            return self.cloneGenome(genome1)
        # find more fit parent
        (moreFitParent,lessFitParent) = (genome1,genome2) if genome1.fitness > genome2.fitness else (genome2,genome1)
        # get all nodes and connections in both parents, associate in a tuple with whether or not it is the more fit genome
        moreFitNodes = [(n, 0) for n in moreFitParent.nodeGenes if n.enabled == True]
        lessFitNodes = [(n, 1) for n in lessFitParent.nodeGenes if n.enabled == True]
        moreFitConns = [(n, 0) for n in moreFitParent.connectionGenes if n.enabled == True]
        lessFitConns = [(n, 1) for n in lessFitParent.connectionGenes if n.enabled == True]
        totalNodes = moreFitNodes + lessFitNodes
        totalConns = moreFitConns + lessFitConns
        # sort by node number and innovation number
        totalNodes.sort(key= lambda x: x[0].nodeNum)
        totalConns.sort(key= lambda x: x[0].innovNum)
        outNodes = []
        outConns = []
        newNodeGenes = []
        newConnGenes = []
        nodeNums = []
        # associate shared genes
        r = lambda : rand.uniform(-.5,1.5)
        for _,g in groupby(totalNodes, key = lambda x:x[0].nodeNum):
            outNodes.append(list(g))
        for _,g in groupby(totalConns, key = lambda x:x[0].innovNum):
            outConns.append(list(g))
        # perform merge operation on nodes
        for subGroup in outNodes:
            if len(subGroup) == 2:
                # ndoe in both
                if subGroup[0][1] == 0:
                    newWt = r()*(subGroup[1][0].bias-subGroup[0][0].bias) + subGroup[0][0].bias
                else:
                    newWt = r()*(subGroup[0][0].bias-subGroup[1][0].bias) + subGroup[1][0].bias
                newNode = node(subGroup[1][0].nodeNum, depth=subGroup[1][0].depth,bias=newWt,nodeType = subGroup[1][0].nodeType)
                newNodeGenes.append(newNode)
                nodeNums.append(newNode.nodeNum)
            else:
                if subGroup[0][1] == 0:
                    newWt = subGroup[0][0].bias
                    newNode = node(subGroup[0][0].nodeNum, depth=subGroup[0][0].depth,bias=newWt,nodeType = subGroup[0][0].nodeType)
                    newNodeGenes.append(newNode)
                    nodeNums.append(newNode.nodeNum)
                elif rand.random() < pctWorseNodesToAdd:
                    newWt = subGroup[0][0].bias
                    newNode = node(subGroup[0][0].nodeNum, depth=subGroup[0][0].depth,bias=newWt,nodeType = subGroup[0][0].nodeType)
                    newNodeGenes.append(newNode)
                    nodeNums.append(newNode.nodeNum)
        refDict = dict([(j,i) for i,j in enumerate(nodeNums)])
        # perform merge operation on connections
        for subGroup in outConns:
            if len(subGroup) == 2:
                # connection in both
                if subGroup[0][1] == 0:
                    newWt = r()*(subGroup[1][0].weight-subGroup[0][0].weight) + subGroup[0][0].weight
                else:
                    newWt = r()*(subGroup[0][0].weight-subGroup[1][0].weight) + subGroup[1][0].weight
                try:
                    inputNode = newNodeGenes[refDict[subGroup[0][0].inputNode.nodeNum]]
                except:
                    pdb.set_trace()
                outputNode = newNodeGenes[refDict[subGroup[0][0].outputNode.nodeNum]]
                newConn = connection(subGroup[0][0].innovNum,inputNode,outputNode,weight=newWt)
                inputNode.outputConnections.append(newConn)
                outputNode.inputConnections.append(newConn)
                newConnGenes.append(newConn)
            elif subGroup[0][1] == 0:
                # connection in better genome
                inputNode = newNodeGenes[refDict[subGroup[0][0].inputNode.nodeNum]]
                outputNode = newNodeGenes[refDict[subGroup[0][0].outputNode.nodeNum]]
                newConn = connection(subGroup[0][0].innovNum,inputNode,outputNode,weight=subGroup[0][0].weight)
                inputNode.outputConnections.append(newConn)
                outputNode.inputConnections.append(newConn)
                newConnGenes.append(newConn)
            else:
                #connection in worse genome
                # check if node was selected
                if subGroup[0][0].inputNode.nodeNum in nodeNums and subGroup[0][0].outputNode.nodeNum in nodeNums:
                    inputNode = newNodeGenes[refDict[subGroup[0][0].inputNode.nodeNum]]
                    outputNode = newNodeGenes[refDict[subGroup[0][0].outputNode.nodeNum]]
                    newConn = connection(subGroup[0][0].innovNum,inputNode,outputNode,weight=subGroup[0][0].weight)
                    inputNode.outputConnections.append(newConn)
                    outputNode.inputConnections.append(newConn)
                    newConnGenes.append(newConn)
        # find dead nodes
        # forward pass
        #genome(self.get_genomeID(),self.inputs,self.outputs, nodeGenes = newNodeGenes,connectionGenes = newConnGenes).printTopology()
        # find nodes with no input or output connections, recursively cycle through connected nodes and delete connections
        for ng in newNodeGenes:
            self.recursiveDelete(ng, newConnGenes)
        newNodeGenes = [n for n in newNodeGenes if (len(n.outputConnections) > 0 or n.nodeType == nodeTypes.OUTPUT) and (len(n.inputConnections) > 0 or n.nodeType == nodeTypes.INPUT)]
        newGenome = genome(self.get_genomeID(),self.inputs,self.outputs, nodeGenes = newNodeGenes,connectionGenes = newConnGenes)
        # fix connections
        newGenome.printTopology()
        return newGenome


    def fixGenome(self,genomeIn):
        [n.disable() for n in genomeIn.connectionGenes if n.inputNode.enabled == False or n.outputNode.enabled == False]

    def recursiveDelete(self,nodeIn,connections):
        if len(nodeIn.inputConnections) == 0 and nodeIn.nodeType is nodeTypes.HIDDEN:
            for conn in nodeIn.outputConnections:
                try:
                    connections.remove(conn)
                    conn.outputNode.inputConnections.remove(conn)
                    self.recursiveDelete(conn.outputNode, connections)
                except:
                    pass
        if len(nodeIn.outputConnections) == 0 and nodeIn.nodeType is nodeTypes.HIDDEN:
            for conn in nodeIn.inputConnections:
                try:
                    connections.remove(conn)
                    conn.inputNode.outputConnections.remove(conn)
                    self.recursiveDelete(conn.inputNode,connections)
                except:
                    pass

    def reverseRecursiveDelete(self):
        for conn in nodeIn.outputConnections:
            connections.remove(conn)
            conn.inputNode.inputConnections.remove(conn)
            if len(conn.inputNode.inputConnections) == 0 and conn.inputNode.nodeType is not nodeTypes.OUTPUT:
                self.recursiveDelete(conn.inputNode,connections)

    def mutationStressTest(self, initPop,maxGens, epochs,learningRate, encodingRate):
        # create instances and mutate
        instances = []
        for _ in range(0,initPop):
            instances.append(self.newInitGenome())
        for currGeneration in range(0,maxGens):
            for ind,genin in enumerate(instances):
                self.randomMutationSpecial(genin, 1)
                ver = self.verifyMutation(genin)
                genin.transcodeNetwork()
                genin.train(self.trainingData,self.trainingLabels, epochs, learningRate)
            newInstances = []
            while len(newInstances) < len(instances):
                choices = rand.sample(instances, 2)
                newInstances.append(self.mergeMethod2(choices[0],choices[1],encodingRate))
            print("Current Generation %i "%(currGeneration))

    def verifyMutation(self,genomeIn):
        # for a genome, all connections must have a corresponding node in the genome
        print("verifying genome %i"%(genomeIn.ID))
        verFailed = False
        genomeNodes = [n for n in genomeIn.nodeGenes]
        inputNodes = [n.inputNode for n in genomeIn.connectionGenes]
        outputNodes = [n.outputNode for n in genomeIn.connectionGenes]
        diffNodes = set(genomeNodes).symmetric_difference(set(inputNodes + outputNodes))
        if len(diffNodes) > 0:
            print("verification failed, node found in connection genes that is not in node genes\n Culprit Nodes:" + str(list(diffNodes)))
            verFailed = True
        # all input and output connections in a node must be in the connections
        connsInNodes = [n for sub in genomeNodes for n in sub.outputConnections] + [n for sub in genomeNodes for n in sub.inputConnections]
        diffConns = set(connsInNodes).symmetric_difference(set(genomeIn.connectionGenes))
        if len(diffConns) > 0:
            print("verifcation failed, connection found in node genes that is not in the connection genes \n Culprtit connections: " + str(list(diffConns)))
            verFailed = True
        return verFailed
    ## evolution ##
    def speciate(self, genomeList,cutoff,c1,c2,c3, speciesStartList = []):
        if speciesStartList is []:
            speciesList = [[genomeList[0]]]
            del genomeList[0]
        else:
            speciesList = speciesStartList
        for currGenome in genomeList:
            suitableSpeciesFound = False
            for ind,species in enumerate(speciesList):
                for gen in species:
                    compat = self.geneCompatMethod2(currGenome,gen,c1,c2,c3)
                    #print(compat)
                    if compat > cutoff:
                        suitableSpeciesFound = True
                        species.append(currGenome)
                        break
            if not suitableSpeciesFound:
                speciesList.append([])
                speciesList[-1].append(currGenome)
        return speciesList
def speciateNEAT(self, genomeList,cutoff,c1,c2,c3, speciesStartList = []):
        if speciesStartList is []:
            speciesList = [[genomeList[0]]]
            del genomeList[0]
        else:
            speciesList = speciesStartList
        for currGenome in genomeList:
            suitableSpeciesFound = False
            for ind,species in enumerate(speciesList):
                for gen in species:
                    compat = self.geneCompatMethod2(currGenome,gen,c1,c2,c3)
                    #print(compat)
                    if compat > cutoff:
                        suitableSpeciesFound = True
                        species.append(currGenome)
                        break
            if not suitableSpeciesFound:
                speciesList.append([])
                speciesList[-1].append(currGenome)
        return speciesList

    def evolveSpeciated(self,pop,c1,c2,c3,cutoff,inputData,outputData,maxGenerations,epochs,learningRate):
        instances = []
        speciesStartList = []
        fitnessArr = np.zeros((pop))
        for _ in range(0,pop):
            instances.append(self.newInitGenome())
        for gen in range(0,maxGenerations):
            print("There are %i total genomes"%(len(instances)))
            for inst in instances:
                self.randomM(inst)
            speciesList = self.speciate(instances, cutoff,c1,c2,c3, speciesStartList = speciesStartList)
            speciesStartList = []
            print("there are %i total species"%(len(speciesList)))
            for species in speciesList:
                (instances,best) = self.evaluateSpecies(species,.8,inputData,outputData,epochs,learningRate)
                print(np.mean([n.fitness for n in species]))
                speciesStartList.append([best])
            print("current generation: %i"%(gen))
        best.printTopology()

    def evaluateSpecies(self,speciesGenomes,proportionToElim,inputData,outputData,epochs,learningRate):
        numGenomes = len(speciesGenomes)
        outputGenomes = []
        fitnessArr = np.zeros(numGenomes)
        for ind,gen in enumerate(speciesGenomes):
            self.fixGenome(gen)
            gen.transcodeNetwork()
            gen.train(inputData,outputData,epochs,learningRate)
            gen.getFitness(inputData,outputData)
        fitnessArr = np.array([n.fitness for n in speciesGenomes])/numGenomes
        numToRemove = math.floor(proportionToElim * numGenomes)
        genToRemove = fitnessArr.argsort()[:numToRemove][::-1]
        successfulGenomes = [i for j, i in enumerate(speciesGenomes) if j not in genToRemove]
        numToReplace = numGenomes - numToRemove
        bestFitness = max(fitnessArr.tolist())
        bestGenome = speciesGenomes[fitnessArr.tolist().index(bestFitness)]
        while len(outputGenomes) < numGenomes:
            choice1 = rand.choice(successfulGenomes)
            choice2 = rand.choice(successfulGenomes)
            newGenome = self.merge(choice1,choice2,1,1)
            if newGenome is not None:
                print("combined genome")
                outputGenomes.append(newGenome)
            else:
                print("merge failed")
        return (outputGenomes, bestGenome)

    def findDisconnectedGenome(self,genomeIn):
        for n in genomeIn.nodeGenes:
            for c in n.outputConnections:
                if c not in genomeIn.connectionGenes:
                    print(c)
                    return True

    def evolveFeedback(self, initPop, c1,c2,c3,c0, maxGens, e0, l0,speciationTarget,dSpec,topNrat,topNSpecRat):
        # experimental idea
        # create instances and mutate
        instances = []
        epochMults = []
        speciesStartList = []
        cutoff = c0
        mutationMult = np.ones(initPop)
        specTarget = math.floor((initPop) * speciationTarget)
        meanFitnessArr = []
        maxFitnessArr = []
        for _ in range(0,initPop):
            instances.append(self.newInitGenome())
        for currGeneration in range(0,maxGens):
            for ind,genin in enumerate(instances):
                genin.epochMult = self.randomMutationSpecial(genin, mutationMult[ind])
            speciesList = self.speciate(instances, cutoff, c1,c2,c3, speciesStartList = speciesStartList)
            cutoff -= (dSpec*(len(speciesList) - specTarget))
            print(cutoff)
            #print(epochMults)
            print("current generation : %i, Number of instances : %i, number of species: %i "%(currGeneration, len(instances),len(speciesList)))
            for ind,species in enumerate(speciesList):
                print("evaluating species %i"%(ind))
                self.evaluateSpeciesFeedback(species,e0,l0)
                self.interSpeciesSelectionFeedback(species, topNrat)
            (specFitnessArr,speciesList) = self.intraSpeciesSelectionFeedback(speciesList, topNSpecRat)
            #(instances,speciesStartList,mutationMult) = self.repopulateFeedback(speciesList,specFitnessArr,initPop)
            (instances,speciesStartList,mutationMult) = self.repopulateFeedback(speciesList,specFitnessArr,initPop)
            print(specFitnessArr)
            maxFitnessArr.append(np.max(specFitnessArr))
            meanFitnessArr.append(np.mean(specFitnessArr))
        self.bestGenome = max(instances + [n for sub in speciesList for n in sub],key= lambda x:x.fitness)
        return (maxFitnessArr, meanFitnessArr)
        #best.printTopology()
    def evolveNEAT(self, initPop, c1,c2,c3,c0,maxGens,topNrat, topNSpecRat):
        instances = []
        epochMults = []
        speciesStartList = []
        cutoff = c0
        meanFitnessArr = []
        maxFitnessArr = []
        mutationMult = np.ones(initPop)
        solFound = False
        solGen = -1
        for _ in range(0,initPop):
            instances.append(self.newInitGenome())
        for currGeneration in range(0,maxGens):
            for ind,genin in enumerate(instances):
                self.randomMutationNEAT(genin,mutationMult[ind])
            speciesList = self.speciate(instances, cutoff, c1,c2,c3, speciesStartList = speciesStartList)
            for ind,species in enumerate(speciesList):
                print("evaluating species %i"%(ind))
                solOut = self.evaluateSpeciesNEAT(species)
                if solOut and not solFound:
                    solGen = currGeneration
                    solFound = True
                self.interSpeciesSelectionFeedback(species, topNrat)
            (specFitnessArr,speciesList) = self.intraSpeciesSelectionFeedback(speciesList, topNSpecRat)
            #(instances,speciesStartList,mutationMult) = self.repopulateFeedback(speciesList,specFitnessArr,initPop)
            (instances,speciesStartList,mutationMult) = self.repopulateFeedback(speciesList,specFitnessArr,initPop)
            #print(cutoff)
            #print(cutoff)
            print(specFitnessArr)
            maxFitnessArr.append(np.max(specFitnessArr))
            meanFitnessArr.append(np.mean(specFitnessArr))
            #print(cutoff)
        self.bestGenome = max(instances + [n for sub in speciesList for n in sub],key= lambda x:x.fitness)
        return (maxFitnessArr, meanFitnessArr,solGen)
    def evolveFeedbackParallel(self, initPop, c1,c2,c3,c0, maxGens, e0, l0,speciationTarget,dSpec,topNrat,topNSpecRat):
        # experimental idea
        # create instances and mutate
        instances = []
        epochMults = []
        speciesStartList = []
        cutoff = c0
        mutationMult = np.ones(initPop)
        specTarget = math.floor((initPop) * speciationTarget)
        p = Pool()
        solFound = False
        solGen = -1
        numNodes = -1
        numConns = -1
        meanFitnessArr = []
        maxFitnessArr = []
        for _ in range(0,initPop):
            instances.append(self.newInitGenome())
        for currGeneration in range(0,maxGens):
            for ind,genin in enumerate(instances):
                genin.epochMult = self.randomMutationSpecial(genin, mutationMult[ind])
            speciesList = self.speciate(instances, cutoff, c1,c2,c3, speciesStartList = speciesStartList)
            cutoff -= (dSpec*(len(speciesList) - specTarget))
            print(cutoff)
            #print(epochMults)
            print("current generation : %i, Number of instances : %i, number of species: %i "%(currGeneration, len(instances),len(speciesList)))
            parallelStruct = []
            for species in speciesList:
                parallelStruct.append((species, e0, l0))
            newSpeciesList = []
            speciesList = p.map(self.evaluateSpeciesParallel, parallelStruct)
            #pdb.set_trace()
            for ind,species in enumerate(speciesList):
                print("evaluating species %i"%(ind))
                self.interSpeciesSelectionFeedback(species, topNrat)
                for gen in species:
                    g = gen.net.evalCorrect(self.trainingData,self.trainingLabels)
                    if g and not solFound:
                        solFound = True
                        solGen = currGeneration
                        numNodes = len([n for n in gen.nodeGenes if n.enabled == True])
                        numConns = len([n for n in gen.connectionGenes if n.enabled == True])
            (specFitnessArr,speciesList) = self.intraSpeciesSelectionFeedback(speciesList, topNSpecRat)
            #(instances,speciesStartList,mutationMult) = self.repopulateFeedback(speciesList,specFitnessArr,initPop)
            (instances,speciesStartList,mutationMult) = self.repopulateFeedback(speciesList,specFitnessArr,initPop)
            #print(cutoff)
            #print(cutoff)
            maxFitnessArr.append(np.max(specFitnessArr))
            meanFitnessArr.append(np.mean(specFitnessArr))
            print(specFitnessArr)
            #print(cutoff)
        #p.close()
        #p.join()
        self.bestGenome = max(instances + [n for sub in speciesList for n in sub],key= lambda x:x.fitness)
        #best.printTopology()
        return (maxFitnessArr, meanFitnessArr,solGen,numNodes,numConns)
    def evolveNoFeedbackParallel(self, initPop, c1,c2,c3,c0, maxGens, e0, l0,speciationTarget,dSpec,topNrat,topNSpecRat):
        # experimental idea
        # create instances and mutate
        instances = []
        epochMults = []
        speciesStartList = []
        cutoff = c0
        mutationMult = np.ones(initPop)
        specTarget = math.floor((initPop) * speciationTarget)
        p = Pool()
        solFound = False
        solGen = -1
        meanFitnessArr = []
        maxFitnessArr = []
        for _ in range(0,initPop):
            instances.append(self.newInitGenome())
        for currGeneration in range(0,maxGens):
            for ind,genin in enumerate(instances):
                genin.epochMult = self.randomMutationSpecial(genin, mutationMult[ind])
            speciesList = self.speciate(instances, cutoff, c1,c2,c3, speciesStartList = speciesStartList)
            print("current generation : %i, Number of instances : %i, number of species: %i "%(currGeneration, len(instances),len(speciesList)))
            parallelStruct = []
            for species in speciesList:
                parallelStruct.append((species, e0, l0))
            newSpeciesList = []
            speciesList = p.map(self.evaluateSpeciesParallel, parallelStruct)
            #pdb.set_trace()
            for ind,species in enumerate(speciesList):
                print("evaluating species %i"%(ind))
                self.interSpeciesSelectionFeedback(species, topNrat)
                for gen in species:
                    g = gen.net.evalCorrect(self.trainingData,self.trainingLabels)
                    if g and not solFound:
                        solFound = True
                        solGen = currGeneration
            (specFitnessArr,speciesList) = self.intraSpeciesSelectionFeedback(speciesList, topNSpecRat)
            #(instances,speciesStartList,mutationMult) = self.repopulateFeedback(speciesList,specFitnessArr,initPop)
            (instances,speciesStartList,_) = self.repopulateFeedback(speciesList,specFitnessArr,initPop)
            #print(cutoff)
            #print(cutoff)
            maxFitnessArr.append(np.max(specFitnessArr))
            meanFitnessArr.append(np.mean(specFitnessArr))
            print(specFitnessArr)
            #print(cutoff)
        p.close()
        p.join()
        self.bestGenome = max(instances + [n for sub in speciesList for n in sub],key= lambda x:x.fitness)
        #best.printTopology()
        return (maxFitnessArr, meanFitnessArr,solGen)
    def evolveFeedbackParallel2(self, initPop, c1,c2,c3,c0, maxGens, e0, l0,speciationTarget,dSpec,topNrat,topNSpecRat):
        # experimental idea
        # create instances and mutate
        instances = []
        epochMults = []
        speciesStartList = []
        cutoff = c0
        mutationMult = np.ones(initPop)
        specTarget = math.floor((initPop) * speciationTarget)
        for _ in range(0,initPop):
            instances.append(self.newInitGenome())
        for currGeneration in range(0,maxGens):
            for ind,genin in enumerate(instances):
                genin.epochMult = self.randomMutationSpecial(genin, mutationMult[ind])
            speciesList = self.speciate(instances, cutoff, c1,c2,c3, speciesStartList = speciesStartList)
            cutoff -= (dSpec*(len(speciesList) - specTarget))
            print(cutoff)
            print("current generation : %i, Number of instances : %i, number of species: %i "%(currGeneration, len(instances),len(speciesList)))
            for ind,species in enumerate(speciesList):
                print("evaluating species %i"%(ind))
                self.evaluateSpeciesFeedbackParallel(species, e0, l0)
                self.interSpeciesSelectionFeedback(species, topNrat)
            (specFitnessArr,speciesList) = self.intraSpeciesSelectionFeedback(speciesList, topNSpecRat)
            (instances,speciesStartList,mutationMult) = self.repopulateFeedback(speciesList,specFitnessArr,initPop)
            print(specFitnessArr)
        self.bestGenome = max(instances + [n for sub in speciesList for n in sub],key= lambda x:x.fitness)
        #best.printTopology()

    def evaluateSpeciesFeedback(self,species, e0, lr):
        elemInSpecies = len(species)
        netArr = []
        for ind,gen in enumerate(species):
            self.fixGenome(gen)
            gen.transcodeNetwork()
            gen.train(self.trainingData, self.trainingLabels,math.floor(e0*gen.epochMult),lr)
            gen.getFitness(self.testData,self.testLabels)
            gen.fitness = gen.fitness/elemInSpecies
            solReached = gen.net.evalCorrect(self.testData,self.testLabels)
        epochMults = []
        return solReached
    def evaluateSpeciesNEAT(self,species):
        elemInSpecies = len(species)
        netArr = []
        for ind,gen in enumerate(species):
            self.fixGenome(gen)
            gen.transcodeNetwork()
            #gen.train(self.trainingData, self.trainingLabels,100,.2)
            gen.getFitness(self.testData,self.testLabels)
            gen.fitness = gen.fitness/elemInSpecies
            solReached = gen.net.evalCorrect(self.testData,self.testLabels)
        return solReached
    def evaluateSpeciesFeedbackParallel(self, species, e0, lr):
        elemInSpecies = len(species)
        netArr = []
        p = Pool()
        for ind, gen in enumerate(species):
            self.fixGenome(gen)
            netArr.append((gen.transcodeNetworkParallel(), e0, gen.epochMult, lr))
        netArr = p.map(self.trainNet, netArr)
        p.close()
        p.join()
        for ind, gen in enumerate(species):
            netArr[ind].updateGenome(gen)

    def trainNet(self, tupleIn):
        netIn = tupleIn[0]
        e0 = tupleIn[1]
        epochMult = tupleIn[2]
        lr = tupleIn[3]
        netIn.train(self.trainingData, self.trainingLabels, math.floor(e0 * epochMult),lr)
        return netIn

    def evaluateSpeciesParallel(self,tupleIn):
        species = tupleIn[0]
        e0 = tupleIn[1]
        lr = tupleIn[2]
        elemInSpecies = len(species)
        for ind,gen in enumerate(species):
            self.fixGenome(gen)
            gen.transcodeNetwork()
            gen.train(self.trainingData, self.trainingLabels,math.floor(e0*gen.epochMult),lr)
            gen.getFitness(self.trainingData,self.trainingLabels)
            gen.fitness = gen.fitness/elemInSpecies
        return species

    def spawnProcessess(self):
        print("Cpu's detected: %i"%cpu_count())
        print("spawning processes...")
        for cpu in range(0,cpu_count()):
            parent_conn,child_conn = Pipe()
            P = Process()


    def interSpeciesSelectionFeedback(self,species,topNrat):
        numToSelect = max(1,math.floor(len(species)*topNrat))
        sortedSpecies = sorted(species, key=lambda x: x.fitness,reverse=True)
        species = sortedSpecies[0:numToSelect]

    def intraSpeciesSelectionFeedback(self,speciesList, topNSpecRat):
        specFitnessArr = []
        numSpecToCont = max(1,math.floor(len(speciesList)*topNSpecRat))
        for species in speciesList:
            specFitnessArr.append(np.mean([n.fitness for n in species]))
        sortedArgs = np.argsort(specFitnessArr)[::-1]
        speciesListTemp = [speciesList[n] for n in sortedArgs[0:numSpecToCont]]
        outArr = [specFitnessArr[n] for n in sortedArgs[0:numSpecToCont]]
        return (outArr,speciesListTemp)

    def repopulateFeedback(self, speciesList,specFitnessArr,targetPop):
        #normalize relative fitness to 1 and use as a probability distribution for repopulation
        probDist = specFitnessArr/np.linalg.norm(specFitnessArr)
        mutationMult = []
        refDict = dict(zip([n[0].ID for n in speciesList],probDist))
        speciesLeaders = [[max(n,key=lambda x:x.fitness)] for n in speciesList]
        outputInstances = []
        while len(outputInstances) < targetPop-len(speciesLeaders):
            speciesChoice = rand.choices(speciesList, weights=probDist.tolist())
            if len(speciesChoice[0]) == 1:
                newGenome = self.mergeMethod2(speciesChoice[0][0],speciesChoice[0][0],0)
                outputInstances.append(newGenome)
                mutationMult.append(refDict[speciesChoice[0][0].ID])
            elif len(speciesChoice[0]) > 1:
                newGenomes = rand.sample(speciesChoice[0],2)
                newGenome = self.mergeMethod2((newGenomes[0]),(newGenomes[1]),.5)
                outputInstances.append(newGenome)
                mutationMult.append(refDict[speciesChoice[0][0].ID])
        return (outputInstances, speciesLeaders,mutationMult)
    def repopulateNEAT(self, speciesList,specFitnessArr,targetPop):
        #normalize relative fitness to 1 and use as a probability distribution for repopulation
        #probDist = specFitnessArr/np.linalg.norm(specFitnessArr)
        mutationMult = []
        speciesLeaders = [[max(n,key=lambda x:x.fitness)] for n in speciesList]
        outputInstances = []
        while len(outputInstances) < targetPop-len(speciesLeaders):
            speciesChoice = rand.choice(speciesList)
            if len(speciesChoice) == 1:
                newGenome = self.mergeMethod2(speciesChoice[0],speciesChoice[0],.5)
                outputInstances.append(newGenome)
                mutationMult.append(0)
            elif len(speciesChoice) > 1:
                newGenomes = rand.sample(speciesChoice,2)
                newGenome = self.mergeNEAT((newGenomes[0]),(newGenomes[1]),0)
                outputInstances.append(newGenome)
                mutationMult.append(0)
        return (outputInstances, speciesLeaders,mutationMult)

    def repopulateParallel(self, speciesList, specFitnessArr, targetPop,threadPool):
        #normalize relative fitness to 1 and use as a probability distribution for repopulation
        probDist = specFitnessArr/np.linalg.norm(specFitnessArr)
        mutationMult = []
        refDict = dict(zip([n[0].ID for n in speciesList],probDist))
        speciesLeaders = [[max(n,key=lambda x:x.fitness)] for n in speciesList]
        outputInstances = []
        pairList = []
        while len(pairList) < targetPop-len(speciesLeaders):
            speciesChoice = rand.choices(speciesList, weights=probDist.tolist())
            if len(speciesChoice[0]) == 1:
                pairList.append((speciesChoice[0][0],speciesChoice[0][0],0))
                mutationMult.append(refDict[speciesChoice[0][0].ID])
            elif len(speciesChoice[0]) > 1:
                newGenomes = rand.sample(speciesChoice[0],2)
                pairList.append((newGenomes[0],newGenomes[1],.5))
                mutationMult.append(refDict[speciesChoice[0][0].ID])
        outputInstances = threadPool.map(self.parallelMerge, pairList)
        return (outputInstances, speciesLeaders, mutationMult)

    def getSpeciesStatistics(self,inputSpecies,sampleRate):
        # interspecies variance
        interGroupVarianceArr = []
        choiceArr = []
        intraGroupVariance = []
        for species in inputSpecies:
            choiceArr.append(rand.choice(species))
            innerVrianceArr = []
            numSamples = max(1, len(species) * sampleRate)
            if numSamples == 1:
                variance = 0
            else:
                samples = rand.sample(species,numSamples)
                for gen1 in samples:
                    for gen2 in samples:
                        if gen1 is not gen2:
                            interGroupVarianceArr.append(self.geneCompat(gen1,gen2))
                interGroupVarianceArr.append(np.var(innerVarianceArr))

    def updateFeedBackParameters(self, feedBackDict, stats):
        pass

    def evaluateTestData(self, genomeIn):
        genomeIn.transcodeNetwork()
        genomeIn.evaluate(self.testData, self.testLabels)

def tic():
    #Homemade version of matlab tic and toc functions
    import time
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()

def toc():
    import time
    if 'startTime_for_tictoc' in globals():
        print("Elapsed time is " + str(time.time() - startTime_for_tictoc) + " seconds.")
    else:
        print("Toc: start time not set")

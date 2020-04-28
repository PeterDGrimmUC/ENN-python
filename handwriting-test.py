from ENN.EXAMMV2 import *
import numpy as np

image_size = 28 # width and length
no_of_different_labels = 10 #  i.e. 0, 1, 2, 3, ..., 9
image_pixels = image_size * image_size
data_path = "/handwritingData/"
train_data = np.loadtxt(data_path + "mnist_train.csv",
                        delimiter=",")
test_data = np.loadtxt(data_path + "mnist_test.csv",
                       delimiter=",")

def digToLabel(dig):
    outp = np.zeros(10)
    outp[dig] = 1
    return outp
def dataToNetForm(data):
    inputData = []
    labelData = []
    for dpt in data:
        labelData.append(digToLabel(int(dpt[0])))
        inputData.append(dpt[1:])
    return (inputData,labelData)

model_inputs = image_size ** 2
model_outputs = 10
proc = masterProcess(model_inputs,model_outputs)
(training_inputs,training_labels) = dataToNetForm(train_data)
(test_inputs,test_labels) = dataToNetForm(test_data)
proc.set_testData(test_inputs,test_labels)
proc.set_trainingData(training_inputs,training_labels)
proc.evolveFeedbackParallel(20,.4,.4,.2,.8,30,2,.01,.3,.01,.3,.8)

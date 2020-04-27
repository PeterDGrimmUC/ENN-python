import pandas as pd
import datetime
import pandas_datareader.data as web
import numpy as np

class marketData:
    def __init__(self, ticker):
        start = datetime.datetime(1990, 1, 1)
        end = datetime.date.today()
        self.ticker = ticker
        self.df = web.DataReader([self.ticker], 'yahoo', start, end)
    def parsePriceData(self, inputs,outputs, dt_Inputs,dt_Outputs):
        # get n inputs at a distance dt from current time, parse into list
        closeData = self.df["Adj Close"][self.ticker].values
        numDays = len(closeData)
        inputData = []
        outputData = []
        for d in range(0, numDays -  (outputs * dt_Outputs) - (inputs * dt_Inputs)):
            tmpInputData = []
            tmpOutputData = []
            for i in range(0, inputs):
                tmpInputData.append(closeData[d + (i * dt_Inputs)])
            for o in range(0, outputs):
                tmpOutputData.append(closeData[d + (inputs*dt_Inputs) + (o * dt_Outputs)]/np.mean(tmpInputData))
            inputData.append(tmpInputData)
            outputData.append(tmpOutputData)
        return (inputData,outputData)


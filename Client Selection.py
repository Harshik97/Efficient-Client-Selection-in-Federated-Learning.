import numpy as np
import json
import pandas as pd
import sklearn
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder
from matplotlib.backends.backend_pdf import PdfPages



class Question1:

    def __init__(self, data=0, labels=0, data_ratio_list =[], names=[], globalRounds=0, localRounds=0, batchSize=0, learningRate=0, d=0, m=0, classes = 10):

        self.clientData = data
        self.clientLabels = labels


        self.learning_rate = learningRate
        self.d = d
        self.m = m
        self.batch_size = batchSize
        self.globalRounds = globalRounds
        self.localRounds = localRounds
        self.classes  = classes
        self.data_ratio_list = data_ratio_list

        self.clientNames = names
        self.clientWeights, self.clientBias, self.clientPredictions = {}, {}, {}
        self.totalClients = len(names)
        self.selectedDClients, self.selectedMClients = 0, 0
        self.globalWeights, self.globalBias = None, None

    def preProcess(self, filePath):

        file = open(filePath)
        data = json.load(file)
        clientNames = list(data['users'])
        totalDataPoints = 0
        clientData, clientLabels = {}, {}
        for client in data['users']:
            for dataset in data['user_data'][client]:
                if dataset == "x":
                    clientData[client] = np.array(data['user_data'][client][dataset])
                    totalDataPoints += clientData[client].shape[0]
                if dataset == "y":
                    clientLabels[client] = np.array(data['user_data'][client][dataset]).reshape(-1, 1)


        ratio_list = []
        for client in clientNames:
            ratio = clientData[client].shape[0]/totalDataPoints
            ratio_list.append(ratio)

        return clientData, clientLabels, clientNames, ratio_list

    def oneHotEncode(self, data):

        oneHotEncoding = np.array([[1 if i == lab else 0 for i in range(10)] for lab in data])
        return oneHotEncoding

    def changeLearningRate(self):

        self.learning_rate = self.learning_rate/2



    def setParams(self):

        for name in self.clientNames:

            features = len(self.clientData[name][0])
            self.clientWeights[name] = np.random.randn(features, self.classes)
            self.clientBias[name] = 0

        self.globalWeights = np.random.randn(features, self.classes)
        self.globalBias = 0

    def dataloader(self,data, labels, shuffle=True):

        data_points = data.shape[0]
        batch_size = self.batch_size
        if shuffle:
            i = np.random.permutation(data_points)
        data = data[i]
        labels = labels[i]

        final_data = np.split(data[:int(data.shape[0] / batch_size) * batch_size], int(data.shape[0] / batch_size))
        final_labels = np.split(labels[:int(labels.shape[0] / batch_size) * batch_size],
                                int(labels.shape[0] / batch_size))

        if data.shape[0] % batch_size != 0:
            final_data.append(data[int(data.shape[0] / batch_size) * batch_size:])
            final_labels.append(labels[int(labels.shape[0] / batch_size) * batch_size:])

        return zip(final_data, final_labels)

    def forward(self,data, clientIndex):

        clientName = self.clientNames[clientIndex]
        weights = self.clientWeights[clientName]
        bias = self.clientBias[clientName]

        preSoftmax = np.dot(data, weights) + bias

        a = np.amax(preSoftmax, axis=1).reshape(-1, 1)
        b = np.sum(np.exp(preSoftmax - a), axis=-1).reshape(-1, 1)
        softmax = (np.exp(preSoftmax - a)) / (b)
        softmax[softmax == 0] = 1e-16
        self.clientPredictions[clientName] = softmax



    def computeLocalLoss(self, labels, clientIndex):

        clientName = self.clientNames[clientIndex]
        predictions = self.clientPredictions[clientName]
        loss = -1 * (1 / self.batch_size) * np.sum(labels * np.log(predictions), axis=-1).sum()
        return loss

    def updateLocalParams(self, data, labels, clientIndex):

        clientName = self.clientNames[clientIndex]

        weights, bias = self.clientWeights[clientName], self.clientBias[clientName]
        predictions = self.clientPredictions[clientName]
        weights -= ((self.learning_rate*np.dot(data.T, predictions - labels))/self.batch_size)
        bias -= ((self.learning_rate*np.sum(predictions - labels, axis = 0))/self.batch_size)

        self.clientWeights[clientName], self.clientBias[clientName] = weights, bias

    def selectDClients(self):

        clients = [x for x in range(self.totalClients)]
        dClients = np.random.choice(np.array(clients), self.d, True, self.data_ratio_list)
        self.selectedDClients = dClients

        if self.globalWeights is not None and self.globalBias is not None:
            for index in self.selectedDClients:
                clientName = self.clientNames[index]
                self.clientWeights[clientName] = self.globalWeights
                self.clientBias[clientName] = self.globalBias


    def globalUpdate(self):


        globalWeights, globalBias = 0, 0

        for clientIndex in self.selectedMClients:
            clientName = self.clientNames[clientIndex]
            data_ratio = self.data_ratio_list[clientIndex]
            weights, bias = self.clientWeights[clientName], self.clientBias[clientName]
            globalWeights += weights
            globalBias += bias

        self.globalWeights = globalWeights/self.m
        self.globalBias = globalBias/self.m


    def runLocalModel(self, clientIndex, trainX, encodedtrainY, update = False):

        trainingData = self.dataloader(trainX, encodedtrainY)
        x, y = [list(x) for x in zip(*trainingData)]
        random_batch = np.random.randint(0, len(x))
        self.forward(x[random_batch], clientIndex)
        loss = self.computeLocalLoss(y[random_batch], clientIndex)
        if update is True:
            self.updateLocalParams(x[random_batch], y[random_batch], clientIndex)
        return loss

    def findDLosses(self):

        dClientsLosses = []
        for index in self.selectedDClients:
            clientName = self.clientNames[index]
            trainX = self.clientData[clientName]
            trainY = self.clientLabels[clientName]
            encodedtrainY = self.oneHotEncode(trainY)
            clientDataRatio = self.data_ratio_list[index]
            loss = self.runLocalModel(index, trainX, encodedtrainY)
            dClientsLosses.append(loss*clientDataRatio)

        return dClientsLosses

    def findMLosses(self, dClientsLosses):

        zipped_lists = zip(dClientsLosses, self.selectedDClients)
        sorted_pair = sorted(zipped_lists, reverse=True)
        tuples = zip(*sorted_pair)
        clientLoss, selectedDClients = [list(tuple) for tuple in tuples]
        self.selectedMClients = selectedDClients[0:self.m]


    def computeGlobalLoss(self):

        allClientLoss = []
        for clientIndex in range(self.totalClients):
            clientName = self.clientNames[clientIndex]
            clientDataRatio = self.data_ratio_list[clientIndex]
            trainX = self.clientData[clientName]
            trainY = self.clientLabels[clientName]
            encodedtrainY = self.oneHotEncode(trainY)
            loss = self.runLocalModel(clientIndex, trainX, encodedtrainY)
            allClientLoss.append(loss * clientDataRatio)

        loss = sum(allClientLoss)
        return loss

    def train(self):

        self.setParams()
        serverLoss = []
        for round in range(self.globalRounds):
                if round == 300 or round == 600:
                    self.changeLearningRate()

                self.selectDClients()
                dClientsLoss = self.findDLosses()
                self.findMLosses(dClientsLoss)
                loss = 0
                for clientIndex in self.selectedMClients:
                        clientName = self.clientNames[clientIndex]
                        trainX = self.clientData[clientName]
                        trainY = self.clientLabels[clientName]
                        encodedtrainY = self.oneHotEncode(trainY)


                        for local in range(self.localRounds):
                            self.runLocalModel(clientIndex, trainX, encodedtrainY, update = True)
                self.globalUpdate()
                globalLoss = self.computeGlobalLoss()
                serverLoss.append(globalLoss/self.m)

        return serverLoss

def plotStatistics(x, y, m):

    fig, ax = plt.subplots()
    ax.plot(x, y[0], '-b', label='d = m')
    ax.plot(x, y[1], '-r', label='d = 2m')
    ax.plot(x, y[2], '-g', label='d = 10m')
    plt.xlabel('Communication Rounds')
    plt.ylabel('Global Loss for m = ' +str(m))
    leg = ax.legend()
    plt.show()


def main():
    filePath = 'xyz'
    global_rounds = 800
    local_rounds = 30
    batch_size = 45
    learning_rate = 0.05
    mRange = [1, 3]
    dRange = [1, 2, 10]
    communication_rounds = [x for x in range(global_rounds)]
    test = Question1()
    data, labels, names, data_ratio_list = test.preProcess(filePath)
    for m in mRange:
        serverLoss = []
        for d in dRange:
            test = Question1(data, labels, data_ratio_list, names, global_rounds, local_rounds, batch_size, learning_rate, d*m, m)
            globalLoss = test.train()
            serverLoss.append(globalLoss)
        plotStatistics(communication_rounds, serverLoss, m)


if __name__ == '__main__':
    main()







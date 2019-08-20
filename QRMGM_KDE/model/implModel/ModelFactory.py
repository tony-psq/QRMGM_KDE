# -*- coding: utf-8 -*-
# @File         : ModelFactory.py
# @Author       : Zhendong Zhang
# @Email        : zzd_zzd@hust.edu.cn
# @University   : Huazhong University of Science and Technology
# @Date         : 2019/8/14
# @Software     : PyCharm
# -*---------------------------------------------------------*-

import abc
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, GRU
from keras.optimizers import Adam

from QRMGM_KDE.model.recurrent import MGM
from QRMGM_KDE.model.LossFunction import QuantileLoss
from QRMGM_KDE.model.Generator import TrainingGenerator, ValidationGenerator, LRDecay


class QuantileRegressionRecurrentModel(object):

    # constructor
    def __init__(self, trainX, trainY, validationX, validationY, hyperParameters=None):
        self.trainX = trainX
        self.trainY = trainY
        self.validationX = validationX
        self.validationY = validationY
        self.hyperParameters = hyperParameters
        self.depth = hyperParameters['depth']
        self.nodeNum = hyperParameters['nodeNum']
        self.quantiles = hyperParameters['quantiles']
        self.activation = hyperParameters['activation']
        self.dropout = hyperParameters['dropout']
        self.recurrentModel = None

    # construct model based on hyperParameters
    @abc.abstractmethod
    def constructModel(self, **kwargs):
        pass

    # train model
    def fit(self, **kwargs):
        # Handle parameters
        # bs:  batch size
        # ce:  convergence epochs
        # ilr: initial learning rate
        # lrd: learning rate decay
        # lrm: learning rate minimum
        # me:  maximum number of epochs
        # ts:  split ratio of training set
        bs, ce, ilr, lrd, lrm, me, ts, kwargs = self.__fit_params__(kwargs)

        self.xMean = np.mean(self.trainX, axis=0, keepdims=True)
        self.xSig = np.std(self.trainX, axis=0, keepdims=True)
        self.yMean = np.mean(self.trainY, axis=0, keepdims=True)
        self.ySig = np.std(self.trainY, axis=0, keepdims=True)
        n = self.trainX.shape[0]
        nTrain = round(ts*n)
        xTrain = self.trainX[:nTrain, :]
        yTrain = self.trainY[:nTrain, :]
        xValid = self.trainX[nTrain:, :]
        yValid = self.trainY[nTrain:, :]

        self.trainX = xTrain
        self.trainY = yTrain
        self.validationX = xValid
        self.validationY = yValid

        loss = QuantileLoss(self.quantiles)

        optimizer = Adam(lr=ilr)
        self.recurrentModel.compile(loss=loss, optimizer=optimizer)
        trainingGen = TrainingGenerator(xTrain, self.xMean, self.xSig, yTrain, self.yMean, self.ySig, None, bs, False, 3)
        validationGen = ValidationGenerator(xValid, self.xMean, self.xSig, yValid, self.yMean, self.ySig, None, 3)
        lr_callback = LRDecay(self.recurrentModel, lrd, lrm, ce)
        self.recurrentModel.fit_generator(trainingGen, steps_per_epoch=nTrain//bs, epochs=me, validation_data=validationGen,
                                          validation_steps=1, callbacks=[lr_callback])

    # predict for validationX
    def predict(self, isDic=False, validationX=None):
        if validationX is None:
            validationX = self.validationX
        validationX = (validationX - self.xMean) / self.xSig
        validationX= np.reshape(validationX, [validationX.shape[0], 1, validationX.shape[1]])
        predictions = self.recurrentModel.predict(validationX)
        predictions = self.yMean + self.ySig*predictions
        if isDic:
            # 采用字典的形式输出，'分为数':[x的预测值]
            dic = {}
            for i in range(self.quantiles.shape[0]):
                key = round(self.quantiles[i], 3)
                dic[key] = predictions[:, i]
            return dic
        return predictions

    # default parameters
    def __fit_params__(self, kwargs):
        batch_size = kwargs.pop("batch_size", 32)
        convergence_epochs = kwargs.pop("convergence_epochs", 10)
        initial_learning_rate = kwargs.pop('initial_learning_rate', 0.01)
        learning_rate_decay = kwargs.pop('learning_rate_decay', 0.9)
        learning_rate_minimum = kwargs.pop('learning_rate_minimum', 1e-4)
        maximum_epochs = kwargs.pop("maximum_epochs", 100)
        training_split = kwargs.pop("training_split", 0.9)
        return batch_size, convergence_epochs, initial_learning_rate, \
            learning_rate_decay, learning_rate_minimum, maximum_epochs, \
            training_split, kwargs


class QRMGM(QuantileRegressionRecurrentModel):

    def constructModel(self, **kwargs):
        recurrentModel = Sequential()
        if self.depth < 2:
            raise Exception("The depth of network is at least 2 but found depth=(" + str(self.depth) + ").")
        else:
            recurrentModel.add(MGM(input_shape=(None, self.trainX.shape[1]), units=self.nodeNum, return_sequences=True))
            recurrentModel.add(Dropout(self.dropout))
            for i in range(self.depth - 2):
                if i == self.depth-3:
                    recurrentModel.add(MGM(units=self.nodeNum, **kwargs))
                else:
                    recurrentModel.add(MGM(units=self.nodeNum, return_sequences=True, **kwargs))
                    recurrentModel.add(Dropout(self.dropout))
            recurrentModel.add(Dense(units=len(self.quantiles), activation=None))
        self.recurrentModel = recurrentModel

class QRLSTM(QuantileRegressionRecurrentModel):

    def constructModel(self, **kwargs):
        recurrentModel = Sequential()
        if self.depth < 2:
            raise Exception("The depth of network is at least 2 but found depth=(" + str(self.depth) + ").")
        else:
            recurrentModel.add(LSTM(input_shape=(None, self.trainX.shape[1]), units=self.nodeNum, return_sequences=True))
            recurrentModel.add(Dropout(self.dropout))
            for i in range(self.depth - 2):
                if i == self.depth-3:
                    recurrentModel.add(LSTM(units=self.nodeNum, **kwargs))
                else:
                    recurrentModel.add(LSTM(units=self.nodeNum, return_sequences=True, **kwargs))
                    recurrentModel.add(Dropout(self.dropout))
            recurrentModel.add(Dense(units=len(self.quantiles), activation=None))
        self.recurrentModel = recurrentModel


class QRGRU(QuantileRegressionRecurrentModel):

    def constructModel(self, **kwargs):
        recurrentModel = Sequential()
        if self.depth < 2:
            raise Exception("The depth of network is at least 2 but found depth=(" + str(self.depth) + ").")
        else:
            recurrentModel.add(GRU(input_shape=(None, self.trainX.shape[1]), units=self.nodeNum, return_sequences=True))
            recurrentModel.add(Dropout(self.dropout))
            for i in range(self.depth - 2):
                if i == self.depth-3:
                    recurrentModel.add(GRU(units=self.nodeNum, **kwargs))
                else:
                    recurrentModel.add(GRU(units=self.nodeNum, return_sequences=True, **kwargs))
                    recurrentModel.add(Dropout(self.dropout))
            recurrentModel.add(Dense(units=len(self.quantiles), activation=None))
        self.recurrentModel = recurrentModel


class QRNN(QuantileRegressionRecurrentModel):

    def constructModel(self, **kwargs):
        recurrentModel = Sequential()
        if self.depth < 2:
            raise Exception("The depth of network is at least 2 but found depth=(" + str(self.depth) + ").")
        else:
            recurrentModel.add(Dense(input_dim=self.trainX.shape[1], units=self.nodeNum, activation=self.activation, **kwargs))
            recurrentModel.add(Dropout(self.dropout))
            for i in range(self.depth - 2):
                if i == self.depth-3:
                    recurrentModel.add(Dense(units=self.nodeNum, activation=self.activation, **kwargs))
                else:
                    recurrentModel.add(Dense(units=self.nodeNum, activation=self.activation, **kwargs))
                    recurrentModel.add(Dropout(self.dropout))
            recurrentModel.add(Dense(units=len(self.quantiles), activation=None))
        self.recurrentModel = recurrentModel

    # train model
    def fit(self, **kwargs):
        # Handle parameters
        # bs:  batch size
        # ce:  convergence epochs
        # ilr: initial learning rate
        # lrd: learning rate decay
        # lrm: learning rate minimum
        # me:  maximum number of epochs
        # ts:  split ratio of training set
        bs, ce, ilr, lrd, lrm, me, ts, kwargs = self.__fit_params__(kwargs)

        self.xMean = np.mean(self.trainX, axis=0, keepdims=True)
        self.xSig = np.std(self.trainX, axis=0, keepdims=True)
        self.yMean = np.mean(self.trainY, axis=0, keepdims=True)
        self.ySig = np.std(self.trainY, axis=0, keepdims=True)
        n = self.trainX.shape[0]
        nTrain = round(ts*n)
        xTrain = self.trainX[:nTrain, :]
        yTrain = self.trainY[:nTrain, :]
        xValid = self.trainX[nTrain:, :]
        yValid = self.trainY[nTrain:, :]

        self.trainX = xTrain
        self.trainY = yTrain
        self.validationX = xValid
        self.validationY = yValid

        loss = QuantileLoss(self.quantiles)

        optimizer = Adam(lr=ilr)
        self.recurrentModel.compile(loss=loss, optimizer=optimizer)
        trainingGen = TrainingGenerator(xTrain, self.xMean, self.xSig, yTrain, self.yMean, self.ySig, None, bs, False, 2)
        validationGen = ValidationGenerator(xValid, self.xMean, self.xSig, yValid, self.yMean, self.ySig, None, 2)
        lr_callback = LRDecay(self.recurrentModel, lrd, lrm, ce)
        self.recurrentModel.fit_generator(trainingGen, steps_per_epoch=nTrain//bs, epochs=me, validation_data=validationGen,
                                          validation_steps=1, callbacks=[lr_callback])

    # predict for validationX
    def predict(self, isDic=False, validationX=None):
        if validationX is None:
            validationX = self.validationX
        validationX = (validationX - self.xMean) / self.xSig
        predictions = self.recurrentModel.predict(validationX)
        predictions = self.yMean + self.ySig*predictions
        if isDic:
            # 采用字典的形式输出，'分为数':[x的预测值]
            dic = {}
            for i in range(self.quantiles.shape[0]):
                key = round(self.quantiles[i], 3)
                dic[key] = predictions[:, i]
            return dic
        return predictions

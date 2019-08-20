import numpy as np
from enum import Enum


class DataName(Enum):
    WindSpeedInInnerMongoliaAll = 1


class DataSetFactory:
    def getDataset(dataName, param=None):
        if dataName is DataName.WindSpeedInInnerMongoliaAll:
            x_train, y_train = DataSetFactory.getWindSpeedInInnerMongoliaAll(param)
        return x_train, y_train

    def getWindSpeedInInnerMongoliaAll(param):
        if param is None:
            usecol = 1
            step = 2
        else:
            usecol = 1
            step = param['step']
        data = np.loadtxt("../data/windspeed/WindSpeedInInnerMongolia.csv", delimiter=",", usecols=usecol)
        x_train = []
        y_train = []
        for i in range(data.shape[0]-step):
            x_train.append(data[i:i+step])
            y_train.append(data[i+step])
        x_train = np.array(x_train)
        y_train = np.array(y_train)
        return x_train, y_train



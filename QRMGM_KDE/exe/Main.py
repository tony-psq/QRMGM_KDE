# -*- coding: utf-8 -*-
# @File         : Main.py
# @Author       : Zhendong Zhang
# @Email        : zzd_zzd@hust.edu.cn
# @University   : Huazhong University of Science and Technology
# @Date         : 2019/8/19
# @Software     : PyCharm
# -*---------------------------------------------------------*-

import numpy as np
from QRMGM_KDE.model.implModel.ModelFactory import QRMGM, QRLSTM, QRGRU, QRNN
from QRMGM_KDE.dataoperator.Dataset import DataSetFactory, DataName
from QRMGM_KDE.evaluation.Evaluation import Evaluation
from QRMGM_KDE.evaluation.Draw import Draw
import math
from QRMGM_KDE.model.implModel.KernelDensityEstimation import KernelDensityEstimation
from datetime import datetime
from scipy import stats
import time
import pandas as pd
import os

if __name__ == '__main__':

    dataName = DataName.WindSpeedInInnerMongoliaAll
    quantiles = np.arange(0.005, 1, 0.005)
    starts = [500, 4750, 3000, 0]
    ends = [1500, 5750, 5000, 2500]
    steps = [3, 4, 6, 3]
    tt = {}
    isDrawPIT = True
    isDrawResults = True
    isDrawPDF = True
    isSave = True
    isShow = False

    resultSaveBasePath = '../results/'
    plotSaveBasePath = '../plots/'
    timeFlag = str(int(time.time()))

    modelNames = ['QRMGM', 'QRLSTM', 'QRGRU', 'QRNN']

    for i in range(len(starts)):
        metricSavePath = resultSaveBasePath + timeFlag + '/metrics_dataset_' + str(i+1) + '_' + timeFlag + '.xlsx'
        metricWriter = pd.ExcelWriter(metricSavePath)
        metrics = []
        metricIndex = []
        param = dict()
        param['step'] = steps[i]
        start = starts[i]
        end = ends[i]
        x_train, y_train = DataSetFactory.getDataset(dataName=dataName, param=param)
        x_train = x_train[start:end, :]
        y_train = y_train[start:end]
        y_train = y_train.reshape([y_train.shape[0], 1])
        input_dim = x_train.shape[1]

        resultSavePath = resultSaveBasePath + timeFlag + '/dataset_' + str(i+1) + '_results_' + timeFlag + '.xlsx'
        plotSavePath = plotSaveBasePath + timeFlag + '/dataset_' + str(i+1) + '_'
        if not os.path.exists(resultSaveBasePath + timeFlag):
            os.makedirs(resultSaveBasePath + timeFlag)
            os.makedirs(plotSaveBasePath + timeFlag)

        resultWriter = pd.ExcelWriter(resultSavePath)
        for modelName in modelNames:
            hyperParameters = dict()
            hyperParameters['depth'] = 3
            hyperParameters['nodeNum'] = 32
            hyperParameters['quantiles'] = quantiles
            hyperParameters['activation'] = 'tanh'
            hyperParameters['dropout'] = 0.2

            # 获取模型
            if modelName == 'QRMGM':
                model = QRMGM(x_train, y_train, x_train, y_train, hyperParameters)
            elif modelName == 'QRLSTM':
                model = QRLSTM(x_train, y_train, x_train, y_train, hyperParameters)
            elif modelName == 'QRGRU':
                model = QRGRU(x_train, y_train, x_train, y_train, hyperParameters)
            elif modelName == 'QRNN':
                model = QRNN(x_train, y_train, x_train, y_train, hyperParameters)
            model.constructModel()
            startTime = datetime.now()
            model.fit(initial_learning_rate=0.01,
                      learning_rate_decay=1.5,
                      convergence_epochs=10,
                      batch_size=32,
                      maximum_epochs=200,
                      learning_rate_minimum=1e-4,
                      training_split=0.8)
            endTime = datetime.now()
            predictions = model.predict(isDic=True)
            trainingTime = (endTime-startTime).seconds

            observations = model.validationY[:, 0]

            evaluation = Evaluation()
            mean = predictions[0.5]
            pointMetrics = evaluation.getPointPredictionMetric(predictions=mean, observations=observations)
            print(pointMetrics)

            lower = predictions[0.025]
            upper = predictions[0.975]
            intervalMetrics = evaluation.getIntervalPredictionMetric(lower, upper, observations)
            print(intervalMetrics)

            predictionsArray = np.zeros(shape=(model.validationX.shape[0], len(model.quantiles)))
            j = 0
            for key, value in predictions.items():
                predictionsArray[:, j] = value
                j = j + 1
            probabilityMetrics = evaluation.getProbabilityPredictionMetric(predictionsArray, observations, quantiles)
            print(probabilityMetrics)

            resultsDataFrame = pd.DataFrame(predictionsArray, columns=quantiles)
            resultsDataFrame.to_excel(resultWriter, modelName)

            metric = np.array(
                [pointMetrics['RMSE'], pointMetrics['MAPE'], trainingTime, intervalMetrics['CP'],
                 intervalMetrics['MWP'], 1.0 / intervalMetrics['CM'], probabilityMetrics['CRPS']])
            metrics.append(metric)
            metricIndex.append(modelName)

            reliabilityMetrics = evaluation.getReliabilityMetric(predictionsArray, observations, quantiles)
            draw = Draw()
            if isDrawPIT:
                if modelName == 'QRMGM':
                    draw.drawPIT(reliabilityMetrics['PIT'], cdf=stats.uniform, xlabel='uniform distribution',
                                 ylabel='PIT', title='dataset '+str(i+1), isShow=isShow, isSave=isSave,
                                 savePath=plotSavePath+modelName+'_pit.jpg')
            if isDrawResults:
                locArray = np.array([[0.33, 0.35], [1.0, 0.35], [1.0, 0.35], [0.7, 0.35]])
                draw.drawPredictions(predictions=mean, observations=observations, lower=lower, upper=upper, alpha='95%',
                                     isInterval=True, xlabel='period', ylabel='wind speed(m/s)',
                                     title='dataset ' + str(i + 1), legendLoc=locArray[i, :], isShow=isShow,
                                     isSave=isSave,
                                     savePath=plotSavePath+modelName+'_results.jpg')
            if isDrawPDF:
                if modelName == 'QRMGM':
                    rate = [0, 0.124999, 0.24999, 0.374999, 0.49999, 0.624999, 0.749999, 0.87499, 0.9999]
                    for r in rate:
                        index = math.floor(r * predictionsArray.shape[0])
                        kde = KernelDensityEstimation()
                        samples = predictionsArray[index, :]
                        samples = samples.reshape(len(samples), 1)
                        kd = kde.getKernelDensity(samples=samples)
                        draw.drawKD(kde=kd, samples=samples, observation=observations[index], xlabel='wind speed(m/s)',
                                    ylabel='probability density', title=str(index + 1) + " period", isShow=isShow,
                                    isSave=isSave,
                                    savePath=plotSavePath+modelName+'_period_'+str(index+1)+'_pdf.jpg')
        resultWriter.save()
        resultWriter.close()
        metricDataFrame = pd.DataFrame(metrics, columns=['RMSE', 'MAPE', 'TT', 'CP', 'MWP', 'MC', 'CRPS'],
                                       index=metricIndex)
        metricDataFrame.to_excel(metricWriter, 'dataset_'+str(i+1))
        metricWriter.save()
        metricWriter.close()

    print('finished!')


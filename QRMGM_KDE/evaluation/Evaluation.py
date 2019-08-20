import numpy as np


class Evaluation(object):

    def getPointPredictionMetric(self, predictions, observations, metricNames=None):
        if metricNames is None:
            metricNames = ['MAE', 'MSE', 'RMSE', 'MAPE', 'R2']
        metrics = {}
        for metricName in metricNames:
            metric = None
            if metricName == 'MAE':
                metric = Evaluation.getMAE(predictions, observations)
            elif metricName == 'MSE':
                metric = Evaluation.getMSE(predictions, observations)
            elif metricName == 'RMSE':
                metric = Evaluation.getRMSE(predictions, observations)
            elif metricName == 'MAPE':
                metric = Evaluation.getMAPE(predictions, observations)
            elif metricName == 'R2':
                metric = Evaluation.getRsquare(predictions, observations)
            else:
                raise Exception('unknown point prediction metric name: '+metricName)
            metrics[metricName] = metric
        return metrics

    @staticmethod
    def getMAE(predictions, observations):
        MAE = np.mean(np.abs(predictions-observations))
        return MAE

    @staticmethod
    def getMSE(predictions, observations):
        MSE = np.mean(np.power(predictions-observations, 2))
        return MSE

    @staticmethod
    def getRMSE(predictions, observations):
        MSE = Evaluation.getMSE(predictions, observations)
        RMSE = np.sqrt(MSE)
        return RMSE

    @staticmethod
    def getMAPE(predictions, observations):
        MAPE = np.mean(np.true_divide(np.abs(predictions-observations), np.abs(observations)))
        return MAPE

    @staticmethod
    def getRsquare(predictions, observations):
        mean = np.mean(observations)
        numerator = np.sum(np.power(observations-predictions, 2))
        denominator = np.sum(np.power(observations-mean, 2))
        Rsquare = 1-numerator/denominator
        return Rsquare

    def getIntervalPredictionMetric(self, lower, upper, observations, metricNames=None):
        if metricNames==None:
            metricNames = ['CP', 'MWP', 'CM']
        metrics = {}
        for metricName in metricNames:
            metric = None
            if metricName == 'CP':
                metric = Evaluation.getCP(lower, upper, observations)
            elif metricName == 'MWP':
                metric = Evaluation.getMWP(lower, upper, observations)
            elif metricName == 'CM':
                metric = Evaluation.getCM(lower, upper, observations)
            else:
                raise Exception('unknown interval prediction metric name: '+metricName)
            metrics[metricName] = metric
        return metrics

    @staticmethod
    def getCP(lower, upper, observations):
        N = observations.shape[0]
        count = 0
        for i in range(N):
            if observations[i]>=lower[i] and observations[i]<=upper[i]:
                count = count + 1
        CP = count/N
        return CP

    @staticmethod
    def getMWP(lower, upper, observations):
        N = observations.shape[0]
        MWP = 0
        for i in range(N):
            if upper[i]<lower[i]:
                print(i)
            MWP = MWP + (upper[i]-lower[i])/np.abs(observations[i])
        MWP = MWP/N
        return MWP

    @staticmethod
    def getCM(lower, upper, observations):
        CM = Evaluation.getCP(lower, upper, observations)/Evaluation.getMWP(lower, upper, observations)
        return CM

    def getProbabilityPredictionMetric(self, predictionsArray, observations, quantiles, metricNames=None):
        if metricNames is None:
            metricNames = ['CRPS']
        metrics = {}
        for metricName in metricNames:
            metric = None
            if metricName == 'CRPS':
                metric = Evaluation.getCRPS(predictionsArray, observations, quantiles)
            else:
                raise Exception('unknown probability prediction metric name: '+metric)
            metrics[metricName] = metric
        return metrics

    @staticmethod
    def cdf(predictionsArray, quantiles):
        y_cdf = np.zeros((predictionsArray.shape[0], quantiles.size + 2))
        y_cdf[:, 1:-1] = predictionsArray
        y_cdf[:, 0] = 2.0 * predictionsArray[:, 1] - predictionsArray[:, 2]
        y_cdf[:, -1] = 2.0 * predictionsArray[:, -2] - predictionsArray[:, -3]

        qs = np.zeros((1, quantiles.size + 2))
        qs[0, 1:-1] = quantiles
        qs[0, 0] = 0.0
        qs[0, -1] = 1.0
        return y_cdf, qs

    @staticmethod
    def getCRPS(predictionsArray, observations, quantiles):
        y_cdf, qs = Evaluation.cdf(predictionsArray, quantiles)
        ind = np.zeros(y_cdf.shape)
        ind[y_cdf > observations.reshape(-1, 1)] = 1.0
        CRPS = np.trapz((qs - ind) ** 2.0, y_cdf)
        CRPS = np.mean(CRPS)
        return CRPS

    # 获取可靠性指标
    def getReliabilityMetric(self, predictionsArray, observations, quantiles, metricNames=None):
        if metricNames is None:
            metricNames = ['PIT']
        metrics = {}
        for metricName in metricNames:
            metric = None
            if metricName == 'PIT':
                metric = Evaluation.getPIT(predictionsArray, observations, quantiles)
            else:
                raise Exception('unknown probability prediction metric name: '+metric)
            metrics[metricName] = metric
        return metrics

    @staticmethod
    def getPIT(predictionsArray, observations, quantiles):
        y_cdf, qs = Evaluation.cdf(predictionsArray, quantiles)
        PIT = np.zeros(observations.shape)
        for i in range(observations.shape[0]):
            PIT[i] = np.interp(np.squeeze(observations[i]), np.squeeze(y_cdf[i, :]), np.squeeze(qs))
        return PIT


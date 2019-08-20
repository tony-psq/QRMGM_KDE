from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity
import numpy as np


class KernelDensityEstimation(object):
    def getKernelDensity(self, samples, param_grid=None, kernel='epanechnikov', cv=5):
        if param_grid is None:
            param_grid = {'bandwidth': np.arange(0, 10, 0.5)}
        kde = KernelDensity(kernel=kernel)
        kde_grid = GridSearchCV(estimator=kde, param_grid=param_grid, cv=cv)
        kde = kde_grid.fit(samples).best_estimator_
        #print(kde_grid.best_params_)
        return kde



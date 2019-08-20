################################################################################
# Loss Functions
################################################################################
import keras.backend as K


def skewed_absolute_error(y_true, y_pred, tau):
    """
    The quantile loss function for a given quantile tau:

    L(y_true, y_pred) = (tau - I(y_pred < y_true)) * (y_pred - y_true)

    Where I is the indicator function.
    """
    dy = y_pred - y_true
    return K.mean((1.0 - tau) * K.relu(dy) + tau * K.relu(-dy), axis=-1)


def quantile_loss(y_true, y_pred, taus):
    """
    The quantiles loss for a list of quantiles. Sums up the error contribution
    from the each of the quantile loss functions.
    """
    e = skewed_absolute_error(
        K.flatten(y_true), K.flatten(y_pred[:, 0]), taus[0])
    for i, tau in enumerate(taus[1:]):
        e += skewed_absolute_error(K.flatten(y_true),
                                   K.flatten(y_pred[:, i + 1]),
                                   tau)
    return e


class QuantileLoss:
    """
    Wrapper class for the quantile error loss function. A class is used here
    to allow the implementation of a custom `__repr` function, so that the
    loss function object can be easily loaded using `keras.model.load`.

    Attributes:

        quantiles: List of quantiles that should be estimated with
                   this loss function.

    """

    def __init__(self, quantiles):
        self.__name__ = "Quantile Loss"
        self.quantiles = quantiles

    def __call__(self, y_true, y_pred):
        return quantile_loss(y_true, y_pred, self.quantiles)

    def __repr__(self):
        return "QuantileLoss(" + repr(self.quantiles) + ")"

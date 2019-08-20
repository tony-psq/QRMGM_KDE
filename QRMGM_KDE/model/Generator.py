import numpy as np
import keras

#  This file was pulled form the keras source code and modified.
#  by zzd (zzd_zzd@hust.edu.cn)
################################################################################
# Keras Interface Classes
################################################################################
class TrainingGenerator:
    """
    This Keras sample generator takes the noise-free training data
    and adds independent Gaussian noise to each of the components
    of the input.

    Attributes:

        x_train: The training input, i.e. the brightness temperatures
                 measured by the satellite.
        y_train: The training output, i.e. the value of the retrieval
                 quantity.
        x_mean: A vector containing the mean of each input component.
        x_sigma: A vector containing the standard deviation of each
                 component.
        batch_size: The size of a training batch.
    """

    def __init__(self, x_train, x_mean, x_sigma, y_train, y_mean, y_sigma, sigma_noise, batch_size, is_shuffle, x_dim):
        self.batch_size = batch_size
        self.x_train = x_train
        self.x_mean = x_mean
        self.x_sigma = x_sigma
        self.y_train = y_train
        self.y_mean = y_mean
        self.y_sigma = y_sigma
        self.is_shuffle = is_shuffle
        self.x_dim = x_dim
        self.sigma_noise = sigma_noise
        if is_shuffle:
            self.indices = np.random.permutation(x_train.shape[0])
        else:
            self.indices = np.array(range(x_train.shape[0]))
        self.i = 0

    def __iter__(self):
        print("iter...")
        return self

    def __next__(self):
        inds = self.indices[np.arange(self.i * self.batch_size, (self.i + 1) * self.batch_size) % self.indices.size]
        x_batch = np.copy(self.x_train[inds, :])
        if self.sigma_noise is not None:
            x_batch += np.random.randn(*x_batch.shape) * self.sigma_noise
        x_batch = (x_batch - self.x_mean) / self.x_sigma
        y_batch = (self.y_train[inds] - self.y_mean) / self.y_sigma
        #y_batch = self.y_train[inds]
        self.i = self.i + 1

        if self.is_shuffle:
            # Shuffle training set after each epoch.
            if self.i % (self.x_train.shape[0] // self.batch_size) == 0:
                self.indices = np.random.permutation(self.x_train.shape[0])
        else:
            self.indices = np.array(range(self.x_train.shape[0]))

        if self.x_dim == 3:
            x_batch = np.reshape(x_batch, (x_batch.shape[0], 1, x_batch.shape[1]))

        return x_batch, y_batch


# TODO: Make y-noise argument optional
class ValidationGenerator:
    """
    This Keras sample generator is similar to the training generator
    only that it returns the whole validation set and doesn't perform
    any randomization.

    Attributes:

        x_val: The validation input, i.e. the brightness temperatures
                 measured by the satellite.
        y_val: The validation output, i.e. the value of the retrieval
                 quantity.
        x_mean: A vector containing the mean of each input component.
        x_sigma: A vector containing the standard deviation of each
                 component.
    """

    def __init__(self, x_val, x_mean, x_sigma, y_val, y_mean, y_sigma,  sigma_noise, x_dim):
        self.x_val = x_val
        self.x_mean = x_mean
        self.x_sigma = x_sigma

        self.y_val = y_val
        self.y_mean = y_mean
        self.y_sigma = y_sigma

        self.sigma_noise = sigma_noise
        self.x_dim = x_dim

    def __iter__(self):
        return self

    def __next__(self):
        x_val = np.copy(self.x_val)
        if self.sigma_noise is not None:
            x_val += np.random.randn(*self.x_val.shape) * self.sigma_noise
        x_val = (x_val - self.x_mean) / self.x_sigma
        y_val = np.copy(self.y_val)
        y_val = (y_val - self.y_mean) / self.y_mean
        if self.x_dim == 3:
            x_val = np.reshape(x_val, [x_val.shape[0], 1, x_val.shape[1]])

        return x_val, y_val


class LRDecay(keras.callbacks.Callback):
    """
    The LRDecay class implements the Keras callback interface and reduces
    the learning rate according to validation loss reduction.

    Attributes:

        lr_decay: The factor c > 1.0 by which the learning rate is
                  reduced.
        lr_minimum: The training is stopped when this learning rate
                    is reached.
        convergence_steps: The number of epochs without validation loss
                           reduction required to reduce the learning rate.

    """

    def __init__(self, model, lr_decay, lr_minimum, convergence_steps):
        self.model = model
        self.lr_decay = lr_decay
        self.lr_minimum = lr_minimum
        self.convergence_steps = convergence_steps
        self.steps = 0

    def on_train_begin(self, logs={}):
        self.losses = []
        self.steps = 0
        self.min_loss = 1e30

    def on_epoch_end(self, epoch, logs={}):
        self.losses += [logs.get('val_loss')]
        if not self.losses[-1] < self.min_loss:
            self.steps = self.steps + 1
        else:
            self.steps = 0
        if self.steps > self.convergence_steps:
            lr = keras.backend.get_value(self.model.optimizer.lr)
            keras.backend.set_value(
                self.model.optimizer.lr, lr / self.lr_decay)
            self.steps = 0
            print("\nReduced learning rate to " + str(lr))

            if lr < self.lr_minimum:
                self.model.stop_training = True

        self.min_loss = min(self.min_loss, self.losses[-1])

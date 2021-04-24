"""
Data loader for the matter power spectrum
"""

import os
import numpy as np

from .gpemulator_singlebin import _map_params_to_unit_cube as input_normalize


class PowerSpecs:
    """
    A data loader to load multi-fidelity training and test data

    Assume two fidelities.
    """

    def __init__(self, folder: str = "data/50_LR_3_HR/", n_fidelities: int = 2):
        self.n_fidelities = n_fidelities

        # training data
        self.X_train = []
        self.Y_train = []
        for i in range(n_fidelities):
            x_train = np.loadtxt(
                os.path.join(folder, "train_input_fidelity_{}.txt".format(i))
            )
            y_train = np.loadtxt(
                os.path.join(folder, "train_output_fidelity_{}.txt".format(i))
            )

            self.X_train.append(x_train)
            self.Y_train.append(y_train)

        # parameter limits for normalization
        self.parameter_limits = np.loadtxt(os.path.join(folder, "input_limits.txt"))

        # testing data
        self.X_test = []
        self.Y_test = []
        self.X_test.append(np.loadtxt(os.path.join(folder, "test_input.txt")))
        self.Y_test.append(np.loadtxt(os.path.join(folder, "test_output.txt")))

        # load k bins (in log)
        self.kf = np.loadtxt(os.path.join(folder, "kf.txt"))

        assert len(self.kf) == self.Y_test[0].shape[1]
        assert len(self.kf) == self.Y_train[0].shape[1]

    @property
    def X_train_norm(self):
        """
        Normalized input parameters
        """
        x_train_norm = []
        for x_train in self.X_train:
            x_train_norm.append(input_normalize(x_train, self.parameter_limits))

        return x_train_norm

    @property
    def X_test_norm(self):
        """
        Normalized input parameters
        """
        x_test_norm = []
        for x_test in self.X_test:
            x_test_norm.append(input_normalize(x_test, self.parameter_limits))

        return x_test_norm

    @property
    def Y_train_norm(self):
        """
        Normalized training output. Subtract the low-fidelity data with
        their sample mean. Don't change high-fidelity data.
        """
        y_train_norm = []
        for y_train in self.Y_train[:-1]:
            mean = y_train.mean(axis=0)
            y_train_norm.append(y_train - mean)

        # don't change high-fidelity data
        y_train_norm.append(self.Y_train[-1])

        return y_train_norm

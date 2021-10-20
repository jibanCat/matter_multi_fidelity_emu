# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# ----
# Modified by jibancat to adopt multi-information source modelling
#
# from MISO paper, for each information source (IS), the function could be modelled as
# f(l, x) = f(0, x) + \delta_l(x)
# with
# \delta_l ~ GP(\mu_l, \Sigma_l)
# are a separate independent Gaussian process. The modelling choice for \mu_l is
# assuming \mu_l(x) = 0.
# 
# Give that, for a MISO model,
# \mu(l, x) = \mu_0(x)
# \Sigma((l, x), (l', x')) = Cov( f(0, x) + \delta_l(x), f(0, x') + \delta_l'(x) )
# = \Sigma_0(x, x') + I_{l, l'} \Sigma_l(x, x')

"""
Contains code for non-linear model multi-information source model.

It is based on this paper:

Nonlinear information fusion algorithms for data-efficient multi-fidelity modelling.
P. Perdikaris, M. Raissi, A. Damianou, N. D. Lawrence and G. E. Karniadakis (2017)
http://web.mit.edu/parisp/www/assets/20160751.full.pdf

Multiple-Information Source Optimization

Original code from:
https://github.com/emukit/emukit

Modified by jibancat to:
1. Only make low-fidelity use ARD; force high-fidelity turn off ARD.
2. Remove the condition on fix noise due to did not find the result not converged.
3. Adapt to multiple information source
"""
from typing import Tuple, List, Type

import numpy as np
import GPy

from emukit.core.interfaces import IModel, IDifferentiable
from emukit.multi_fidelity.convert_lists_to_array import convert_y_list_to_array, convert_x_list_to_array
from numpy.core import numeric


def make_non_linear_kernels(base_kernel_class: Type[GPy.kern.Kern],
                            n_IS: int, n_input_dims: int, ARD: bool=False,
                            n_output_dim: int = 1) -> List:
    """
    This function takes a base kernel class and constructs the structured multi-information source kernels

    The first level is for separate GPs for each information source,
    the second level is for a kernel to combine the outputs from each information source

    At the first level the kernel is simply:
    .. math
        k_{base,IS1}(x, x') + k_{base, IS2}(x, x') + ... etc

    At the subsequent level, the kernel is of the form
    .. math
        k_{base,IS}(x, x')k_{base,IS}( [y_{i-1,IS1}, y_{i-1,IS2}, ... etc], [y{i-1,IS1}', y_{i-1,IS2}', ... etc]) +
        + k_{base,IS0}(x, x')

    :param base_kernel_class: GPy class definition of the kernel type to construct the kernels at
    :param n_fidelities: Number of fidelities in the model. A kernel will be returned for each fidelity
    :param n_input_dims: The dimensionality of the input.
    :param ARD: If True, uses different lengthscales for different dimensions. Otherwise the same lengthscale is used
                for all dimensions. Default False.
    :return: A list of kernels with one entry for each fidelity starting from 0 level to information sources
    """
    # should have n_IS * n_input_dims input space for the first level,
    # corresponds to n_IS separate GPs.
    all_base_dims_list = [
        list(range(n_input_dims))
        for IS in range(n_IS)
    ]

    # output dimensions are after first level separate GPs
    all_out_dims_list = [
        list(range( n_input_dims + IS * n_output_dim, n_input_dims + (IS + 1) * n_output_dim))
        for IS in range(n_IS)
    ]

    # level 1:
    # ----
    # Separate independent GPs
    # k_{base,IS1}(x, x') + k_{base, IS2}(x, x') + ... etc
    #
    # You should have enough data to train them because they are low-fidelity.
    kernels_1 = []
    for IS in range(n_IS):
        # separate GPs: only activate the dimension corresponds to this information source
        kernels_1.append(
            base_kernel_class(
                n_input_dims,
                active_dims=all_base_dims_list[IS],
                ARD=ARD,
                name='kern_level_1_IS_{}'.format(IS)
            )
        )

    # level 2: A kernel combine the outputs from difference information sources
    # \Sigma_0(x, x') + I_{l, l'} \Sigma_l(x, x')
    for i in range(1, 2):
        fidelity_name = 'level' + str(i + 1)

        # for IS in range(n_IS):
        #     # K_\rho means the you belief about the IS
        #     interaction_kernel = base_kernel_class(n_input_dims, active_dims=all_base_dims_list[IS], ARD=False,
        #                                         name='scale_kernel_no_ARD_' + fidelity_name + "_IS_{}".format(IS))
        #     scale_kernel = base_kernel_class(n_output_dim, active_dims=all_out_dims_list[IS], name='previous_level_' + fidelity_name + "_IS_{}".format(IS))

        #     if IS==0:
        #         mixed_kernel = interaction_kernel * scale_kernel
        #     else:
        #         mixed_kernel = mixed_kernel + interaction_kernel * scale_kernel

        interaction_kernel = base_kernel_class(n_input_dims, active_dims=all_base_dims_list[0], ARD=ARD,
                                            name='scale_kernel_no_ARD_' + fidelity_name)
        scale_kernel = base_kernel_class(n_output_dim * n_IS, active_dims=list(range(n_input_dims, n_input_dims + n_IS)), name='previous_level_' + fidelity_name, ARD=ARD)

        mixed_kernel = interaction_kernel * scale_kernel


        # bias kernel is the covariance for the 0 level, \Sigma_0
        bias_kernel = base_kernel_class(n_input_dims, active_dims=all_base_dims_list[0],
                                        ARD=ARD, name='bias_kernel_no_ARD_' + fidelity_name)

        kernel_0 = mixed_kernel + bias_kernel

    # return [K0, K_IS1, K_IS2, etc]
    return [kernel_0] + kernels_1


class NonLinearMultiInformationModel(IModel, IDifferentiable):
    """
    Non-linear Model for multiple fidelities. This implementation of the model only handles 1-dimensional outputs.

    The theory implies the training points should be nested such that any point in a higher fidelity exists in all lower
    fidelities, in practice the model will work if this constraint is ignored.
    """

    def __init__(self, X_init: np.ndarray, Y_init: np.ndarray, n_IS: int, kernels: List[GPy.kern.Kern],
                 n_samples=100, verbose=False, optimization_restarts=5) -> None:
        """
        By default the noise at intermediate levels will be fixed to 1e-4.

        Note: num_fidelities should be fixed to 2

        :param X_init: Initial X values.
        :param Y_init: Initial Y values.
        :param n_IS: Number of information sources in problem.
        :param kernels: List of kernels for each GP model at each level. The first level has multiple kernels for multiple
                        information sources. The first kernel should take input of dimension d_in. The second kernel should
                        take input of dimension (d_in+1 * n_IS) where d_in is
                        the dimensionality of the features (input parameters).
        :param n_samples: Number of samples to use to do quasi-Monte-Carlo integration at each level. Default 100
        :param verbose: Whether to output messages during optimization. Defaults to False.
        :param optimization_restarts: Number of random restarts
                                      when optimizing the Gaussian processes' hyper-parameters.
        """

        if not isinstance(X_init, np.ndarray):
            raise TypeError('X_init expected to be a numpy array')

        if not isinstance(Y_init, np.ndarray):
            raise TypeError('Y_init expected to be a numpy array')

        self.verbose = verbose
        self.optimization_restarts = optimization_restarts

        # [IS] not change the original emukit's n_fidelities variable
        # but not it means two levels
        self.n_fidelities = 2
        self.n_IS = n_IS
        self.IS_true = 0

        # Generate random numbers from standardized gaussian for monte-carlo integration
        self.monte_carlo_rand_numbers = np.random.randn(n_samples)[:, np.newaxis]

        # Make lowest fidelity model
        self.models = [0] # 0 for placeholder

        self._fidelity_idx = -1

        # [IS] Level 0 with index 1 ~ N.
        # Note: true IS: index 0; Other IS: 1~N information sources
        # TODO: make sure fidelity idx is consistent throughout the code
        # Note: is_lowest_fidelity - 0 for true simulations, 1 ~ for information sources

        # [IS] Separate GP for multiple information sources
        for IS in range(1, 1 + n_IS):
            is_lowest_fidelity = X_init[:, self._fidelity_idx] == IS #TODO

            self.models.append(
                GPy.models.GPRegression(
                    X_init[is_lowest_fidelity, :-1], # :-1 is for avoiding the index col
                    Y_init[is_lowest_fidelity, :],
                    kernels[IS],
                )
            )

        # [IS] Level 1 with index 0 (true simulation)
        # Make models for fidelities but lowest fidelity
        for i in range(1, self.n_fidelities):
            is_true = X_init[:, self._fidelity_idx] == self.IS_true #TODO
            # Append previous fidelity mean to X
            # [IS] pass outputs from multiple information sources
            augmented_input_list = [X_init[is_true, :-1]]
            for IS in range(1, n_IS + 1):
                previous_mean, _ = self._predict_deterministic_IS(X_init[is_true, :-1], IS)

                augmented_input_list.append(previous_mean)

            # [IS] augmented input = (params, IS output 1, IS output 2, etc)
            augmented_input = np.concatenate(
                augmented_input_list,
                axis=1
            )

            self.models[self.IS_true] = GPy.models.GPRegression(
                augmented_input,
                Y_init[is_true, :],
                kernels[self.IS_true]
            )

        # Remove this condition due to the results are unstable and not coverged
        # # Fix noise parameters for all models except top fidelity
        # for model in self.models[:-1]:
        #     model.Gaussian_noise.fix(1e-4)

    def set_data(self, X: np.ndarray, Y: np.ndarray) -> None:
        """
        Updates training data in the model.

        :param X: New training features.
        :param Y: New training targets.
        """
        # [IS] set all information sources; 0 index for true simulation
        for IS in range(1, self.n_IS + 1):
            is_IS = (IS == X[:, -1])

            X_low_fidelity = X[is_IS, :-1]
            Y_low_fidelity = Y[is_IS, :]
            self.models[IS].set_XY(X_low_fidelity, Y_low_fidelity)

        # [IS] we forced set n_fidelities = 2
        for i in range(1, self.n_fidelities):
            is_true = (self.IS_true == X[:, -1])

            X_this_fidelity = X[is_true, :-1]
            Y_this_fidelity = Y[is_true, :]

            # [IS] pass outputs from multiple information sources
            augmented_input_list = [X_this_fidelity]
            for IS in range(self.n_IS):
                previous_mean, _ = self._predict_deterministic_IS(X_this_fidelity, IS)

                augmented_input_list.append(previous_mean)

            # [IS] augmented input = (params, IS output 1, IS output 2, etc)
            augmented_input = np.concatenate(
                augmented_input_list,
                axis=1
            )

            self.models[self.IS_true].set_XY(augmented_input, Y_this_fidelity)

    @property
    def X(self):
        """
        :return: input array of size (n_points x n_inputs_dims) across every fidelity in original input domain meaning
                 it excludes inputs to models that come from the output of the previous level
        """
        # [IS] true X is from index 0
        x_list = [self.models[0].X[:, :-1]]

        for IS in range(1, self.n_IS + 1):
            x_list.append(self.models[IS].X)

        return convert_x_list_to_array(x_list)

    @property
    def Y(self):
        """
        :return: output array of size (n_points x n_outputs) across every fidelity level
        """
        return convert_y_list_to_array([model.Y for model in self.models])

    @property
    def n_samples(self):
        return self.monte_carlo_rand_numbers.shape[0]

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predicts mean and variance at fidelity given by the last column of X

        Note that the posterior isn't Gaussian and so this function doesn't tell us everything about our posterior
        distribution.

        :param X: Input locations with fidelity index appended.
        :returns: mean and variance of posterior distribution at X.
        """

        fidelity = X[:, self._fidelity_idx]

        # Do prediction 1 test point at a time
        variance = np.zeros((X.shape[0], 1))
        mean = np.zeros((X.shape[0], 1))

        for i in range(X.shape[0]):
            sample_mean, sample_var = self._predict_samples(X[[i], :-1], fidelity[i])
            # Calculate total variance and mean from samples
            variance[i, :] = np.mean(sample_var) + np.var(sample_mean)
            mean[i, :] = np.mean(sample_mean)

        return mean, variance

    def get_prediction_gradients(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predicts mean and variance and the gradients of the mean and variance with respect to X.

        :param X: input location.
        :returns: (mean, mean gradient, variance, variance gradient) Gradients will be shape (n_points x (d-1)) because
                  we don't return the gradient with respect to the fidelity index.
        """

        fidelity = X[:, self._fidelity_idx]

        # Initialise vectors
        sample_mean = np.zeros((self.n_samples ** (self.n_fidelities - 1), X.shape[0]))
        d_sample_mean_dx = np.zeros((self.n_samples ** (self.n_fidelities - 1), X.shape[0], X.shape[1] - 1))
        d_sample_var_dx = np.zeros((self.n_samples ** (self.n_fidelities - 1), X.shape[0], X.shape[1] - 1))

        # Iteratively obtain predictions and associated gradients for each input point
        for i in range(X.shape[0]):
            mean, dmean_dx, var, dvar_dx = self._predict_samples_with_gradients(X[[i], :-1], fidelity[i])
            # Assign to outputs
            sample_mean[:, [i]] = mean
            d_sample_mean_dx[:, i, :] = dmean_dx
            d_sample_var_dx[:, i, :] = dvar_dx

        # Calculate means + total variance
        total_mean = np.mean(sample_mean, axis=0, keepdims=True).T
        total_mean_grad = np.mean(d_sample_mean_dx, axis=0)

        # Calculate total variance derivative
        tmp = 2 * np.mean(d_sample_mean_dx * sample_mean[:, :, None], axis=0)
        total_variance_grad = np.mean(d_sample_var_dx, axis=0) + tmp - 2 * total_mean * total_mean_grad

        return total_mean_grad, total_variance_grad

    def _predict_samples(self, X: np.ndarray, fidelity: float):
        """
        Draw samples from model at given fidelity. Returns samples of mean and variance at specified fidelity.
        :param X: Input array without output of previous layer appended.
        :param fidelity: zero based fidelity index.
        :returns sample_mean, sample_variance: mean and variance predictions at input points.
        """

        fidelity = int(fidelity)

        # Predict at first fidelity
        # [IS] Get the predictions from IS first
        all_sample_means = []
        all_sample_variances = []
        for IS in range(1, self.n_IS + 1):
            sample_mean, sample_variance = self.models[IS].predict(X)

            all_sample_means.append(sample_mean)
            all_sample_variances.append(sample_variance)

        # [IS] if not passing through the next level, return the outputs from all IS.
        sample_mean = np.concatenate(
            all_sample_means,
            axis=1
        )
        sample_variance = np.concatenate(
            all_sample_variances,
            axis=1
        )

        # Predict at all fidelities up until the one we are interested in
        # [IS] predict the true index: 0
        for i in range(1, fidelity + 1):
            sample_mean, sample_variance, _ = self._propagate_samples_through_level(X, i, all_sample_means, all_sample_variances)
        return sample_mean, sample_variance

    def _predict_samples_with_gradients(self, X: np.ndarray, fidelity: float):
        """
        Draw samples of mean and variance from model at given fidelity and the gradients of these samples wrt X.

        We calculate the gradients by applying the chain rule as the gradients of each Gaussian process is known wrt
        its inputs.

        :param X: Input array without output of previous layer appended.
        :param fidelity: zero based fidelity index.
        :returns mean, mean gradient, variance, variance gradient: mean and variance predictions at input points.
        """

        fidelity = int(fidelity)

        # Predict at first fidelity
        # [IS] Predict for information sources
        all_dsample_mean_dxs = []
        all_dsample_var_dxs = []
        all_sample_means = []
        all_sample_variances = []
        for IS in range(1, self.n_IS):
            dsample_mean_dx, dsample_var_dx = self.models[IS].predictive_gradients(X)
            dsample_mean_dx = dsample_mean_dx[:, :, 0] #TODO: not sure what 0 means
            sample_mean, sample_variance = self.models[IS].predict(X)

            all_dsample_mean_dxs.append(dsample_mean_dx)
            all_dsample_var_dxs.append(dsample_var_dx)
            all_sample_means.append(sample_mean)
            all_sample_variances.append(sample_variance)

        # [IS] if not passing through the next level, return the outputs from all IS.
        sample_mean = np.concatenate(
            all_sample_means,
            axis=1
        )
        sample_variance = np.concatenate(
            all_sample_variances,
            axis=1
        )
        dsample_mean_dx = np.concatenate(
            all_dsample_mean_dxs,
            axis=1
        )
        dsample_var_dx = np.concatenate(
            all_dsample_var_dxs,
            axis=1
        )

        for i in range(1, fidelity + 1):
            # [IS] no need to copy since we are not overwriting
            # previous_sample_variance = sample_variance.copy()
            # Predict at all fidelities up until the one we are interested in
            # [IS] this part should assume we get the output from 0 index (true index)
            sample_mean, sample_variance, x_augmented = \
                self._propagate_samples_through_level(X, i, all_sample_means, all_sample_variances)
            dsample_mean_dx, dsample_var_dx = \
                self._propagate_samples_through_level_gradient(all_dsample_mean_dxs, all_dsample_var_dxs,
                                                               i, all_sample_variances, x_augmented)
        return sample_mean, dsample_mean_dx, sample_variance, dsample_var_dx

    def _propagate_samples_through_level(self, X, i_level, all_sample_means, all_sample_variances):
        """
        Sample from the posterior of all information sources and propagates these samples through next level (index 0).

        :param X: Input array without output of previous layer appended.
        :param i_level: level to push through
        :param all_sample_means: mean from previous level, including all IS in a list.
        :param all_sample_variances: variance from previous level, , including all IS in a list.
        """
        assert self.n_IS == len(all_sample_means)

        # Create inputs for each sample
        x_repeat = np.repeat(X, self.n_samples ** i_level, axis=0)

        x_augmented_list = [x_repeat]

        # [IS] MC sample all of the information sources, and then do the same trick as the original MF model
        for IS in range(self.n_IS):
            sample_mean = all_sample_means[IS]
            sample_variance = all_sample_variances[IS]

            # Draw samples from posterior of previous fidelity for each IS
            samples = self.monte_carlo_rand_numbers * np.sqrt(sample_variance) + sample_mean.T
            samples = samples.flatten()[:, None]

            x_augmented_list.append(samples)

        # Augment input with mean of previous fidelity
        x_augmented = np.concatenate(
            x_augmented_list,
            axis=1
        )

        # Predict mean and variance for true IS, index 0
        sample_mean, sample_variance = self.models[self.IS_true].predict(x_augmented)
        return sample_mean, sample_variance, x_augmented

    def _propagate_samples_through_level_gradient(self, all_dsample_mean_dxs, all_dsample_var_dxs, i_fidelity, all_sample_variances,
                                                  x_augmented):
        """
        Calculates gradients of sample mean and variance with respect to X when propagated through a level

        :param dsample_mean_dx: Gradients of mean prediction of samples from previous level, including all information sources
        :param dsample_var_dx: Gradients of variance prediction of samples from previous level, including all information sources
        :param i_fidelity: level index
        :param sample_variance: The variance prediction of the samples from the previous level, including all information sources
        :param x_augmented: The X input for this level augmented with the outputs
                            from the previous level as the final column
        """
        # Get partial derivatives of mean and variance with respect to
        # both X and output of previous fidelity
        # [IS] derivatives from index 0 (true)
        dmean_dx, dvar_dx = self.models[self.IS_true].predictive_gradients(x_augmented)
        dmean_dx = dmean_dx[:, :, 0]

        # [IS] Adding up derivatives from each IS
        for IS in range(self.n_IS):
            sample_variance = all_sample_variances[IS]
            dsample_var_dx = all_dsample_var_dxs[IS]
            dsample_mean_dx = all_dsample_mean_dxs[IS]

            # Convert variance derivative to std derivative
            clipped_var = np.clip(sample_variance, 1e-10, np.inf)
            dsample_std_dx = dsample_var_dx / (2 * np.sqrt(clipped_var))
            # Calculate gradients of samples wrt x
            # This calculates a (n_samples**(i-1), n_samples, n_dims) matrix
            tmp = self.monte_carlo_rand_numbers[:, np.newaxis, :] * dsample_std_dx[:, np.newaxis, :]
            dsamples_dx = dsample_mean_dx[np.newaxis, :, :] + tmp
            dsamples_dx_reshaped = np.reshape(dsamples_dx, (self.n_samples ** i_fidelity, dsample_std_dx.shape[1]))

            if IS == 0:                
                # Combine partial derivatives to get full derivative wrt X
                dsample_mean_dx = dmean_dx[:, :-1] + dmean_dx[:, [-1]] * dsamples_dx_reshaped
                dsample_var_dx = dvar_dx[:, :-1] + dvar_dx[:, [-1]] * dsamples_dx_reshaped
            else:
                dsample_mean_dx += dmean_dx[:, [-1]] * dsamples_dx_reshaped
                dsample_var_dx += dvar_dx[:, [-1]] * dsamples_dx_reshaped

        return dsample_mean_dx, dsample_var_dx

    def optimize(self) -> None:
        """
        Optimize the full model
        """

        # Optimize the first model
        # [IS] Optimize the models for IS first
        for IS in range(1, self.n_IS + 1):
            self.models[IS].optimize_restarts(self.optimization_restarts, verbose=self.verbose, robust=True)

        # Optimize all models for all fidelities but lowest fidelity
        # [IS] Optimize the true model (index 0)
        for i in range(1, self.n_fidelities):
            # Set new X values because previous model has changed
            augmented_input_list = [self.models[self.IS_true].X[:, :-1]]
            for IS in range(1, self.n_IS + 1):
                previous_mean, _ = self.models[IS].predict(self.models[self.IS_true].X)

                augmented_input_list.append(previous_mean)

            augmented_input = np.concatenate(
                augmented_input_list,
                axis=1
            )
            self.models[self.IS_true].set_X(augmented_input)

            # Optimize parameters
            self.models[self.IS_true].optimize_restarts(self.optimization_restarts, verbose=self.verbose, robust=True)

    def get_f_minimum(self) -> np.ndarray:
        """
        Get the minimum of the top fidelity model.
        """
        return np.min(self.models[self.IS_true].Y)

    def _predict_deterministic(self, X: np.ndarray, fidelity: int) -> Tuple:
        """
        This is a helper function when predicting at points that are in the training set. It is more efficient than
        sampling and is useful when constructing the model.

        IS: 0 for the true simulation; 1 ~ for information sources
        """
        # Predict at first fidelity
        # TODO: check all models[0]
        means_IS = []
        variances_IS = []
        for IS in range(1, self.n_IS + 1):
            mean, variance = self.models[IS].predict(X)

            means_IS.append(mean)
            variances_IS.append(variance)

        mean = np.concatenate(means_IS, axis=1)
        variance = np.concatenate(variances_IS, axis=1)

        for i in range(1, fidelity):
            augmented_input_list = [X]
            for mean_IS in means_IS:
                augmented_input_list.append(mean_IS)
            # Push samples through this fidelity model
            augmented_input = np.concatenate(augmented_input_list, axis=1)
            mean, variance = self.models[self.IS_true].predict(augmented_input)

        return mean, variance

    def _predict_deterministic_IS(self, X: np.ndarray, IS: int) -> Tuple:
        """
        """
        # Predict at first fidelity
        # TODO: check all models[0]
        mean, variance = self.models[IS].predict(X)

        return mean, variance

"""
Multi-Fidelity emulator with output to be a bin value of
power spectrum.
"""
from typing import Tuple, List, Optional, Dict

import numpy as np

import GPy
from emukit.model_wrappers import GPyModelWrapper, GPyMultiOutputWrapper

from emukit.multi_fidelity.kernels import LinearMultiFidelityKernel
from emukit.multi_fidelity.convert_lists_to_array import (
    convert_y_list_to_array,
    convert_xy_lists_to_arrays,
    convert_x_list_to_array,
)

from .non_linear_multi_fidelity_models.non_linear_multi_fidelity_model import NonLinearMultiFidelityModel, make_non_linear_kernels
# from emukit.multi_fidelity.models.non_linear_multi_fidelity_model import NonLinearMultiFidelityModel, make_non_linear_kernels

from .latin_hypercube import map_to_unit_cube_list

class SingleBinGP:
    """
    A GPRegression models GP on each k bin of powerspecs
    
    :param X_hf: (n_points, n_dims) input parameters
    :param Y_hf: (n_points, k modes) power spectrum
    """
    def __init__(self, X_hf: np.ndarray, Y_hf: np.ndarray):
        # a list of GP emulators
        gpy_models: List = []

        # make a GP on each P(k) bin
        for i in range(Y_hf.shape[1]):
            # Standard squared-exponential kernel with a different length scale for
            # each parameter, as they may have very different physical properties.
            nparams = np.shape(X_hf)[1]

            kernel = GPy.kern.RBF(nparams, ARD=True)            

            gp = GPy.models.GPRegression(X_hf, Y_hf[:, [i]], kernel)
            gpy_models.append(gp)

        self.gpy_models = gpy_models

    def optimize_restarts(self, n_optimization_restarts: int) -> None:
        """
        Optimize GP on each bin of the power spectrum.
        """
        models = []

        for i,gp in enumerate(self.gpy_models):
            gp.optimize_restarts(n_optimization_restarts)
            models.append(gp)

        self.models = models

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predicts mean and variance for fidelity specified by last column of X.
        Note that we predict from gp from each k bin.

        :param X: point(s) at which to predict
        :return: predicted P(all k bins) (mean, variance) at X
        """
        means = np.full((X.shape[0], len(self.models)), fill_value=np.nan)
        variances = np.full((X.shape[0], len(self.models)), fill_value=np.nan)

        for i,model in enumerate(self.models):
            mean, variance = model.predict(X)

            means[:, i] = mean[:, 0]
            variances[:, i] = variance[:, 0]

        return means, variances

    def to_dict(self) -> Dict:
        """
        Save hyperparameters into a dict
        """
        param_dict = {}

        for i,model in enumerate(self.models):        

            param_dict["bin_{}".format(i)] = model.to_dict()

        return param_dict

    @staticmethod
    def _map_params_to_unit_cube(
        params: np.ndarray, param_limits: np.ndarray
    ) -> np.ndarray:
        """
        Map the parameters onto a unit cube so that all the variations are
        similar in magnitude.
        
        :param params: (n_points, n_dims) parameter vectors
        :param param_limits: (n_dim, 2) param_limits is a list 
            of parameter limits.
        :return: params_cube, (n_points, n_dims) parameter vectors 
            in a unit cube.
        """
        nparams = np.shape(params)[1]
        params_cube = map_to_unit_cube_list(params, param_limits)
        assert params_cube.shape[1] == nparams

        # this caused problems for selecting too few samples, especially for
        # MF interpolation it's necessary to select only 2~3 points for HF.
        # So not assertion but print warnings. TODO: using logging instead.
        #
        # Check that we span the parameter space
        # note: this is a unit LH cube spanning from \theta \in [0, 1]^num_dim
        for i in range(nparams):
            cond1 = np.max(params_cube[:, i]) > 0.9
            cond2 = np.min(params_cube[:, i]) < 0.1
            if cond1 or cond2:
                print(
                    "[Warning] the LH cube not spanning from \theta \in [0, 1]^num_dim."
                )

        return params_cube

class SingleBinLinearGP:
    """
    A thin wrapper around GPy.core.GP that does some input checking and provides
    a default likelihood.
    Also transform the multi-fidelities params and powerspecs to the Multi-Output
    GP can take.
    And normalize the scale of powerspecs.

    :param params_list:  (n_fidelities, n_points, n_dims) list of parameter vectors.
    :param kf_list:      (n_fidelities, k_modes)
    :param powers_list:  (n_fidelities, n_points, k modes) list of flux power spectra.
    :param param_limits: (n_fidelities, n_dim, 2) list of param_limits.

    :param n_fidelities: number of fidelities stored in the list.

    :param n_restarts (int): number of optimization restarts you want in GPy.
    """

    def __init__(
        self,
        params_list: List[np.ndarray],
        kf_list: List[np.ndarray],
        param_limits_list: List[np.ndarray],
        powers_list: List[np.ndarray],
        kernel_list: Optional[List],
        n_fidelities: int,
        likelihood: GPy.likelihoods.Likelihood = None,
    ):
        # a list of GP emulators
        gpy_models: List = []

        self.n_fidelities = len(params_list)

        # Map the parameters onto a unit cube so that all the variations are
        # similar in magnitude.
        normed_param_list = []
        for i in range(n_fidelities):
            params_cube = self._map_params_to_unit_cube(
                params_list[i], param_limits_list[i]
            )

            normed_param_list.append(params_cube)

        # convert into X,Y for MultiOutputGP
        # not normalize due to no improvements
        X, Y = convert_xy_lists_to_arrays(normed_param_list, powers_list)

        # linear multi-fidelity setup
        if X.ndim != 2:
            raise ValueError("X should be 2d")

        if Y.ndim != 2:
            raise ValueError("Y should be 2d")

        if np.any(X[:, -1] >= n_fidelities):
            raise ValueError(
                "One or more points has a higher fidelity index than number of fidelities"
            )

        # make a GP on each P(k) bin
        for i in range(Y.shape[1]):
            y_metadata = {"output_index": X[:, -1].astype(int)}

            # Make default likelihood as different noise for each fidelity
            likelihood = GPy.likelihoods.mixed_noise.MixedNoise(
                [GPy.likelihoods.Gaussian(variance=1.0) for _ in range(n_fidelities)]
            )

            # Standard squared-exponential kernel with a different length scale for
            # each parameter, as they may have very different physical properties.
            kernel_list = []
            for j in range(n_fidelities):
                nparams = np.shape(params_list[j])[1]

                # kernel = GPy.kern.Linear(nparams, ARD=True)
                # kernel = GPy.kern.RatQuad(nparams, ARD=True)
                kernel = GPy.kern.RBF(nparams, ARD=True)
                
                # final fidelity not ARD due to lack of training data
                if j == n_fidelities - 1:
                    kernel = GPy.kern.RBF(nparams, ARD=False)

                kernel_list.append(kernel)

            # make multi-fidelity kernels
            kernel = LinearMultiFidelityKernel(kernel_list)

            gp = GPy.core.GP(X, Y[:, [i]], kernel, likelihood, Y_metadata=y_metadata)
            gpy_models.append(gp)

        self.gpy_models = gpy_models

    def optimize(self, n_optimization_restarts: int) -> None:
        """
        Optimize GP on each bin of the power spectrum.
        """
        models = []

        for i,gp in enumerate(self.gpy_models):
            # fix noise and optimize
            getattr(gp.mixed_noise, "Gaussian_noise").fix(1e-6)
            for j in range(1, self.n_fidelities):
                getattr(
                    gp.mixed_noise, "Gaussian_noise_{}".format(j)
                ).fix(1e-6)

            model = GPyMultiOutputWrapper(gp, n_outputs=self.n_fidelities, n_optimization_restarts=n_optimization_restarts)

            print("[Info] Optimizing {} bin ...".format(i))

            # first step optimization with fixed noise
            model.gpy_model.optimize_restarts(
                n_optimization_restarts,
                verbose=model.verbose_optimization,
                robust=True,
                parallel=False,
            )

            # unfix noise and re-optimize
            getattr(model.gpy_model.mixed_noise, "Gaussian_noise").unfix()
            for j in range(1, self.n_fidelities):
                getattr(
                    model.gpy_model.mixed_noise, "Gaussian_noise_{}".format(j)
                ).unfix()

            # first step optimization with fixed noise
            model.gpy_model.optimize_restarts(
                n_optimization_restarts,
                verbose=model.verbose_optimization,
                robust=True,
                parallel=False,
            )

            models.append(model)

        self.models = models

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predicts mean and variance for fidelity specified by last column of X.
        Note that we predict from gp from each k bin.

        :param X: point(s) at which to predict
        :return: predicted P(all k bins) (mean, variance) at X
        """
        means = np.full((X.shape[0], len(self.models)), fill_value=np.nan)
        variances = np.full((X.shape[0], len(self.models)), fill_value=np.nan)

        for i,model in enumerate(self.models):
            mean, variance = model.predict(X)

            means[:, i] = mean[:, 0]
            variances[:, i] = variance[:, 0]

        return means, variances

    def to_dict(self) -> Dict:
        """
        Save hyperparameters into a dict
        """
        param_dict = {}

        for i,model in enumerate(self.models):

            this_param_dict = {}
        
            # a constant scaling value
            this_param_dict["scale"] = model.gpy_model.multifidelity.scale.values.tolist()
            # append dict from each key
            for j, kern in enumerate(model.gpy_model.multifidelity.kernels):
                this_param_dict["kern_{}".format(j)] = kern.to_dict()

            param_dict["bin_{}".format(i)] = this_param_dict

        return param_dict

    @staticmethod
    def _map_params_to_unit_cube(
        params: np.ndarray, param_limits: np.ndarray
    ) -> np.ndarray:
        """
        Map the parameters onto a unit cube so that all the variations are
        similar in magnitude.
        
        :param params: (n_points, n_dims) parameter vectors
        :param param_limits: (n_dim, 2) param_limits is a list 
            of parameter limits.
        :return: params_cube, (n_points, n_dims) parameter vectors 
            in a unit cube.
        """
        nparams = np.shape(params)[1]
        params_cube = map_to_unit_cube_list(params, param_limits)
        assert params_cube.shape[1] == nparams

        # this caused problems for selecting too few samples, especially for
        # MF interpolation it's necessary to select only 2~3 points for HF.
        # So not assertion but print warnings. TODO: using logging instead.
        #
        # Check that we span the parameter space
        # note: this is a unit LH cube spanning from \theta \in [0, 1]^num_dim
        for i in range(nparams):
            cond1 = np.max(params_cube[:, i]) > 0.9
            cond2 = np.min(params_cube[:, i]) < 0.1
            if cond1 or cond2:
                print(
                    "[Warning] the LH cube not spanning from \theta \in [0, 1]^num_dim."
                )

        return params_cube

class SingleBinNonLinearGP:
    """
    A thin wrapper around GPy.core.GP that does some input checking and provides
    a default likelihood.
    Also transform the multi-fidelities params and powerspecs to the Multi-Output
    GP can take.
    And normalize the scale of powerspecs.

    :param params_list:  (n_fidelities, n_points, n_dims) list of parameter vectors.
    :param kf_list:      (n_fidelities, k_modes)
    :param powers_list:  (n_fidelities, n_points, k modes) list of flux power spectra.
    :param param_limits: (n_fidelities, n_dim, 2) list of param_limits.

    :param n_fidelities: number of fidelities stored in the list.

    :param n_restarts (int): number of optimization restarts you want in GPy.
    """

    def __init__(
        self,
        params_list: List[np.ndarray],
        kf_list: List[np.ndarray],
        param_limits_list: List[np.ndarray],
        powers_list: List[np.ndarray],
        n_fidelities: int,
        n_samples: int = 100,
        optimization_restarts: int = 5,
    ):
        # a list of GP emulators
        models: List = []

        self.n_fidelities = len(params_list)

        # Map the parameters onto a unit cube so that all the variations are
        # similar in magnitude.
        normed_param_list = []
        for i in range(n_fidelities):
            params_cube = self._map_params_to_unit_cube(
                params_list[i], param_limits_list[i]
            )

            normed_param_list.append(params_cube)

        # convert into X,Y for MultiOutputGP
        # not normalize due to no improvements
        X, Y = convert_xy_lists_to_arrays(normed_param_list, powers_list)

        # linear multi-fidelity setup
        if X.ndim != 2:
            raise ValueError("X should be 2d")

        if Y.ndim != 2:
            raise ValueError("Y should be 2d")

        if np.any(X[:, -1] >= n_fidelities):
            raise ValueError(
                "One or more points has a higher fidelity index than number of fidelities"
            )

        # make a GP on each P(k) bin
        for i in range(Y.shape[1]):
            # make GP non linear kernel
            base_kernel_1 = GPy.kern.RBF
            kernels = make_non_linear_kernels(
                base_kernel_1, n_fidelities, X.shape[1] - 1, ARD=True, n_output_dim=1,
            )  # -1 for the multi-fidelity labels

            model = NonLinearMultiFidelityModel(X, Y[:, [i]], n_fidelities, kernels=kernels, verbose=True, n_samples=n_samples, optimization_restarts=optimization_restarts)

            models.append(model)

        self.models = models

    def optimize(self) -> None:
        """
        Optimize GP on each bin of the power spectrum.
        """

        for i,gp in enumerate(self.models):
            print("[Info] Optimizing {} bin ...".format(i))

            for m in gp.models:
                m.Gaussian_noise.variance.fix(1e-6)
            
            gp.optimize()

            for m in gp.models:
                m.Gaussian_noise.variance.unfix()
            
            gp.optimize()


    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predicts mean and variance for fidelity specified by last column of X.
        Note that we predict from gp from each k bin.

        :param X: point(s) at which to predict
        :return: predicted P(all k bins) (mean, variance) at X
        """
        means = np.full((X.shape[0], len(self.models)), fill_value=np.nan)
        variances = np.full((X.shape[0], len(self.models)), fill_value=np.nan)

        for i,model in enumerate(self.models):
            mean, variance = model.predict(X)

            means[:, i] = mean[:, 0]
            variances[:, i] = variance[:, 0]

        return means, variances

    def to_dict(self) -> Dict:
        """
        Save hyperparameters into a dict
        """
        param_dict = {}

        for i,model in enumerate(self.models):
            this_param_dict = {}
            # append a list of kernel paramaters
            for j, m in enumerate(model.models):
                this_param_dict["fidelity_{}".format(j)] = m.kern.to_dict()

            param_dict["bin_{}".format(i)] = this_param_dict

        return param_dict
    @staticmethod
    def _map_params_to_unit_cube(
        params: np.ndarray, param_limits: np.ndarray
    ) -> np.ndarray:
        """
        Map the parameters onto a unit cube so that all the variations are
        similar in magnitude.
        
        :param params: (n_points, n_dims) parameter vectors
        :param param_limits: (n_dim, 2) param_limits is a list 
            of parameter limits.
        :return: params_cube, (n_points, n_dims) parameter vectors 
            in a unit cube.
        """
        nparams = np.shape(params)[1]
        params_cube = map_to_unit_cube_list(params, param_limits)
        assert params_cube.shape[1] == nparams

        # this caused problems for selecting too few samples, especially for
        # MF interpolation it's necessary to select only 2~3 points for HF.
        # So not assertion but print warnings. TODO: using logging instead.
        #
        # Check that we span the parameter space
        # note: this is a unit LH cube spanning from \theta \in [0, 1]^num_dim
        for i in range(nparams):
            cond1 = np.max(params_cube[:, i]) > 0.9
            cond2 = np.min(params_cube[:, i]) < 0.1
            if cond1 or cond2:
                print(
                    "[Warning] the LH cube not spanning from \theta \in [0, 1]^num_dim."
                )

        return params_cube

class SingleBinDeepGP:
    """
    A thin wrapper around GPy.core.GP that does some input checking and provides
    a default likelihood.
    Also transform the multi-fidelities params and powerspecs to the Multi-Output
    GP can take.
    And normalize the scale of powerspecs.

    :param params_list:  (n_fidelities, n_points, n_dims) list of parameter vectors.
    :param kf_list:      (n_fidelities, k_modes)
    :param powers_list:  (n_fidelities, n_points, k modes) list of flux power spectra.
    :param param_limits: (n_fidelities, n_dim, 2) list of param_limits.

    :param n_fidelities: number of fidelities stored in the list.

    :param n_restarts (int): number of optimization restarts you want in GPy.
    """

    def __init__(
        self,
        params_list: List[np.ndarray],
        kf_list: List[np.ndarray],
        param_limits_list: List[np.ndarray],
        powers_list: List[np.ndarray],
        n_fidelities: int,
    ):
        # DGP model
        from emukit.examples.multi_fidelity_dgp.multi_fidelity_deep_gp import MultiFidelityDeepGP

        # a list of GP emulators
        models: List = []

        self.n_fidelities = len(params_list)

        # Map the parameters onto a unit cube so that all the variations are
        # similar in magnitude.
        normed_param_list = []
        for i in range(n_fidelities):
            params_cube = self._map_params_to_unit_cube(
                params_list[i], param_limits_list[i]
            )

            normed_param_list.append(params_cube)

        # make a GP on each P(k) bin
        for i in range(powers_list[0].shape[1]):

            model = MultiFidelityDeepGP(normed_param_list, [power[:, [i]] for power in powers_list])

            models.append(model)

        self.models = models

    def optimize(self) -> None:
        """
        Optimize GP on each bin of the power spectrum.
        """

        for i,gp in enumerate(self.models):
            print("[Info] Optimizing {} bin ...".format(i))
            gp.optimize()

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predicts mean and variance for fidelity specified by last column of X.
        Note that we predict from gp from each k bin.

        :param X: point(s) at which to predict
        :return: predicted P(all k bins) (mean, variance) at X
        """
        means = np.full((X.shape[0], len(self.models)), fill_value=np.nan)
        variances = np.full((X.shape[0], len(self.models)), fill_value=np.nan)

        for i,model in enumerate(self.models):
            mean, variance = model.predict(X)

            means[:, i] = mean[:, 0]
            variances[:, i] = variance[:, 0]

        return means, variances

    def to_dict(self) -> Dict:
        """
        Save hyperparameters into a dict
        """
        NotImplementedError

    @staticmethod
    def _map_params_to_unit_cube(
        params: np.ndarray, param_limits: np.ndarray
    ) -> np.ndarray:
        """
        Map the parameters onto a unit cube so that all the variations are
        similar in magnitude.
        
        :param params: (n_points, n_dims) parameter vectors
        :param param_limits: (n_dim, 2) param_limits is a list 
            of parameter limits.
        :return: params_cube, (n_points, n_dims) parameter vectors 
            in a unit cube.
        """
        nparams = np.shape(params)[1]
        params_cube = map_to_unit_cube_list(params, param_limits)
        assert params_cube.shape[1] == nparams

        # this caused problems for selecting too few samples, especially for
        # MF interpolation it's necessary to select only 2~3 points for HF.
        # So not assertion but print warnings. TODO: using logging instead.
        #
        # Check that we span the parameter space
        # note: this is a unit LH cube spanning from \theta \in [0, 1]^num_dim
        for i in range(nparams):
            cond1 = np.max(params_cube[:, i]) > 0.9
            cond2 = np.min(params_cube[:, i]) < 0.1
            if cond1 or cond2:
                print(
                    "[Warning] the LH cube not spanning from \theta \in [0, 1]^num_dim."
                )

        return params_cube

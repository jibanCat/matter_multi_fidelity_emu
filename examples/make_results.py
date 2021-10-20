"""
Train and test the models (AR1, NARGP, HF only, LF only)
"""

from typing import List

import numpy as np
from matplotlib import pyplot as plt
import matplotlib

from matter_multi_fidelity_emu.gpemulator_singlebin import (
    SingleBinGP,
    SingleBinLinearGP,
    SingleBinNonLinearGP,
)
from matter_multi_fidelity_emu.data_loader import PowerSpecs

# set a random number seed to reproducibility
np.random.seed(0)

matplotlib.use("pdf")

save_figure = lambda filename: plt.savefig(
    "{}.pdf".format(filename), format="pdf", dpi=300
)


def generate_data(folder: str = "data/50_LR_3_HR"):
    data = PowerSpecs(folder=folder)
    return data


def validate_mf(data: PowerSpecs, model: SingleBinNonLinearGP, fidelity: int = 1):
    """
    Validate the trained MFEmulators
    """
    all_means = []
    all_vars = []
    all_pred_exacts = []
    for n_validations, (x_test, y_test) in enumerate(
        zip(data.X_test_norm[0], data.Y_test[0])
    ):
        x_test_index = np.concatenate(
            (x_test[None, :], np.ones((1, 1)) * fidelity), axis=1
        )
        mean, var = model.predict(x_test_index)

        all_means.append(mean[0])
        all_vars.append(var[0])

        # predicted/exact
        all_pred_exacts.append(10 ** mean[0] / 10 ** y_test)

    return all_means, all_vars, all_pred_exacts


def validate_sf(data: PowerSpecs, model: SingleBinGP):
    """
    Validate the trained single-fidelity emulator
    """
    all_means = []
    all_vars = []
    all_pred_exacts = []
    for n_validations, (x_test, y_test) in enumerate(
        zip(data.X_test_norm[0], data.Y_test[0])
    ):
        mean, var = model.predict(x_test[None, :])

        all_means.append(10 ** mean[0])
        all_vars.append(10 ** var[0])

        # predicted/exact
        all_pred_exacts.append(10 ** mean[0] / 10 ** y_test)

    return all_means, all_vars, all_pred_exacts


def do_benchmark(
    folder: str = "data/50_LR_3_HR",
    n_optimization_restarts: int = 30,
    n_fidelities: int = 2,
    turn_off_bias_nargp: bool = False,
):
    """
    Train and test models, and plot
    1. predicted / exact power spectrum
    2. absolute error plot
    3. parameter plots

    Only support 2 fidelities now.

    Parameters:
    ----
    folder: the folder contains the the training and testing data. See data/50_LR_3_HR
        for example.
    n_optimization_restarts: number of optimization you want to repeat. The GPy will
        choose the best hyperparameters among those repetitions. More is better.
    n_fidelities: only supports 2 now. You may try a larger number but some tweaks might
        be needed.
    turn_off_bias_nargp: not adding bias kernel for NARGP in high-fidelity. In case you
        find the optimization result is not stable, try turning off bias kernel. Some time
        the training data at high-fidelity is not enough to train the bias kernel and
        induce some unstable predictions.
    """

    # get training and testing data. Normalization included.
    data = PowerSpecs(folder=folder)

    # Multi-fidelity
    # linear multi-fidelity
    ar1 = SingleBinLinearGP(
        data.X_train_norm,
        data.Y_train_norm,
        kernel_list=None,
        n_fidelities=n_fidelities,
    )
    # non-linear multi-fidelity
    nargp = SingleBinNonLinearGP(
        data.X_train_norm,
        data.Y_train_norm,
        n_fidelities=n_fidelities,
        n_samples=500,
        optimization_restarts=n_optimization_restarts,
        turn_off_bias=turn_off_bias_nargp,
    )

    # Single-fidelity
    # high-fidelity only emulator
    hf_only = SingleBinGP(data.X_train_norm[-1], data.Y_train[-1])
    lf_only = SingleBinGP(data.X_train_norm[0], data.Y_train[0])

    # optimize each model
    ar1.optimize(n_optimization_restarts=n_optimization_restarts)
    nargp.optimize()
    hf_only.optimize_restarts(n_optimization_restarts=n_optimization_restarts)
    lf_only.optimize_restarts(n_optimization_restarts=n_optimization_restarts)

    # testing set
    means_ar1, vars_ar1, pred_exacts_ar1 = validate_mf(data, model=ar1)
    means_nargp, vars_nargp, pred_exacts_nargp = validate_mf(data, model=nargp)
    means_hfonly, vars_hfonly, pred_exacts_hfonly = validate_sf(data, model=hf_only)
    means_lfonly, vars_lfonly, pred_exacts_lfonly = validate_sf(data, model=lf_only)

    # versus HF
    do_emulator_error_plots(
        data,
        means_ar1,
        means_hfonly,
        pred_exacts_ar1,
        pred_exacts_hfonly,
        label_mf="AR1",
        label_sf="HF only",
        figure_name="ar1",
    )
    do_emulator_error_plots(
        data,
        means_nargp,
        means_hfonly,
        pred_exacts_nargp,
        pred_exacts_hfonly,
        label_mf="NARGP",
        label_sf="HF only",
        figure_name="nargp",
    )
    # versus LF
    do_emulator_error_plots(
        data,
        means_ar1,
        means_lfonly,
        pred_exacts_ar1,
        pred_exacts_lfonly,
        label_mf="AR1",
        label_sf="LF only",
        figure_name="ar1_lf",
    )
    do_emulator_error_plots(
        data,
        means_nargp,
        means_lfonly,
        pred_exacts_nargp,
        pred_exacts_lfonly,
        label_mf="NARGP",
        label_sf="LF only",
        figure_name="nargp_lf",
    )

    # pred/exact plot
    do_pred_exact(data, means_ar1, pred_exacts_ar1, label_mf="AR1", figure_name="ar1")
    do_pred_exact(
        data, means_nargp, pred_exacts_nargp, label_mf="NARGP", figure_name="nargp"
    )


def do_emulator_error_plots(
    data: PowerSpecs,
    means_mf: List[np.ndarray],
    means_sf: List[np.ndarray],
    pred_exacts_mf: List[np.ndarray],
    pred_exacts_sf: List[np.ndarray],
    label_mf: str = "NARGP",
    label_sf: str = "HF only",
    figure_name: str = "",
):
    """
    1. predicted / exact power spectrum
    2. absolute error plot
    """

    # mean emulation error
    emulator_errors = np.abs(np.array(pred_exacts_mf) - 1)
    plt.loglog(
        10 ** data.kf, np.mean(emulator_errors, axis=0), label=label_mf, color="C0"
    )
    plt.fill_between(
        10 ** data.kf,
        y1=np.min(emulator_errors, axis=0),
        y2=np.max(emulator_errors, axis=0),
        color="C0",
        alpha=0.3,
    )

    emulator_errors = np.abs(np.array(pred_exacts_sf) - 1)
    plt.loglog(
        10 ** data.kf, np.mean(emulator_errors, axis=0), label=label_sf, color="C1"
    )
    plt.fill_between(
        10 ** data.kf,
        y1=np.min(emulator_errors, axis=0),
        y2=np.max(emulator_errors, axis=0),
        color="C1",
        alpha=0.3,
    )
    plt.legend()
    plt.ylabel(r"$| P_\mathrm{predicted}(k) / P_\mathrm{true}(k) - 1|$")
    plt.xlabel(r"$k (h/\mathrm{Mpc})$")
    save_figure("absolute_errors_" + figure_name)
    plt.close()
    plt.clf()


def do_pred_exact(
    data: PowerSpecs,
    means_mf: List[np.ndarray],
    pred_exacts_mf: List[np.ndarray],
    label_mf: str = "NARGP",
    figure_name: str = "",
):
    """
    Pred/Exact plot
    """
    for i, pred_exact_mf in enumerate(pred_exacts_mf):
        if i == 0:
            plt.semilogx(
                10 ** data.kf, pred_exact_mf, label=label_mf, color="C{}".format(i)
            )
        else:
            plt.semilogx(10 ** data.kf, pred_exact_mf, color="C{}".format(i))

    plt.legend()
    plt.ylim(0.96, 1.06)
    plt.xlabel(r"$k (h/\mathrm{Mpc})$")
    plt.ylabel(r"$\mathrm{Predicted/Exact}$")
    save_figure("predict_exact_" + figure_name)
    plt.close()
    plt.clf()


def plot_parameters(
    X_train: List[np.ndarray],
    X_test: List[np.ndarray],
    parameter_names: List[str] = [
        r"$\Omega_0$",
        r"$\Omega_b$",
        r"$h$",
        r"$A_s$",
        r"$n_s$",
    ],
):
    """
    Plot the selected samples with all other samples in the input data.
    This would enable us to investigate locations of the selected training samples.
    """
    n_parameters = X_train[0].shape[1]

    for i in range(n_parameters):
        for j in range(i + 1, n_parameters):
            plt.scatter(
                X_train[0][:, i],
                X_train[0][:, j],
                marker="o",
                label="LowRes training data",
                color="C0",
                s=100,
            )
            plt.scatter(
                X_train[1][:, i],
                X_train[1][:, j],
                marker="o",
                label="HighRes training data",
                color="C1",
                s=40,
            )
            plt.scatter(
                X_test[0][:, i],
                X_test[0][:, j],
                marker="x",
                label="Test spectra",
                color="C2",
                s=100,
            )
            plt.legend()
            plt.xlabel(parameter_names[i])
            plt.ylabel(parameter_names[j])

            save_figure("nested_" + parameter_names[i] + parameter_names[j])
            plt.close()
            plt.clf()
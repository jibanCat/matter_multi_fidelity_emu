import numpy as np
from matplotlib import pyplot as plt
import matplotlib

from matter_multi_fidelity_emu.gpemulator_singlebin import (
    SingleBinGP,
    SingleBinLinearGP,
    SingleBinNonLinearGP,
    SingleBinNonLinearMISO,
)
from matter_multi_fidelity_emu.data_loader import PowerSpecs

from .make_results import generate_data, validate_mf, validate_sf, do_emulator_error_plots, plot_parameters, do_pred_exact

# set a random number seed to reproducibility
np.random.seed(0)

matplotlib.use("pdf")

save_figure = lambda filename: plt.savefig(
    "{}.pdf".format(filename), format="pdf", dpi=300
)

def do_benchmark(
    folder: str = "data/50_LR_3_HR_box100",
    folder_sr: str = "data/50_LR_3_HR_box100_srgan",
    n_optimization_restarts: int = 30,
    n_fidelities: int = 2,
    n_IS: int = 2,
    turn_off_bias_nargp: bool = False,
    ARD_true: bool = False,
    n_samples: int = 100,
    figure_name: str = "ard_off_opt_30_samples_100", # additional figure name
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
    folder_sr: : the folder contains the the training and testing data. See
        data/50_LR_3_HR_srgan for example.
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
    data_sr = PowerSpecs(folder=folder_sr)
    # prepare MISO data. index 0 for true simulations
    # order: [HR, LR, SR]
    X_train_norm_miso = [
        data.X_train_norm[-1],
        data.X_train_norm[0],
        data_sr.X_train_norm[0],
    ]

    Y_train_norm_miso = [
        data.Y_train_norm[-1],
        data.Y_train_norm[0],
        data_sr.Y_train_norm[0],
    ]

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
    # multi-information source modelling
    miso_nargp = SingleBinNonLinearMISO(
        X_train_norm_miso,
        Y_train_norm_miso,
        n_IS=n_IS,
        n_samples=n_samples,
        optimization_restarts=n_optimization_restarts,
    )

    # Single-fidelity
    # high-fidelity only emulator
    hf_only = SingleBinGP(data.X_train_norm[-1], data.Y_train[-1])
    lf_only = SingleBinGP(data.X_train_norm[0], data.Y_train[0])
    sr_only = SingleBinGP(data_sr.X_train_norm[0], data_sr.Y_train[0])

    # optimize each model
    miso_nargp.optimize()
    ar1.optimize(n_optimization_restarts=n_optimization_restarts)
    nargp.optimize()
    hf_only.optimize_restarts(n_optimization_restarts=n_optimization_restarts)
    lf_only.optimize_restarts(n_optimization_restarts=n_optimization_restarts)
    sr_only.optimize_restarts(n_optimization_restarts=n_optimization_restarts)

    # testing set
    means_miso, vars_miso, pred_exacts_miso = validate_mf(data, model=miso_nargp)
    means_ar1, vars_ar1, pred_exacts_ar1 = validate_mf(data, model=ar1)
    means_nargp, vars_nargp, pred_exacts_nargp = validate_mf(data, model=nargp)
    means_hfonly, vars_hfonly, pred_exacts_hfonly = validate_sf(data, model=hf_only)
    means_lfonly, vars_lfonly, pred_exacts_lfonly = validate_sf(data, model=lf_only)
    means_sronly, vars_sronly, pred_exacts_sronly = validate_sf(data, model=sr_only)

    # versus HF
    do_emulator_error_plots(
        data,
        means_miso,
        means_hfonly,
        pred_exacts_miso,
        pred_exacts_hfonly,
        label_mf="MISO",
        label_sf="HF only",
        figure_name="miso" + figure_name,
    )

    do_emulator_error_plots(
        data,
        means_ar1,
        means_hfonly,
        pred_exacts_ar1,
        pred_exacts_hfonly,
        label_mf="AR1",
        label_sf="HF only",
        figure_name="ar1" + figure_name,
    )
    do_emulator_error_plots(
        data,
        means_nargp,
        means_hfonly,
        pred_exacts_nargp,
        pred_exacts_hfonly,
        label_mf="NARGP",
        label_sf="HF only",
        figure_name="nargp" + figure_name,
    )
    do_emulator_error_plots(
        data,
        means_sronly,
        means_hfonly,
        pred_exacts_sronly,
        pred_exacts_hfonly,
        label_mf="SR only",
        label_sf="HF only",
        figure_name="sronly" + figure_name,
    )
    # versus LF
    do_emulator_error_plots(
        data,
        means_miso,
        means_lfonly,
        pred_exacts_miso,
        pred_exacts_lfonly,
        label_mf="MISO",
        label_sf="LF only",
        figure_name="miso_lf" + figure_name,
    )
    do_emulator_error_plots(
        data,
        means_ar1,
        means_lfonly,
        pred_exacts_ar1,
        pred_exacts_lfonly,
        label_mf="AR1",
        label_sf="LF only",
        figure_name="ar1_lf" + figure_name,
    )
    do_emulator_error_plots(
        data,
        means_nargp,
        means_lfonly,
        pred_exacts_nargp,
        pred_exacts_lfonly,
        label_mf="NARGP",
        label_sf="LF only",
        figure_name="nargp_lf" + figure_name,
    )
    do_emulator_error_plots(
        data,
        means_sronly,
        means_lfonly,
        pred_exacts_sronly,
        pred_exacts_lfonly,
        label_mf="SR only",
        label_sf="LF only",
        figure_name="sronly_lf" + figure_name,
    )

    # pred/exact plot
    do_pred_exact(data, means_ar1, pred_exacts_ar1, label_mf="AR1", figure_name="ar1" + figure_name)
    do_pred_exact(
        data, means_nargp, pred_exacts_nargp, label_mf="NARGP", figure_name="nargp" + figure_name
    )
    do_pred_exact(
        data, means_miso, pred_exacts_miso, label_mf="MISO", figure_name="miso" + figure_name
    )
    do_pred_exact(
        data, means_sronly, pred_exacts_sronly, label="SR only", figure_name="sronly" + figure_name
    )

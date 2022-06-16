"""
Build on top of make_results.py, but saving all relevant
variables for future paper plots.
"""
import os
import json

from examples.make_results import *

# set a random number seed to reproducibility
np.random.seed(0)

matplotlib.use("pdf")


def generate_data(folder: str = "data/50_dmonly_3_fullphysics"):
    data = PowerSpecs(folder=folder)
    return data



def do_validations(
    folder: str = "data/50_dmonly_3_fullphysics",
    n_optimization_restarts: int = 20,
    n_fidelities: int = 2,
    turn_off_bias_nargp: bool = False,
    output_folder: str = "output/50_dmonly_3_fullphysics",
    ARD_last_fidelity: bool = False,
    parallel: bool = False,
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
    ARD_last_fidelity: whether to apply ARD for the last (highest) fidelity.
        Default, False.
    """
    # create output folder, recursively
    os.makedirs(output_folder, exist_ok=True)
    old_dir = os.getcwd()
    print("Current path:", old_dir)

    # get training and testing data. Normalization included.
    data = PowerSpecs(folder=folder)

    # change path for saving figures
    os.chdir(output_folder)
    print(">> ", os.getcwd())

    # Multi-fidelity
    # linear multi-fidelity
    ar1 = SingleBinLinearGP(
        data.X_train_norm,
        data.Y_train_norm,
        kernel_list=None,
        n_fidelities=n_fidelities,
        ARD_last_fidelity=ARD_last_fidelity,
    )
    # non-linear multi-fidelity
    nargp = SingleBinNonLinearGP(
        data.X_train_norm,
        data.Y_train_norm,
        n_fidelities=n_fidelities,
        n_samples=500,
        optimization_restarts=n_optimization_restarts,
        turn_off_bias=turn_off_bias_nargp,
        ARD_last_fidelity=ARD_last_fidelity,
    )

    # Single-fidelity
    # high-fidelity only emulator
    hf_only = SingleBinGP(data.X_train_norm[-1], data.Y_train[-1])
    lf_only = SingleBinGP(data.X_train_norm[0], data.Y_train[0])

    # optimize each model
    ar1.optimize(n_optimization_restarts=n_optimization_restarts, parallel=parallel)
    nargp.optimize(parallel=parallel)
    hf_only.optimize_restarts(n_optimization_restarts=n_optimization_restarts, parallel=parallel)
    lf_only.optimize_restarts(n_optimization_restarts=n_optimization_restarts, parallel=parallel)

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

    # saving hyperparameters
    with open("ar1.json", "w") as f:
        json.dump(ar1.to_dict(), f, indent=2)

    with open("nargp.json", "w") as f:
        json.dump(nargp.to_dict(), f, indent=2)

    with open("hf_only.json", "w") as f:
        json.dump(hf_only.to_dict(), f, indent=2)

    with open("lf_only.json", "w") as f:
        json.dump(lf_only.to_dict(), f, indent=2)

    # saving AR1
    os.makedirs("AR1/", exist_ok=True)

    np.savetxt(os.path.join("AR1", "all_gp_mean"), np.array(means_ar1))
    np.savetxt(os.path.join("AR1", "all_gp_var"), np.array(vars_ar1))
    np.savetxt(os.path.join("AR1", "pred_exacts"), np.array(pred_exacts_ar1))
    np.savetxt(os.path.join("AR1", "all_true"), np.array(data.Y_test[0]))
    np.savetxt(os.path.join("AR1", "kf"), np.array(data.kf))
    # [HF] also save the predictions from hf-only
    np.savetxt(os.path.join("AR1", "all_hf_gp_mean"), np.array(means_hfonly))
    np.savetxt(os.path.join("AR1", "all_hf_gp_var"), np.array(vars_hfonly))
    np.savetxt(os.path.join("AR1", "pred_exacts_hf"), np.array(pred_exacts_hfonly))
    # [LF] also save the predictions from lf-only
    np.savetxt(os.path.join("AR1", "all_lf_gp_mean"), np.array(means_lfonly))
    np.savetxt(os.path.join("AR1", "all_lf_gp_var"), np.array(vars_lfonly))
    np.savetxt(os.path.join("AR1", "pred_exacts_lf"), np.array(pred_exacts_lfonly))

    # saving NARGP
    os.makedirs("NARGP/", exist_ok=True)

    np.savetxt(os.path.join("NARGP", "all_gp_mean"), np.array(means_nargp))
    np.savetxt(os.path.join("NARGP", "all_gp_var"), np.array(vars_nargp))
    np.savetxt(os.path.join("NARGP", "pred_exacts"), np.array(pred_exacts_nargp))
    np.savetxt(os.path.join("NARGP", "all_true"), np.array(data.Y_test[0]))
    np.savetxt(os.path.join("NARGP", "kf"), np.array(data.kf))
    # [HF] also save the predictions from hf-only
    np.savetxt(os.path.join("NARGP", "all_hf_gp_mean"), np.array(means_hfonly))
    np.savetxt(os.path.join("NARGP", "all_hf_gp_var"), np.array(vars_hfonly))
    np.savetxt(os.path.join("NARGP", "pred_exacts_hf"), np.array(pred_exacts_hfonly))
    # [LF] also save the predictions from lf-only
    np.savetxt(os.path.join("NARGP", "all_lf_gp_mean"), np.array(means_lfonly))
    np.savetxt(os.path.join("NARGP", "all_lf_gp_var"), np.array(vars_lfonly))
    np.savetxt(os.path.join("NARGP", "pred_exacts_lf"), np.array(pred_exacts_lfonly))


    # back to root folder
    os.chdir(old_dir)

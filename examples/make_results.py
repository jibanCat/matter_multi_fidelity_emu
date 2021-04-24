"""
Train and test the models (AR1, NARGP, HF only, LF only)
"""

from matter_multi_fidelity_emu.gpemulator_singlebin import SingleBinGP, SingleBinLinearGP, SingleBinNonLinearGP
from matter_multi_fidelity_emu.data_loader import PowerSpecs

def generate_data(folder: str = "data/50_LR_3_HR"):
    data = PowerSpecs(folder=folder)
    return data


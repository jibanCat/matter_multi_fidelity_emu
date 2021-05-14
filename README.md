# A multi-fidelity emulator for the matter power spectrum from MP-Gadget

The code is derived from the multi-fidelity emulator code reported in

> M-F Ho, S Bird, and C Shelton. A Multi-Fidelity Emulator for 
> the Matter Power Spectrum using Gaussian Processes.
> [arXiv:2105.01081 [astro-ph.CO]](https://arxiv.org/abs/2105.01081),

including the matter power spectrum data (z=0) to reproduce the multi-fidelity trained with 50 low-fidelity simulations and 3 high-fidelity simulations.

Requirements:
- Python 3.6+
- numpy
- scipy
- GPy
- pyDOE
- emukit


Email me if there's any issues about versions: mho026-at-ucr.edu

## Reproduce the emulation results of the 50LR-3HR multi-fidelity emulator

```bash
python -c "from examples.make_results import *; do_benchmark('data/50_LR_3_HR', n_optimization_restarts=20)"
```

Results will be generated as figures after the above command finished.

## Where can I get the power spectrum data?

Simulations are run with [MP-Gadget code](https://github.com/MP-Gadget/MP-Gadget/).

A simulation submission file generator is here: github.com/jibanCat/SimulationRunnerDM.
I used this generator to prepare the training/testing data in this repo.

## Notebooks

Notebooks will be posted in `./notebooks/` folder.

- `Build 50LR-3HR Multi-Fidelity Emulator for Matter Power Spectrum.ipynb`: reproduce the results of `examples.make_results` code.

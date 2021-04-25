# A multi-fidelity emulator for the matter power spectrum from MP-Gadget

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

# rk-bfr-jump

A Bayesian framework for functional linear and logistic regression models, based on the theory of RKHS's, that leverages the capabilities of reversible jump MCMC samplers. An overview of the models and some experiments are available in Sections 2 and 4 of [this article](https://arxiv.org/abs/2312.14086).

The reversible jump samplers use an implementation from the [Eryn](https://mikekatz04.github.io/Eryn/html/index.html) library, with speed improvements thanks to parallelization with [numba](https://numba.readthedocs.io/en/stable/). There are very minor tweaks to Eryn's source code; see `Eryn_changes.md` for a summary of the changes.

## Code structure

- The folder `rkbfr_jump` contains the inference and prediction pipeline implemented. There is a `utils` folders inside with some utility files for simulation, experimentation and visualization.
- The folder `reference_methods` contains the implementation of some functional algorithms used for comparison.
- The `experiments` folder contains the numerical results from the experiments in [the accompanying article](https://arxiv.org/pdf/2312.14086).

The file `experiments.py` contains several experiments to test the performance of the models against other usual alternatives, functional or otherwise; a typical execution can be seen in the `launch.sh` file. Additionally, there are Jupyter notebooks that demonstrate the usage of the code.

*Code developed for Python 3.11 (see requirements.txt or environment.yml).*

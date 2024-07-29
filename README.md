# rk-bfr-jump

A Bayesian framework for functional linear and logistic regression models, built on the theory of RKHS's, that leverages the capabilities of reversible jump MCMC samplers. An overview of the models and some experiments are available in Sections 2 and 4 of [this accompanying article](https://arxiv.org/pdf/2312.14086).

The reversible jump samplers use an implementation from the [Eryn](https://github.com/mikekatz04/Eryn) library, with very minor tweaks to its source code. See `Eryn_changes.md` for a summary of the changes.

## Code structure

- The folder `rkbfr_jump` contains the inference and prediction pipeline implemented, using the [Eryn](https://emcee.readthedocs.io/) MCMC samplers and following the style of the [scikit-learn](https://scikit-learn.org/) and [scikit-fda](https://fda.readthedocs.io/) libraries. There is a `utils` folders inside with several utility files for simulation, experimentation and visualization.
- The folder `reference_methods` contains the implementation of some functional algorithms used for comparison.
- The `experiments` folder contains the numerical experimental results from the experiments in [the article](https://arxiv.org/pdf/2312.14086).

The file `experiments.py` contains several experiments to test the performance of the models against other usual alternatives, functional or otherwise; a typical execution can be seen in the `launch.sh` file. Additionally, there are Jupyter notebooks that demonstrate the usage of the code.

*Code developed for Python 3.11.*
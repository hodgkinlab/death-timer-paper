# Code for fitting the Ensemble and Threshold model to data

This repository is a companion to "Circumventing apoptosis exposes critical protein-threshold mechanisms in the regulation of lymphocyte death times." by Ruhle, et al 2023. It consists of the data presented in that paper and the code to reproduce fits to the Ensemble and Threshold (ET) model.

The code is based on python 3.10 but should work with 3.9 and later version. The specific modules used are documented in `requirements.txt` but, again, the code should not be dependent on the precise version of these modules.

To generate a figure:
```
python generate_figure.py --figures <figure> ...
```

To list available figures:
```
python generate_figure.py --help
```

To generate all figures:
```
python generate_figure.py --figures all
```
Fitting and generating all figures will take ~2 hours on a recent Mac laptop.

Figure output is placed in the `outputs` directory.

After fitting a model to data, the weights are cached in `outputs/weights-cache.json`. This is important because some weights are reused for several plots. This also provides a significant performance boost when rerunning, for example, when modifying plots. The weights for a particular run can be inspected here as well. Because of the stochastic nature of the fitting algorithm, weights will vary from run to run. The weights used in the paper are in the file `paper-weights.json`. To reproduce the exact plots presented in the paper, copy this file to `outputs/weights-cache.json`.




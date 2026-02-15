# Code for fitting the Ensemble and Threshold model to data

If you are looking for the code that is a companion to "A protein ensemble threshold model predicts lymphocyte death time" by Ruhle, et al 2023 use commit 9776bdb. [Click this link to read the bioXriv pre-print](https://www.biorxiv.org/content/10.1101/2023.10.25.563681). 

This repository consists of the data presented in that paper and the code to reproduce fits to the Ensemble and Threshold (ET) model. 

The code is based on python 3.10 but should work with 3.9 and later versions. The specific modules used are documented in `requirements.txt` but, again, the code should not be dependent on the precise version of these modules. Install as per any other python software, the recommended method would be to use conda or a virtual environment. Installation should take a couple of minutes.

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
Fitting and generating all figures will take ~2 hours on a recent Mac laptop. Figures will not match the layout in the paper.

Figure output is placed in the `outputs` directory.

After fitting a model to data, the weights are cached in `outputs/weights-cache.json`. This is important because some weights are reused for several plots. This also provides a significant performance boost when rerunning, for example, when modifying plots. The weights for a particular run can be inspected here as well. Because of the stochastic nature of the fitting algorithm, weights will vary from run to run. The weights used in the paper are in the file `paper-weights.json`. To reproduce the exact plots presented in the paper, copy this file to `outputs/weights-cache.json`.

Drug inhibition factors reported here, and used internally in the code, are (`1-ih`) for the inhibition factors reported in the paper.

Data for non-computational figures are in the Excel spreadsheet file `Data for additional figures.xlsx'`


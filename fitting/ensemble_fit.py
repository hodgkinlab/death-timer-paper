"""
Subclass of Fit for fitting ensemble models. Most of the effort goes into
- loading and preparing data
- managing the fit targets (i.e. cell numbers, cohort numbers, relative to WT or KO, etc)
- Making nice plots
"""
import math

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib
from FlowCal.plot import _LogicleScale
matplotlib.scale.register_scale(_LogicleScale)

from fitting.fit import Fit
from models.factory import model_for

class EnsembleFit(Fit):
  def __init__(self, parms):
    self.result = None
    self.experiment = parms.get('experiments', first=True)
    self.run_label = parms.get('label')
    self.normalise_method = parms.get('normalise_method')
    normalise_early_times = parms.get('normalise_early_times', True)
    self.include_wildtype = parms.get('include_wildtype', False)
    self.model_type = parms.get('model_type')
    self.fit_from_time = parms.get('fit_from_time')
    self.condition = parms.get('condition')
    self._error = self._survival_error if parms.get('fit_strategy') == 'survival' else self._histogram_error

    wt_protein_levels, ko_protein_levels = self.experiment.get_protein_levels(parms)

    survival_df = self.experiment.get_counts(survival_method=self.normalise_method, condition=self.condition)
    survival_df = survival_df[(survival_df['time'] >= self.fit_from_time) & (survival_df['time'] <= self.experiment.t_max_counts)]
    if normalise_early_times:
      survival_df.loc[survival_df['time']<self.fit_from_time, 'survival fraction'] = 1
    self.survival_df = survival_df
    survival = self.survival_df['survival fraction'].to_numpy()


    self.ko_protein_levels = ko_protein_levels
    self.wt_protein_levels = wt_protein_levels
    self.times = ko_protein_levels['time'].unique()

    # Extract and store counts in the protein level data (all protein counts are the same so use BIM)
    df = ko_protein_levels.groupby(['time'])['BIM'].count()
    df = df.to_frame()
    df.reset_index(inplace=True)
    df.rename(columns={'BIM': 'count'}, inplace=True)
    self.ko_protein_counts = df
    df = wt_protein_levels.groupby(['time'])['BIM'].count()
    df = df.to_frame()
    df.reset_index(inplace=True)
    df.rename(columns={'BIM': 'count'}, inplace=True)
    self.wt_protein_counts = df

    model = model_for(ko_protein_levels=ko_protein_levels,
                      wt_protein_levels=wt_protein_levels,
                      actual_survival=self.survival_df,
                      include_wildtype=self.include_wildtype,
                      parms=parms)

    self.model = model
    Fit.__init__(self, model, survival, parms)

  def predict(self, weights, early_threshold_adjustment=None):
    parms = []
    for p in self.proteins:
      if p in weights:
        parms.append(weights[p])
      if p+'^2' in weights:
        parms.append(weights[p+'^2'])
    if early_threshold_adjustment is not None:
      parms.append(early_threshold_adjustment)
    self.result = {'weights': parms}
    self.model.predict_and_store(parms, self.ko_protein_levels, 'KO')
    self.model.predict_and_store(parms, self.wt_protein_levels, 'WT')
    self.predicted_histograms = self.model.predict_histograms(parms, self.ko_protein_levels)
    return None

  def plot(self,
           ensemble_xlim=None,
           xscale_data=None,
           logicle=False,
           survival_only=False,
           fig=None,
           dashes=None,
           label_prefix=None):

    if survival_only:
      nrows, ncols = 1, 1
    else:
      ntimes = len(self.times)
      nplots = ntimes + 1
      nrows = math.ceil(nplots/2)
      ncols = 2

    if fig is None:
      fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6.2, 3*nrows))
      ax = np.ndarray.flatten(ax) if type(ax) == np.ndarray else [ax]
    else:
      ax = fig.axes

    self.model.plot_survival(ax=ax[0], dashes=dashes, label_prefix=label_prefix)
    if survival_only:
      return fig

    ko = self.model.ko_protein_levels.copy()
    ko['ensemble'] = self.model.ensemble(ko, self.result['weights'])
    ko.loc[:, 'genotype'] = 'KO'
    wt = self.model.wt_protein_levels.copy()
    wt['ensemble'] = self.model.ensemble(wt, self.result['weights'])
    wt.loc[:, 'genotype'] = 'WT'
    data = pd.concat([ko, wt])

    if ensemble_xlim is not None:
      data = data[(ensemble_xlim[0]<data['ensemble']) & (data['ensemble']<ensemble_xlim[1])]

    survival = self.survival_frame()
    measured  = survival[survival['source'] == 'data']['death']
    predicted = survival[survival['source'].str.startswith('KO')]['death']
    legend = True
    ymax = 0
    xmin = 1e6
    xmax = 0
    self.times.sort()
    for t, i in zip(self.times, range(1, nplots)):
      dt = data[data['time'] == t]
      self.model.plot_ensemble(ax=ax[i], data=dt, legend=legend)
      legend = False
      if i != (nrows-1):
        ax[i].set_xlabel(None)
      ax[i].set_ylabel(f'{t}h')
      ymax = max(ax[i].get_ylim()[1], ymax)
      xmax = max(ax[i].get_xlim()[1], xmax)
      xmin = min(ax[i].get_xlim()[0], xmin)

    if xscale_data is None and logicle:
      xscale_data = data['ensemble'].to_numpy() # used to determine the Logicle plot scaling

    for i, m, p in zip(range(1, nplots), measured, predicted):
      if logicle:
        ax[i].set_xscale('logicle', data=xscale_data)
      ax[i].set_ylim([0, ymax])
      if ensemble_xlim is not None:
        ax[i].set_xlim(ensemble_xlim)
      ax[i].text(-50, ymax/1.8, f'data: {round(m*100)}%')
      ax[i].text(-50, ymax/2.2, f'fit:  {round(p*100)}%')

    return fig

  def _survival_error(self, parms, x_truth, *args):
    error = Fit.survival_error(self, parms, x_truth, self.ko_protein_levels)
    if self.include_wildtype:
      error += Fit.survival_error(self, parms, [1] * len(x_truth), self.wt_protein_levels)
    return error

  def _histogram_error(self, parms, x_truth, *args):
    error = Fit.histogram_error(self, parms, x_truth, self.ko_protein_levels)
    return error

  def error(self, parms, x_truth, *args):
    return self._error(parms, x_truth, *args)
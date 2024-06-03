"""
The ensemble model is the abstract class of models where an ensemble histogram is calculated from
protein level histograms. Cells in the ensemble histogram below a threshold die. The exact nature
of the ensemble calculation is left to the implementing class. This class provides plots showing
the ensemble histogram and fit vs data.
"""
from abc import abstractmethod

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from models.model import Model

class EnsembleModel(Model):
  name = None
  THRESHOLD = 10

  @staticmethod
  def apply_ensemble(df, parms, weight_dict):
    # This is a utility method to add the ensemble
    # into the dataframe.
    data_cols = parms.get('proteins')
    e = 0
    for p in data_cols:
      d = df[p].to_numpy()
      p = weight_dict[p]
      e += p*d
    df['ensemble'] = e
    return df


  def __init__(self,
               ko_protein_levels=None,
               wt_protein_levels=None,
               include_wildtype=False,
               actual_survival=None,
               threshold_step=0,
               parms=None):
    self.wt_protein_levels = wt_protein_levels
    self.ko_protein_levels = ko_protein_levels
    self.threshold_step = threshold_step
    self.actual_survival = actual_survival
    self.include_wildtype = include_wildtype

    fixed_protein_weights = parms.get('fixed_protein_weights')
    self.fixed_protein_weights = {} if fixed_protein_weights is None else fixed_protein_weights


    Model.__init__(self)

    self.data_cols = parms.get('proteins')
    self.all_cells = None
    self.predicted_survival = None
    self.predicted_wt_survival = None
    self.x_pred = None
    self.ax = None

  def data(self, df):
    return [df[p].to_numpy() for p in self.data_cols]

  def fixed_data(self, df):
    fd = 0
    for protein, weight in self.fixed_protein_weights.items():
      fd += df[protein].to_numpy()*weight
    return fd


  @abstractmethod
  def ensemble(self, row, parms):
    raise NotImplementedError


  def predict_and_store(self, parms, target, genotype):
    sr, _ = self._predict(parms, target)
    if genotype == 'KO':
      self.predicted_survival = sr
    else:
      self.predicted_wt_survival = sr

  def _predict(self, parms, target):
    all_cells = target.copy()

    if self.threshold_step > 0:
      ensemble_parms = parms[0:-1]
      threshold_parm = parms[-1]
    else:
      ensemble_parms = parms
      threshold_parm = 1

    all_cells.loc[:, 'ensemble'] = self.ensemble(all_cells, ensemble_parms).copy()
    all_cell_counts = all_cells.groupby('time').count()

    early_cells = all_cells[all_cells['time'] < self.threshold_step]
    early_living_cells = early_cells[early_cells['ensemble']>self.THRESHOLD*threshold_parm]

    late_cells = all_cells[all_cells['time'] >= self.threshold_step]
    late_living_cells = late_cells[late_cells['ensemble']>self.THRESHOLD]

    living_cells = pd.concat([late_living_cells, early_living_cells])
    living_cell_counts = living_cells.groupby('time').count()
    sr = living_cell_counts.div(all_cell_counts).fillna(0)

    self.all_cells = all_cells

    return sr, living_cells

  def predict_survival(self, parms, target):
    sr, _ = self._predict(parms, target)
    return sr['ensemble'].to_numpy()

  def predict_histograms(self, parms, target):
    _, living_cells = self._predict(parms, target)
    return living_cells

  def plot_ensemble_by_genotype(self, ax, data=None, x='ensemble', legend=True):
    data = data[data[x]>0]
    ax = sns.histplot(data=data, x=x, hue='time', legend=legend, kde=True, log_scale=True, ax=ax)
    ax.axvline(self.THRESHOLD, color='red', linestyle='dashed')
    return ax

  def _get_palette(self, sources):
    palette = {}
    for source in sources:
      if 'data' in source:
        palette[source] = '#0C0C0C'
      if 'KO' in source:
        palette[source] = '#E20613'
      if 'WT' in source:
        palette[source] = '#B2B2B2'

    return palette

  def plot_ensemble(self, ax, data, legend=True):
    palette = self._get_palette(['KO', 'WT'])
    ax = sns.histplot(data=data,
                      x='ensemble',
                      hue='genotype',
                      legend=legend,
                      kde=True,
                      common_norm=False,
                      stat='density',
                      palette=palette,
                      ax=ax)
    ax.axvline(self.THRESHOLD, color='red', linestyle='dashed')

    return ax

  def wt_error(self, parms):
    survival, _ = self._predict(parms, self.wt_protein_levels)
    survival = survival['ensemble'].to_numpy()
    wt_truth = np.ones(len(survival))
    wt_error = np.linalg.norm(wt_truth - survival)
    return wt_error

  def ko_error(self, parms):
    predicted_survival, _ = self._predict(parms, self.ko_protein_levels)
    predicted_survival = predicted_survival['ensemble'].to_numpy()
    actual_survival = self.actual_survival['survival fraction'].to_numpy()
    ko_error = np.linalg.norm(actual_survival - predicted_survival)
    return ko_error

  def plot_survival(self, ax=None, dashes=None, label_prefix=None):
    if dashes is None:
      dashes = [(1, 0), (2, 2), (2, 2)]
    survival = self.survival_frame(label_prefix=label_prefix)
    sources = list(survival['source'].unique())
    palette = self._get_palette(sources)
    ax = sns.lineplot(data=survival,
                      x='time',
                      y='death',
                      hue='source',
                      style='source',
                      markers=True,
                      palette=palette,
                      dashes=dashes,
                      ax=ax)
    ax.set_ylim(0, 1)

    return ax

  def survival_frame(self, add_errors=True, label_prefix=None):
    survival = self.actual_survival.copy()
    cols = survival.columns.to_list()
    cols.remove('survival fraction')
    cols.remove('time')
    survival.drop(cols, axis='columns', inplace=True)
    data_label = 'data'
    survival.rename({'survival fraction': data_label, 'index': 'time'}, axis='columns', inplace=True)

    ko_values = self.predicted_survival['ensemble'].values
    ko_error = np.linalg.norm(ko_values - survival[data_label])
    ko_label = f'KO ({ko_error:.2f})' if add_errors else 'KO'
    ko_label = label_prefix + ' ' + ko_label if label_prefix is not None else ko_label
    survival[ko_label] = ko_values

    wt_values = self.predicted_wt_survival['ensemble'].values
    wt_truth = np.ones(len(survival[data_label]))
    wt_error = np.linalg.norm(wt_truth - wt_values)
    wt_label = f'WT ({wt_error:.2f})' if add_errors else 'WT'
    wt_label = label_prefix + ' ' + wt_label if label_prefix is not None else wt_label
    survival[wt_label] = self.predicted_wt_survival['ensemble'].values
    survival = survival.melt('time', var_name='source', value_name='death')
    survival['death'] = 1 - survival['death']

    return survival
  def plot(self, _):
    fig, ax = plt.subplots(nrows=2, constrained_layout=True)
    self.plot_ensemble(ax[1])
    self.plot_survival(ax[0])
    fig.suptitle(f'In progress fit for {self.label} - {self.name} model')
    return fig, ax

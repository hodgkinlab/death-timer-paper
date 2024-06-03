"""
The Experiment class provides additional parameters for experiments that cannot
be obtained from the CSVs or path metadata
"""
from abc import ABC, abstractmethod

import numpy as np

from data_loading.levels import BProteinLevels, TProteinLevels
from data_loading.counts import Bcounts, Tcounts
from data_loading.loader import PROTEINS


class Experiment(ABC):
  def __init__(self, name, t_max_protein, t_max_counts, t_min_protein, t_min_counts, default_condition,
               available_conditions, directory, survival_parameters=None, ensemble_xlim_hint=None, histogram_xlim_hint=None):
    self.name = name
    self.t_max_protein = t_max_protein
    self.t_max_counts  = t_max_counts
    self.t_min_protein = t_min_protein
    self.t_min_counts  = t_min_counts
    self._default_condition = default_condition
    self.survival_parameters = survival_parameters
    self._available_conditions = {self.default_condition(): PROTEINS} if available_conditions is None else available_conditions
    self._directory = name if directory is None else directory
    self._ensemble_xlim_hint = ensemble_xlim_hint if ensemble_xlim_hint else [-100, 300]
    self._histogram_xlim_hint = histogram_xlim_hint

  def set_default_condition(self, condition):
    self._default_condition = condition

  def default_condition(self):
    return self._default_condition

  def directory(self):
    return self._directory

  def histogram_xlim_hint(self):
    return self._histogram_xlim_hint

  def ensemble_xlim_hint(self):
    return self._ensemble_xlim_hint

  def __str__(self):
    return self.name

  def __repr__(self):
    return self.name

  def to_json(self):
    return self.name

  @abstractmethod
  def get_protein_levels(self):
    NotImplementedError('Need to implement get_protein_levels()')

  def available_conditions(self):
    return self._available_conditions

class BcellExperiment(Experiment):
  def __init__(self, name, t_max_protein, t_max_counts, t_min_protein=72, t_min_counts=72, default_condition='-',
               available_conditions=None, directory=None, survival_parameters=None, ensemble_xlim_hint=None, histogram_xlim_hint=None):
    super().__init__(name, t_max_protein, t_max_counts, t_min_protein, t_min_counts, default_condition, available_conditions, directory,
                     survival_parameters=survival_parameters, ensemble_xlim_hint=ensemble_xlim_hint, histogram_xlim_hint=histogram_xlim_hint)
  def get_protein_levels(self, parms):
    fit_from_time = parms.get('fit_from_time')
    do = BProteinLevels(parms)
    condition = parms.get('condition')
    if condition is None:
      condition = self.default_condition()
    all_protein_levels = do.df
    ko_protein_levels = all_protein_levels[(all_protein_levels['genotype'] == 'BaxBak KO') &
                                           (all_protein_levels['condition'] == condition) &
                                           (all_protein_levels['time'] >= fit_from_time) &
                                           (all_protein_levels['time'] <= self.t_max_protein)]
    wt_protein_levels = all_protein_levels[(all_protein_levels['genotype'] == 'WT') &
                                           (all_protein_levels['condition'] == condition) &
                                           (all_protein_levels['time'] >= fit_from_time) &
                                           (all_protein_levels['time'] <= self.t_max_protein)]

    self.protein_data_object = do
    return wt_protein_levels, ko_protein_levels

  def get_counts(self, survival_method=None, condition=None):
    condition = self.default_condition() if condition is None else condition
    return Bcounts(self).survival(survival_method=survival_method, condition=condition)

class TcellExperiment(Experiment):
  def __init__(self, name, t_max_protein, t_max_counts, t_min_protein=96, t_min_counts=96,
               default_condition='-', directory=None, ensemble_xlim_hint=None):
    super().__init__(name, t_max_protein, t_max_counts, t_min_protein, t_min_counts,
                     default_condition, None, directory, ensemble_xlim_hint=ensemble_xlim_hint)


  def get_protein_levels(self, parms):
    do = TProteinLevels(parms)
    all_protein_levels = do.df
    ko_protein_levels = all_protein_levels[(all_protein_levels['genotype'] == 'BaxBak KO') &
                                           (all_protein_levels['time'] >= 96) &
                                           (all_protein_levels['time'] <= self.t_max_protein)]
    wt_protein_levels = all_protein_levels[(all_protein_levels['genotype'] == 'WT') &
                                           (all_protein_levels['time'] >= 96) &
                                           (all_protein_levels['time'] <= self.t_max_protein)]
    return wt_protein_levels, ko_protein_levels

  def get_counts(self, survival_method, **_):
    return Tcounts(self).survival(survival_method=survival_method, condition=self.default_condition())


class CombinedExperiment:
  def __init__(self, experiments):
    self.experiments = experiments
    self.name = ''.join("%s" % ','.join(map(str, x.name)) for x in experiments)


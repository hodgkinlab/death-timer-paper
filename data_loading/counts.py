
from abc import ABC, abstractmethod
import numpy as np
from scipy.stats import lognorm

from data_loading.loader import load_counts

NORMALISE_METHODS = ['WT self counts', 'KO counts', 'WT self cohort number', 'KO cohort number', 'fitted survival', 'PI stain']

def _survival(x, mean, std):
  return 1 - lognorm.cdf(x, std, 0, np.exp(mean))

class Counts(ABC):

  def __init__(self, experiment, keep_replicates=False):
    df = load_counts(experiment)

    if keep_replicates:
      self.df = df
    else:
      self.df = df.groupby(['experiment', 'condition', 'genotype', 'time']).sum()
      self.df.drop(columns='replicate', inplace=True)

    self.df.reset_index(inplace=True)
    self.gb = ['experiment', 'condition', 'time']
    self.experiment = experiment
    self._default_condition = experiment.default_condition()

    self.survival_parameters = experiment.survival_parameters

  def available_conditions(self):
    return list(self.experiment.available_conditions().keys()) + [self._default_condition]

  @staticmethod
  @abstractmethod
  def filter_survival(sr, exp):
    NotImplementedError('wt_condition is not implemented')

  def survival(self, survival_method=None, condition=None):
    assert (survival_method in NORMALISE_METHODS), f'{survival_method} is not a valid survival method'

    if survival_method == 'PI stain':
      df = self.df
      sr = df[(df['genotype'] == 'WT') & (df['condition'] == condition)].copy()
      sr = sr.groupby(['condition'], group_keys=False).apply(self._normalise_by_max_value_, kwds={'measure': 'PI stain'}).reset_index(drop=True)
      sr.rename(columns={'PI stain': 'survival fraction'}, inplace=True)
      return sr

    if survival_method == 'fitted survival':
      df = self.df
      sr = df[(df['genotype'] == 'WT') & (df['condition'] == condition)].copy()
      survival_parameters = self.survival_parameters[condition]
      survival_mean = survival_parameters['survival_mean']
      survival_std  = survival_parameters['survival_std']
      survival = _survival(sr['time'], survival_mean, survival_std)
      sr['WT counts'] = survival
      sr['survival fraction'] = survival
      return sr

    if survival_method in ['WT self counts', 'WT self cohort number']:
      relative_to = 'WT'
    else:
      relative_to = 'KO'

    if survival_method in ['WT self cohort number', 'KO cohort number']:
      wt_measure = 'WT self cohort number'
      ko_measure = 'KO cohort number'
    else:
      wt_measure = 'WT counts'
      ko_measure = 'KO counts'

    if condition is None:
      df = self.df
    else:
      df = self.df[self.df['condition'] == condition]

    ko = 'BaxBak KO'
    wt = 'WT'

    wt = df[df['genotype']==wt]
    wt = wt.drop(columns='genotype')
    wt.rename(columns={'cohort number': 'WT self cohort number', 'counts': 'WT counts'}, inplace=True)
    wt.reset_index(drop=True, inplace=True)

    if relative_to == 'KO':
      wt.set_index(self.gb, inplace=True)
      ko = df[df['genotype']==ko]
      ko = ko.drop(columns='genotype')
      ko.rename(columns={'cohort number': 'KO cohort number', 'counts': 'KO counts'}, inplace=True)
      ko.set_index(self.gb, inplace=True)

      # Experiments can start with different cell numbers between WT and KO.
      # This normalises WT to KO counts so that survival is 1 at t=0.
      wt0 = wt[wt_measure][0]
      ko0 = ko[ko_measure][0]
      if wt0 == 0 and ko0 == 0:
        r = 1
      else:
        r = ko0/wt0
      wt[wt_measure] *= r
      wt['survival fraction'] = wt[wt_measure] / ko[ko_measure]

      # Add the raw KO numbers into the data frame.
      sr = wt
      sr['KO counts'] = ko['KO counts']
      sr.reset_index(inplace=True)
    elif wt_measure == 'WT self cohort number':
      g = ['condition']
      if 'experiment' in wt.columns:
        g.append('experiment')
      sr = wt.groupby(g, group_keys=False).apply(self._normalise_by_start_value_).reset_index(drop=True)
      sr.rename(columns={wt_measure: 'survival fraction'}, inplace=True)
    else:
      raise NotImplementedError('Normalisation method')

    return sr


class Bcounts(Counts):
  def __init__(self, experiment, keep_replicates=False):
    super().__init__(experiment, keep_replicates=keep_replicates)

  def default_condition(self):
    return self._default_condition

  def _normalise_by_start_value_(self, dff):
    for k in ['WT counts', 'WT self cohort number']:
      start_counts = dff[dff['time'] == 0][k].iloc[0]
      dff[k] = dff[k].map(lambda x: x / start_counts)
    return dff

  def _normalise_by_max_value_(self, dff):
    for k in self.dt:
      max_counts = dff[k].max()
      dff[k] = dff[k].map(lambda x: x / max_counts)
    return dff

  def from_experiment(self, experiment):
    return self.df[self.df['experiment'] == experiment]

  @staticmethod
  def filter_survival(sr, exp):
    return sr[sr['experiment']==exp]

  def condition_name(self):
    if self.df['condition'].unique().all() is None:
      return 'genotype'
    else:
      return ['condition', 'genotype']

  @staticmethod
  def ko():
    return 'BaxBak KO'

  @staticmethod
  def wt():
    return 'WT'

class Tcounts(Counts):

  def __init__(self, experiment):
    super().__init__(experiment)

  def default_condition(self):
    return '96h wash'

  def from_experiment(self, _):
    return self.df

  def _normalise_by_max_value_(self, df, kwds={'measure': 'counts'}):
    measure = kwds['measure']
    df[measure] = df[measure]/df[measure].max()
    return df

  @staticmethod
  def filter_survival(sr, exp):
    return sr



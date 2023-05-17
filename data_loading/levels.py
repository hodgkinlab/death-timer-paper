import pickle

from scipy import stats
import pandas as pd

from data_loading.loader import KEEP_CHANNELS, PROTEINS, load_protein_levels, GENOTYPES, load_T_protein_levels


def _T_normalise(df, tm):
  df72 = df[(df['time'] == tm) & (df['genotype'] == 'BaxBak KO')]
  for protein in PROTEINS:
    n = stats.gmean(df72[protein])
    df[protein] = df[protein] / n
  return df

def _meano(x):
  rf = pd.DataFrame()
  for p in PROTEINS:
    rf[p] = [stats.gmean(x[p])]
  return rf

def _gstdo(x):
  rf = pd.DataFrame()
  for p in PROTEINS:
    rf[p] = [stats.gstd(x[p])]
  return rf

def _volume_correct(df):
  for p in PROTEINS:
    df[p] = df[p]/(df['FSC-W']/60000)**3
  return df

def _normalise(df):
  df72 = df[(df['time'] == 72) & (df['genotype'] == 'BaxBak KO')]
  dfm = df72.groupby(['experiment']).apply(_meano).reset_index()
  dfn = df
  experiments = dfm['experiment'].unique()
  for experiment in experiments:
    for protein in PROTEINS:
      n = dfm[dfm['experiment']==experiment][protein].iloc[0]
      dfn.loc[dfn['experiment']==experiment, protein] = dfn.loc[dfn['experiment']==experiment, protein] / n
  return dfn

def _bim_cleanup(df, parms):
  df['BIM'] = df['BIM'] - parms.get('BIM_noise_floor')

def _clean_from_negative(df):
  # Replace -ve values with 1. Should not be a significant issue for recent experiments
  for p in ['BCL2', 'BCLxL', 'MCL1', 'BIM']:
    df.loc[df[p] <= 0, p] = 1

class BProteinLevels:
  def __init__(self, parms):
    experiment = parms.get('experiments')

    self.experiment = experiment

    df = load_protein_levels([experiment], 'B cells')

    # Replace -ve values with 1. Should not be a significant issue for recent experiments
    _clean_from_negative(df)

    df = _normalise(df)

    df = df.drop(columns=KEEP_CHANNELS)

    self.df = df

  def from_experiment(self, experiment):
    return self.df[self.df['experiment'] == experiment]

  def gstds(self):
    return self.df.groupby(['experiment', 'genotype', 'time']).apply(_gstdo).reset_index().drop(columns='level_3')

  @staticmethod
  def group():
    return ['experiment', 'time', 'genotype', 'condition']

  @staticmethod
  def genotypes():
    return GENOTYPES

  def default_condition(self):
    return self.experiment.default_condition()

  @staticmethod
  def filter_for_plot(df, cond):
    return df[df['condition'] == cond]

class TProteinLevels:
  def __init__(self, parms):
    self.experiment = parms.get('experiments')

    gb = ['genotype', 'condition', 'time', 'replicate']

    df = load_T_protein_levels(self.experiment.directory(), conditions=self.experiment.default_condition())
    df['experiment'] = self.experiment.name

    _clean_from_negative(df)

    df = _T_normalise(df, 96)

    df = df.drop(columns=KEEP_CHANNELS)

    self.df = df
    # self.means = self.df.groupby(gb).mean().reset_index()

  @staticmethod
  def group():
    return ['time', 'genotype', 'condition']

  def plot_groups(self):
    return [self.experiment.name]

  @staticmethod
  def filter_for_plot(df, _):
    return df

  def default_condition(self):
    return self.experiment.default_condition()


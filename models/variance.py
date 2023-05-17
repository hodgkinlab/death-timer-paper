import numpy as np
import pandas as pd
import seaborn as sns

from models.model import Model, ModelFactory
from data_loading.loader import PROTEINS


class VarianceModel(Model):
  name = 'gstd mode'

  def __init__(self, **kwargs):
    self.variances = kwargs.pop('variances')
    Model.__init__(self, **kwargs)

  def _pred(self, row, *args):
    (parms,) = args
    data = [row[p] for p in PROTEINS]
    return np.dot(parms, data)

  def predict_survival(self, parms: np.ndarray, *args) -> float:
    pred_df = self.variances.apply(self._pred, axis='columns', args=(parms,))
    p = pred_df.to_numpy()
    self.predicted_survival = p
    return p

  def plot_survival(self, survival, ax=None):
    actual = pd.DataFrame()
    actual['time'] = survival['time']
    actual['death'] = 1 - survival['fraction']
    actual['source'] = 'data'
    predicted = pd.DataFrame()
    predicted['time'] = survival['time']
    predicted['death'] = 1 - self.predicted_survival
    predicted['source'] = 'fit'
    plotto = pd.concat([actual, predicted]).reset_index()
    ax = sns.lineplot(data=plotto,
                      x='time',
                      y='death',
                      hue='source',
                      style='source',
                      markers=True,
                      ax=ax)
    ax.set_ylim(0, 1)
    return ax


class VarianceModelFactory(ModelFactory):
  def model(self,
            variances,
            actual_survival,
            fit_target,
            verbose=False,
            fitting_algorith='differential evolution',
            data_cols=PROTEINS,
            plot_progress=False):
    return VarianceModel(variances,
                         actual_survival,
                         fitting_algorithm=fitting_algorith,
                         verbose=verbose)
from data_loading.loader import PROTEINS
from models.ensemble import EnsembleModel
from models.model import ModelFactory

class LogSubtractionModel(EnsembleModel):
  name = 'log-subtraction'

  def data(self, df):
    return [df[f'log({p})'].to_numpy() for p in PROTEINS]

  def ensemble(self, df, parms):
    data = self.data(df)
    e = 0
    for p, d in zip(data, parms):
      e += p*d
    return e


class LogSubtractionModelFactory(ModelFactory):
  def model(self,
            ko_protein_levels=None,
            wt_protein_levels=None,
            actual_survival=None,
            fit_target=None,
            fitting_algorithm='differential evolution',
            verbose=False,
            plot_progress=False):
    return LogSubtractionModel(ko_protein_levels=ko_protein_levels,
                       wt_protein_levels=wt_protein_levels,
                       actual_survival=actual_survival,
                       fit_target=fit_target,
                       fitting_algorithm=fitting_algorithm,
                       plot_progress=plot_progress,
                       verbose=verbose)

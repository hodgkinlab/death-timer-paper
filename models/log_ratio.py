from models.ensemble import EnsembleModel
from models.model import ModelFactory

class LogRatioModel(EnsembleModel):
  name = 'log-ratio'

  def data(self, df):
    return [df[f'log({p})'].to_numpy() for p in self.data_cols]

  def ensemble(self, df, parms):
    data = self.data(df)
    return (parms[0]*data[0] + parms[1]*data[1] + parms[2]*data[2]) / data[3]


class LogRatioModelFactory(ModelFactory):
  def model(self,
            ko_protein_levels=None,
            wt_protein_levels=None,
            actual_survival=None,
            fit_target=None,
            fitting_algorithm='differential evolution',
            verbose=False,
            plot_progress=False):
    return LogRatioModel(ko_protein_levels=ko_protein_levels,
                       wt_protein_levels=wt_protein_levels,
                       actual_survival=actual_survival,
                       fit_target=fit_target,
                       fitting_algorithm=fitting_algorithm,
                       plot_progress=plot_progress,
                       verbose=verbose)
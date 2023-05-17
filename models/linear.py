"""
An ensemble model in which the ensemble is the weighted sum of the protein levels.
BIM is handled in the same as other proteins but the parameter bound is used to
ensure it is -ve and hence subtracted.
"""
from models.ensemble import EnsembleModel
from models.model import ModelFactory


class LinearModel(EnsembleModel):
  name = 'linear'

  def ensemble(self, df, parms):
    e = self.fixed_data(df)
    data = self.data(df)
    for d, p in zip(data, parms):
      e += p*d
    return e

class LinearModelFactory(ModelFactory):
  def model(self,
            ko_protein_levels=None,
            wt_protein_levels=None,
            include_wildtype=False,
            actual_survival=None,
            threshold_step=0,
            parms=None):
    return LinearModel(ko_protein_levels=ko_protein_levels,
                       wt_protein_levels=wt_protein_levels,
                       actual_survival=actual_survival,
                       include_wildtype=include_wildtype,
                       threshold_step=threshold_step,
                       parms=parms)

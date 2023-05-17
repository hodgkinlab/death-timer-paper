"""
An ensemble model in which the ensemble is the weighted sum of the protein levels.
BIM is handled in the same as other proteins but the parameter bound is used to
ensure it is -ve and hence subtracted.
"""
from models.ensemble import EnsembleModel
from models.model import ModelFactory


class QuadraticModel(EnsembleModel):
  name = 'quadratic'

  def ensemble(self, df, parms):
    e = self.fixed_data(df)
    data = self.data(df)
    for i in range(len(data)):
      d = data[i]
      p = parms[2*i]
      e += p*d
      p = parms[2*i+1]
      e += p*d
    return e

class QuadraticModelFactory(ModelFactory):
  def model(self,
            ko_protein_levels=None,
            wt_protein_levels=None,
            include_wildtype=False,
            actual_survival=None,
            threshold_step=0,
            parms=None):
    return QuadraticModel(ko_protein_levels=ko_protein_levels,
                       wt_protein_levels=wt_protein_levels,
                       actual_survival=actual_survival,
                       include_wildtype=include_wildtype,
                       threshold_step=threshold_step,
                       parms=parms)

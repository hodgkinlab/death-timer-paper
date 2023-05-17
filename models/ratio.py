from data_loading.loader import PROTEINS
from models.ensemble import EnsembleModel
from models.model import ModelFactory

class RatioModel(EnsembleModel):
  name = 'ratio'

  def ensemble(self, df, parms):
    e = 0
    for i, d in enumerate(['BCL2', 'BCLxL', 'MCL1']):
        e += parms[i]*df[d].to_numpy()
    return e / df['BIM'].to_numpy()


class RatioModelFactory(ModelFactory):
  def model(self,
            ko_protein_levels=None,
            wt_protein_levels=None,
            include_wildtype=False,
            actual_survival=None,
            threshold_step=0,
            parms=None):
    return RatioModel(ko_protein_levels=ko_protein_levels,
                       wt_protein_levels=wt_protein_levels,
                       actual_survival=actual_survival,
                       include_wildtype=include_wildtype,
                       threshold_step=threshold_step,
                       parms=parms)


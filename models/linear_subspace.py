"""
The
"""
from data_loading.loader import PROTEINS
from models.ensemble import EnsembleModel
from models.model import ModelFactory


class LinearSubspaceModel(EnsembleModel):
  name = 'linear-subspace'

  def __init__(self, **kwargs):
    constant_proteins = kwargs.pop('constant_proteins')
    constant_protein_values = kwargs.pop('constant_protein_values')
    assert set(constant_proteins).issubset(PROTEINS), 'Invalid constant protein'
    assert len(constant_proteins)==len(constant_protein_values), 'Constant protein list and values don\'t not match'

    self.constant_protein_indexes = [PROTEINS.index(p) for p in constant_proteins]
    self.constant_protein_values = constant_protein_values.copy()
    self.variable_proteins = []
    for p in PROTEINS:
      if p not in constant_proteins:
        self.variable_proteins.append(p)


    EnsembleModel.__init__(self, **kwargs)

    if 'BIM' in constant_proteins:
      idx = constant_proteins.index('BIM')
      self.constant_protein_values[idx] = -constant_protein_values[idx]

  def ensemble(self, row, parms):
    data = self.data(row)
    e = 0
    for cpv, cpi in zip(self.constant_protein_values, self.constant_protein_indexes):
      e += cpv*data[cpi]

    d2 = []
    for i, d in enumerate(data):
      if i not in self.constant_protein_indexes:
        d2.append(d)

    for d, p in zip(d2, parms):
      e += p * d

    return e

  def format_result(self, result):
    weights = {}
    for d, p in zip(self.variable_proteins, result.x):
      weights[d] = p
    return {'error': result.fun, 'weights': weights}

class LinearSubspaceModelFactory(ModelFactory):
  def model(self,
            ko_protein_levels=None,
            wt_protein_levels=None,
            actual_survival=None,
            fitting_algorithm='differential evolution',
            verbose=False,
            constant_proteins=None,
            constant_protein_values=None,
            plot_progress=False):
    return LinearSubspaceModel(ko_protein_levels=ko_protein_levels,
                       wt_protein_levels=wt_protein_levels,
                       actual_survival=actual_survival,
                       fitting_algorithm=fitting_algorithm,
                       plot_progress=plot_progress,
                       constant_proteins=constant_proteins,
                       constant_protein_values=constant_protein_values,
                       verbose=verbose)
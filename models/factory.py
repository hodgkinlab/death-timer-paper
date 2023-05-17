"""
A factory for constructing models
"""
from models.combined import CombinedModelFactory
from models.linear import LinearModelFactory
from models.linear_subspace import LinearSubspaceModelFactory
from models.log_ratio import LogRatioModelFactory
from models.log_subtraction import LogSubtractionModelFactory
from models.quadratic import QuadraticModelFactory
from models.ratio import RatioModelFactory
from models.variance import VarianceModelFactory


def model_for(**model_parms):
  model_type = model_parms['parms'].get('model_type')
  model_factory = __registry__[model_type]
  return model_factory.model(**model_parms)

__registry__ = {
  'log-subtraction': LogSubtractionModelFactory(),
  'log-ratio': LogRatioModelFactory(),
  'linear': LinearModelFactory(),
  'quadratic': QuadraticModelFactory(),
  'ratio': RatioModelFactory(),
  'variance': VarianceModelFactory(),
  'linear-subspace': LinearSubspaceModelFactory(),
  'combined-model': CombinedModelFactory()
}

def model_type_validator(model_type):
  return model_type in __registry__

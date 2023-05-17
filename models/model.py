"""
This is the abstract base class for a survival model. A model can
- fit to data with a couple of optimisation algorithms
- return parameters
- make predictions

Models should also consider providing a factory and get registered (see factory.py)
"""
from abc import ABC, abstractmethod

import numpy as np

class Model(ABC):
  def __init__(self):
    self._norm_value = np.Inf

  @abstractmethod
  def predict_survival(self, parms, *args):
    raise NotImplementedError

  @abstractmethod
  def predict_histograms(self, parms, *args):
    raise NotImplementedError

  @abstractmethod
  def plot_survival(self, xk, ax):
    raise NotImplementedError

  def reporter(self, xk, convergence):
    state = None
    print(f'error={self._norm_value} x_pred={xk} convergence={convergence}')
    if state:
      print(state)
    return True

class ModelFactory(ABC):
  @abstractmethod
  def model(self):
    raise NotImplementedError


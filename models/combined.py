"""
Implements Model and takes several other models that to be fit simultaneously
"""
import numpy as np

from models.model import Model, ModelFactory


class CombinedModel(Model):
  def __init__(self, models):
    assert len(models) > 0, "No models"
    Model.__init__(self)
    self.models = models

  def predict_survival(self, parms, *args):
    return [model.predict_survival(parms, args) for model in self.models]

  def plot_survival(self, xk, axes):
    for model, ax in zip(self.models, axes):
      model.plot_survival(ax)

  def error(self, parms, x_truth, *args):
    err = 0
    for model, truth in zip(self.models, x_truth):
      err += np.square(model.survival_error(parms, truth))
    return np.sqrt(err)

  def wt_error(self, parms):
    err = 0
    for model in self.models:
      err += np.square(model.wt_error(parms))
    return np.sqrt(err)

  def ko_error(self, parms):
    err = 0
    for model in self.models:
      err += np.square(model.ko_error(parms))
    return np.sqrt(err)

  def predict_histograms(self, parms, *args):
    raise Exception('This method should not be called for this class')

class CombinedModelFactory(ModelFactory):
  def model(self, models):
    return CombinedModel(models)

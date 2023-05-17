"""
Fit several models at once!
"""
from data_loading.experiments import CombinedExperiment
from fitting.fit import Fit
from models.combined import CombinedModel

class MultiFit(Fit):
  def plot(self):
    return [f.plot() for f in self.fits]

  def predict(self, parms, eta):
    for fit in self.fits:
      fit.predict(parms, eta)

  def __init__(self, fits):
    assert len(fits) > 0, "No fits"

    self.fits = fits
    self.bounds = []
    self.bounds = fits[0].bounds
    self.experiment = CombinedExperiment([f.experiment for f in fits])
    self.result = None

    label = ''
    plus = ''
    for fit in self.fits:
      label = label + plus + fit.label
      plus = ' + '
    self.label = label

    Fit.__init__(self, CombinedModel([f.model for f in fits]), [f.survival for f in fits], fits[0].parms)

  def error(self, parms, x_truth, *args):
    e = 0
    for truth, fit in zip(x_truth, self.fits):
      e += fit.error(parms, truth, *args) ** 2
    return e**0.5
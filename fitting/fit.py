"""
The abstract base class for
- fitting survival models to data
- returning data
- making predictions
"""
from abc import ABC, abstractmethod

import pandas as pd
from scipy.optimize import differential_evolution, shgo, brute
import scipy.stats as stats
import numpy as np


class Fit(ABC):

  @abstractmethod
  def plot(self, parms, *args, **kwargs):
    raise NotImplementedError

  @abstractmethod
  def predict(self, parms):
    raise NotImplementedError

  def __init__(self, model, survival, parms):
    self.parms = parms
    self.label = parms.get('label')
    self.bounds = parms.get('bounds')
    self.proteins = parms.get('proteins')
    self.result = None
    self.raw_result = None
    self.model = model
    self.survival = survival
    self._histogram_error = self._log_bin_distance
    self.normalisation_provider = self._cohort_normalisation_provider

  def format_result(self, result):
    weights = {}
    parms = result.x
    rc = {'error': result.fun, 'weights': weights}
    if len(parms) == 2*len(self.proteins):
      for i in range(len(self.proteins)):
        p = self.proteins[i]
        weights[p] = parms[2*i]
        weights[p+'^2'] = parms[2*i+1]
    else:
      for d, p in zip(self.proteins, parms):
        weights[d] = p
      if len(parms) > len(self.proteins):
        rc['early_threshold_adjustment'] = parms[len(self.proteins)]

    return rc

  def norm(self, x_pred: float, x_truth: float):
    self._norm_value = np.linalg.norm(x_pred - x_truth)
    return self._norm_value

  def survival_error(self, parms, x_truth, *args):
    x_pred = self.model.predict_survival(parms, *args)
    return self.norm(x_pred, x_truth)

  def wt_error(self):
    return self.model.wt_error(self.result['weights'])

  def ko_error(self):
    return self.model.ko_error(self.result['weights'])

  def _cohort_normalisation_provider(self, time):
    # Normalisation values calculated from cohorts
    wt_real_counts = self.survival_df[self.survival_df['time'] == time]['survival fraction'].iloc[0]
    ko_real_counts = 1
    wt_protein_counts = self.wt_protein_counts[self.wt_protein_counts['time'] == time]['count'].iloc[0]
    ko_protein_counts = self.ko_protein_counts[self.ko_protein_counts['time'] == time]['count'].iloc[0]
    wt_count_normalise = wt_real_counts / wt_protein_counts * 100  # multiply by 100 to keep the error sort of sane
    ko_count_normalise = ko_real_counts / ko_protein_counts * 100
    return wt_count_normalise, ko_count_normalise

  def _population_normalisation_provider(self, time):
    # Normalisation values calculated from counts
    wt_real_counts = self.survival_df[self.survival_df['time'] == time]['WT counts'].iloc[0]
    ko_real_counts = self.survival_df[self.survival_df['time'] == time]['KO counts'].iloc[0]
    wt_protein_counts = self.wt_protein_counts[self.wt_protein_counts['time'] == time]['count'].iloc[0]
    ko_protein_counts = self.ko_protein_counts[self.ko_protein_counts['time'] == time]['count'].iloc[0]
    wt_count_normalise = wt_real_counts / wt_protein_counts / 100  # Divide by 100 to keep the error sort of sane
    ko_count_normalise = ko_real_counts / ko_protein_counts / 100
    return wt_count_normalise, ko_count_normalise

  def histogram_error(self, parms, x_truth, *args):
    living_cells = self.model.predict_histograms(parms, *args)
    wt = self.wt_protein_levels
    error = 0
    for time in self.times:
      lct = living_cells[living_cells['time'] == time]
      wtt = wt[wt['time'] == time]
      wt_count_normalise, ko_count_normalise = self.normalisation_provider(time)
      for protein in self.proteins:
        predicted = lct[protein]
        measured  = wtt[protein]
        e = self._histogram_error(predicted, measured, wt_count_normalise, ko_count_normalise)

        error += e

    return error

  def get_count_normalisation(self):
    ns = []
    for time in self.times:
      wt_count_normalise, ko_count_normalise = self.normalisation_provider(time)
      n = pd.DataFrame({'time': [time], 'wt_correction': [wt_count_normalise/ko_count_normalise]})
      ns.append(n)
    return pd.concat(ns)

  def _log_bin_distance(self, predicted, measured, wt_count_normalise, ko_count_normalise):
    return self._bin_distance(np.log(predicted), np.log(measured), wt_count_normalise, ko_count_normalise, density=False)

  def _normalised_log_bin_distance(self, predicted, measured, wt_count_normalise, ko_count_normalise):
    return self._bin_distance(np.log(predicted), np.log(measured), wt_count_normalise, ko_count_normalise, density=True)

  def _linear_bin_distance(self, predicted, measured, wt_count_normalise, ko_count_normalise):
    return self._bin_distance(predicted, measured, wt_count_normalise, ko_count_normalise, density=False)

  def _normalised_linear_bin_distance(self, predicted, measured, wt_count_normalise, ko_count_normalise):
    return self._bin_distance(predicted, measured, wt_count_normalise, ko_count_normalise, density=True)

  def _bin_distance(self, predicted, measured, wt_count_normalise, ko_count_normalise, density=False):
    n_bins = 200
    if len(predicted) == 0:
      b_start = min(measured)
      b_end = max(measured)
      pred_hist = np.zeros(n_bins)
    else:
      b_start = min(min(predicted), min(measured))
      b_end   = max(max(predicted), max(measured))
      pred_hist, _ = np.histogram(predicted, bins=n_bins, range=(b_start, b_end), density=density)
    meas_hist, x = np.histogram(measured, bins=n_bins, range=(b_start, b_end), density=density)

    if not density:
      # Counts from FACS data are not reliable, so we normalise to known survival counts.
      pred_hist = pred_hist.astype(np.float)
      pred_hist *= ko_count_normalise
      meas_hist = meas_hist.astype(np.float)
      meas_hist *= wt_count_normalise

    e = 0
    for i in range(n_bins):
      e += np.linalg.norm(pred_hist[i] - meas_hist[i])

    return e

  def _ks(self, ph, wh, wt_count_normalise, ko_count_normalise):
    if len(ph) == 0:
      ph = np.zeros(len(wh))
    return stats.ks_2samp(ph, wh).statistic

  def _n(self, ph, wh):
    (pm, ps) = stats.norm.fit(ph)
    (wm, ws) = stats.norm.fit(wh)
    return (pm - wm) ** 2 + (ps - ws) ** 2

  def _ln(self, ph, wh):
    (pm, pl, ps) = stats.lognorm.fit(ph/sum(ph))
    (wm, wl, ws) = stats.lognorm.fit(wh/sum(wh))
    return (pm-wm)**2 + (ps-ws)**2

  def fit(self):
    res = self.optimise(self.survival, self.bounds, self.label)
    self.result = self.format_result(res)
    self.result['WT error'] = self.model.wt_error(res.x)
    self.result['KO error'] = self.model.ko_error(res.x)

    return self.result

  def result(self):
    return self.raw_result

  def survival_frame(self, add_errors=True):
    return self.model.survival_frame(add_errors=add_errors)

  def optimise(self, x_truth: float, bounds: list[(float, float)], label):
    return self.de(x_truth, bounds, label)

  def de(self, x_truth, bounds, label):
    self.label = label
    # The convergence criterion is set fairly aggressively because
    # the error surface seems to have a shallow trough in it. These
    # parameters help reproducibility of the fit.
    # r = brute(self.error,
    #             bounds,
    #             disp=True,
    #             workers=-1,
    #             args=(x_truth,)
    #             )
    # return type('', (), {'x':r})
    return differential_evolution(self.error,
                                  bounds,
                                  workers=-1,
                                  updating='deferred',
                                  tol=1e-4,
                                  args=(x_truth,)
                                  )

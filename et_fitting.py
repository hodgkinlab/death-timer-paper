import argparse
import json
import os
from pprint import pprint
from types import NoneType

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from data.experiments import experiment_validator, mr_70, mr_2022_6, mr_92a, experiment_from_name, mr_95, \
  mr_2022_11, mr_2023_12, mr_2023_13, mr_2023_14, mr_2023_5_unstim
from data_loading.counts import NORMALISE_METHODS
from data_loading.experiments import TcellExperiment, Experiment
from data_loading.loader import PROTEINS, protein_validator
from fitting.ensemble_fit import EnsembleFit
from fitting.multi_fit import MultiFit
from models.factory import model_type_validator
from parameters.parameters import Parameters

EXPORT_FILE_TYPE = 'pdf'
OUTPUT_DIR = 'outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)
THREE_WAY_EXPERIMENTS = [mr_2023_12, mr_2023_13, mr_2023_14]


# This is dictionary of function used to validate a parameter dictionary
VALID_PARAMETERS = {
  'bounds': lambda b: type(b) == list,
  'model_type': model_type_validator,
  'threshold_step': lambda t: type(t) == int,
  'fit_from_time': lambda t: type(t) == int,
  'normalise_early_times': lambda t: type(t) == bool,
  'normalise_method': lambda m: m in NORMALISE_METHODS,
  'include_wildtype': lambda i: type(i) == bool,
  'experiments': experiment_validator,
  'proteins': protein_validator,
  'label': lambda l: True,
  'condition': lambda c: type(c) in [NoneType, str],
  'fixed_protein_weights': lambda w: type(w) in [NoneType, dict],
  'fit_strategy': lambda ft: ft in ['survival', 'histograms'],
}

# A good set of starting parameters
DEFAULT_PARAMETERS = {
  'bounds': [(0, 40), (0, 40), (0, 40), (-40, 0)],
  'model_type': 'linear',
  'threshold_step': 0,
  'fit_from_time': 72,
  'normalise_early_times': False,
  'normalise_method': 'fitted survival',
  'include_wildtype': False,
  'proteins': PROTEINS,
  'fixed_protein_weights': None,
  'condition': None,
  'fit_strategy': 'histograms',
  'label': '-- hey! --',
}

def parse_args():
  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--figure',
                      dest='figure',
                      required=True,
                      help='plot a specific figure or set of figures')
  return parser.parse_args()

def _load_weights_cache():
  # Load weights from the cache, return an empty dictionary is the cache hasn't been initialised
  cache_fn = os.path.join(OUTPUT_DIR, 'weights-cache.json')
  if os.path.exists(cache_fn):
    with open(cache_fn, 'rb') as f:
      weights = json.load(f)
  else:
    weights = {}
  return weights

def weights_from_experiment(exp, model_type='linear'):
  # Return weights from the cache or None is they are not present
  if issubclass(type(exp), Experiment):
    exp = exp.name
  weights = _load_weights_cache()
  weights_for_model_type = weights.get(model_type, {})
  return weights_for_model_type.get(exp, None)

def save_weights(exp, weights, model_type='linear'):
  # Save weights to the cache in an easily readable format.
  if issubclass(type(exp), Experiment):
    exp = exp.name
  weights_d = _load_weights_cache()
  weights_for_model_type = weights_d.get(model_type, {})
  weights_for_model_type[exp] = weights
  weights_d[model_type] = weights_for_model_type
  cache_fn = os.path.join(OUTPUT_DIR, 'weights-cache.json')
  with open(cache_fn, 'w') as f:
    json.dump(weights_d, f, indent=2)


def _kde_for(df):
  # Calculate a kernel density estimate for histrogram plotting
  # This what kdeplot does!
  kde_frame = pd.DataFrame()

  pl = np.log10(df['level'])
  kde = stats.gaussian_kde(pl, bw_method='scott')

  bw = np.sqrt(kde.covariance.squeeze())
  xmin = min(pl) - 3*bw
  xmax = max(pl) + 3*bw
  x = np.linspace(xmin, xmax, 400)

  kde_frame['x'] = np.power(10, x)
  kde_frame['level'] = kde(x)

  return kde_frame

def plot_histograms_from_weights(parms, weights, xlim=None):
  # The parms object contains:
  # - the experiment
  # - parameters for data handling
  # - model hyper-parameters
  # weights are previously fitted (or guessed!) weights.

  # Create a fitter object and run the model with the weights
  f = EnsembleFit(parms)
  f.predict(weights)

  # The count_normalisation contains the normalisation from FACS counts to cell counts
  # relative to KO which is taken as 1
  count_normalisation = f.get_count_normalisation()
  count_normalisation.rename(columns={'wt_correction': 'WT'}, inplace=True)
  count_normalisation['KO'] = 1
  count_normalisation['WT fit'] = 1
  count_normalisation = pd.melt(count_normalisation, id_vars=['time'], var_name='source', value_name='correction', value_vars=['KO', 'WT', 'WT fit'])

  # The measured wild type protein levels
  wt  = f.wt_protein_levels.copy()
  wt.loc[:, 'source'] = 'WT'
  # The measured KO protein levels
  ko  = f.ko_protein_levels.copy()
  ko.loc[:, 'source'] = 'KO'
  # The predicted WT protein levels
  prd = f.predicted_histograms.copy()
  prd.loc[:, 'source'] = 'WT fit'
  # The time points in tha data
  times = f.times
  times.sort()

  # All 3 sets of data in the one frame
  df = pd.concat([wt, ko, prd])
  df.reset_index(inplace=True, drop=True)

  # Reshape so that protein levels are in one column, rather than columns per protein
  raw_protein_levels = pd.melt(df, id_vars=['time', 'source'], value_name='level', var_name='protein', value_vars=PROTEINS)
  kg = raw_protein_levels.groupby(['time', 'source', 'protein'])

  # The cell counts in protein histograms derived from FACS
  source_counts = kg.count()

  # KDEs derived from the protein levels.
  kde = kg.apply(_kde_for)

  # KDEs are normalised to an area of one, so they are rescaled to match
  # relative counts in each group.

  # 1. Multiply KDE histograms by counts in each source (WT, KO, predicted)
  out = kde.merge(source_counts, left_on=['time', 'source', 'protein'], right_on=['time', 'source', 'protein'], how='left')\
    .assign(level=lambda x: x['level_x'].multiply(x['level_y']))
  kde = out.drop(columns=['level_y', 'level_x'])

  # 2. Now divide by the total count. Now counts for each source group are normalised
  #    relative to each other according to FACS counts
  total_counts = raw_protein_levels.groupby(['time', 'protein']).count()
  kde.reset_index(inplace=True)
  out = kde.merge(total_counts, left_on=['time', 'protein'], right_on=['time', 'protein'], how='left')\
    .assign(level=lambda x: x['level_x'].div(x['source_y']))
  kde = out.drop(columns=['source_y', 'level_y', 'level_x'])
  kde.rename(columns={'source_x': 'source'}, inplace=True)

  # 3. Finally, correct by counts determined by division data (true counts)
  out = kde.merge(count_normalisation, left_on=['time', 'source'], right_on=['time', 'source'], how='left')\
    .assign(level=lambda x: x['level'].multiply(x['correction']))
  kde = out.reset_index()

  fig, axes = plt.subplots(nrows=len(times), ncols=4, figsize=(12, 12))

  ymax = 0
  legend = True
  set_title = True
  palette = {'WT': '#0C0C0C', 'KO': '#E20613', 'WT fit': '#B2B2B2'}
  dashes = [(1, 0), (2, 2), (2, 2)]
  for row, time in enumerate(times):
    set_ylabel = True
    kdet = kde[kde['time'] == time]
    for col, protein in enumerate(PROTEINS):
      ax = axes[row, col]

      kdep = kdet[kdet['protein']==protein]
      sns.lineplot(data=kdep, x='x', y='level', hue='source', style='source', dashes=dashes, palette=palette, legend=legend, ax=ax)
      ax.set_xlabel(None)
      ax.set(xscale='log')
      if xlim:
        ax.set_xlim(xlim[0], xlim[1])

      legend = False
      if set_ylabel:
        yl = f'{time}h'
        set_ylabel = False
      else:
        yl = None
      ax.set_ylabel(yl)
      ax.set_xlabel(None)

      if set_title:
        ax.set_title(protein)

      ymax = max(ymax, ax.get_ylim()[1])
    set_title = False

  for col, protein in enumerate(PROTEINS):
    for row, time in enumerate(times):
      ax = axes[row, col]
      ax.set_xlim(0.005, 50)
      ax.set_ylim(0, ymax)

  return fig

def get_parms(experiment):
  # Build a basic set of starting parameters that gets passed through the pipeline
  parms = Parameters(DEFAULT_PARAMETERS)
  parms.set('experiments', experiment)
  label = f'{experiment.name}'
  parms.set('label', label)
  if type(experiment) == TcellExperiment:
    parms.set('fit_from_time', 96)
  else:
    parms.set('fit_from_time', 72)
  parms.validate(VALID_PARAMETERS)
  return parms

def get_or_calculate_weights(fitter, experiment, model_type='linear'):
  # If weights are found in the cache return those otherwise
  # run the fit and save the new weights in the cache.
  if issubclass(type(experiment), Experiment):
    experiment = experiment.name
  weights = weights_from_experiment(experiment, model_type=model_type)
  if weights is None:
    print(f'Fitting weights for {experiment}')
    result = fitter.fit()
    pprint(result)
    weights = result['weights']
    save_weights(experiment, weights, model_type=model_type)
  else:
    print(f'Using cached weights for {experiment}')
  return weights

def single_fit(exp,
               weights=None,
               label=None,
               xscale_data=None,
               model_type='linear',
               fit_from_time=72,
               normalise_method='fitted survival'):

  # Fit a single experiment to data

  parms = get_parms(exp)
  parms.set('model_type', model_type)
  parms.set('fit_from_time', fit_from_time)
  parms.set('normalise_method', normalise_method)
  parms.validate(VALID_PARAMETERS)

  f = EnsembleFit(parms)

  if weights is None:
    weights = get_or_calculate_weights(f, exp, model_type=model_type)
  else:
    print('Using provided weights')

  f.predict(weights)
  ensemble_fig = f.plot(ensemble_xlim=exp.ensemble_xlim_hint(), logicle=True, xscale_data=xscale_data)
  histogram_fig = plot_histograms_from_weights(parms, weights, xlim=exp.histogram_xlim_hint())

  if type(label) == dict:
    e_label = label['ensembles']
    h_label = label['histograms']
  else:
    e_label = label
    h_label = label

  fn = os.path.join(OUTPUT_DIR, f'{e_label}.{EXPORT_FILE_TYPE}')
  ensemble_fig.savefig(fn)
  plt.close()
  fn = os.path.join(OUTPUT_DIR, f'{h_label}.{EXPORT_FILE_TYPE}')
  histogram_fig.savefig(fn)
  plt.close()

  return 0

def get_inhibition_factor(condition, experiment):
  # Retrieve an inhibition factor from the cache, return None if not present.
  if condition == '0uM':
    return 1
  weights = _load_weights_cache()
  inhfs = weights.get('inhibition factors', {})
  inhfs = inhfs.get(experiment.name, {})
  return inhfs.get(condition, None)

def save_inhibition_factor(condition, experiment, inhf):
  # Save and inihibtion factor to the cache.
  weights = _load_weights_cache()
  inhfs = weights.get('inhibition factors', {})
  exps = inhfs.get(experiment.name, {})
  exps[condition] = inhf
  inhfs[experiment.name] = exps
  weights['inhibition factors'] = inhfs
  cache_fn = os.path.join(OUTPUT_DIR, 'weights-cache.json')
  with open(cache_fn, 'w') as f:
    json.dump(weights, f, indent=2)

def get_or_calculate_inhibition_factor(condition, experiment=mr_95):
  # Retrieve the inhibition factor from the cache or calculate it if it isn't there.
  inhf = get_inhibition_factor(condition, experiment)
  if inhf is not None:
    return inhf

  # The protein being inhibited is in the condition name
  protein = 'BCL2' if 'BCL2' in condition else 'MCL1'

  # Get the uninhibited weights.
  parms = get_parms(experiment)
  parms.set('fit_from_time', 48)
  parms.validate(VALID_PARAMETERS)
  f = EnsembleFit(parms)
  weights = get_or_calculate_weights(f, experiment)

  # Make a copy of the weights with the weight for the inhibited protein removed.
  fixed_weights = weights.copy()
  del fixed_weights[protein]

  parms = get_parms(experiment)

  # Drug experiments are fitted from 48h because death
  # has already started at this point.
  parms.set('fit_from_time', 48)

  # Only one weight is being fitted hence only one bound
  parms.set('bounds', [(0, 40)])

  # Other weights are fixed and used as is.
  parms.set('fixed_protein_weights', fixed_weights)

  # Only fit to this protein
  parms.set('proteins', [protein])

  # Pass the condition so the pipeline know which data to use
  parms.set('condition', condition)

  parms.validate(VALID_PARAMETERS)
  f = EnsembleFit(parms)
  print(f'starting {condition}')
  result = f.fit()
  pprint(result)
  new_weights = result['weights']
  # The inhibition factor is the ratio of the new weight to the old weight
  inhf = new_weights[protein] / weights[protein]
  save_inhibition_factor(condition, experiment, inhf)

  return inhf

def make_multi_fitter():
  # Make a fitter object for fitting several experiments simultaneously
  # to data with the same parameters.
  parm_set = []
  for exp in THREE_WAY_EXPERIMENTS:
    p = get_parms(exp)
    p.set('label', f'{exp.name} - 3 way fit')
    parm_set.append(p)

  fits = [EnsembleFit(p) for p in parm_set]
  return MultiFit(fits), fits, parm_set

def three_way_fit(survival_only=False, ensemble_figs=[None] * 3):
  fitter, fits, parm_set = make_multi_fitter()
  weights = get_or_calculate_weights(fitter, experiment='3 way')

  ensemble_plots = {}
  histogram_plots = {}
  for fit, parms, fig in zip(fits, parm_set, ensemble_figs):
    fit.predict(weights)
    exp = parms.get('experiments')
    if exp.name == 'MR2022.6':
      xlim = [-100, 100]
    else:
      xlim = [-100, 300]
    ensemble_fig = fit.plot(ensemble_xlim=xlim, logicle=True, survival_only=survival_only, fig=fig)
    histogram_fig = plot_histograms_from_weights(parms, weights)
    ensemble_plots[exp.name] = ensemble_fig
    histogram_plots[exp.name] = histogram_fig

  return (ensemble_plots, histogram_plots)

def plot_drug_fit_for_condition(experiment, inhf={}, dashes=None, fig=None, label_prefix=None, condition=None):
  weights = weights_from_experiment(experiment)

  parms = get_parms(experiment)
  parms.set('condition', condition)
  parms.set('fit_from_time', 48)
  parms.validate(VALID_PARAMETERS)

  for p in inhf.keys():
    weights[p] *= inhf[p]

  f = EnsembleFit(parms)
  f.predict(weights)
  fig = f.plot(logicle=True, ensemble_xlim=[-100, 200], survival_only=False, fig=fig, dashes=dashes, label_prefix=label_prefix)

  return fig

def vary_protein_list(experiment, fig_name):
  # Determine and plot the error in the fit based on the combination
  # of proteins used.

  # The BIM weight needs to be allowed to be positive in the case of BIM alone
  bd = {'BCL2': (0, 40), 'BCLxL': (0, 40), 'MCL1': (0, 40), 'BIM': (-40, 40)}

  # Create a list of the 15 possible protein combinations
  list_of_lists = [PROTEINS]
  for p in PROTEINS:
    pl = PROTEINS.copy()
    list_of_lists.append([p])
    pl.remove(p)
    list_of_lists.append(pl)
    for p2 in PROTEINS[PROTEINS.index(p)+1:]:
      list_of_lists.append([p, p2])

  # Sort the list of lists based on the length, i.e. the number of proteins
  list_of_lists.sort(key=len)

  # For each combination of proteins do a fit and accumulate on a frame
  df = pd.DataFrame({'protein list': [], 'experiment': [], 'fit error': [], 'WT survival error': [], 'KO survival error': []})
  for pl in list_of_lists:
    label = '+'.join(pl)
    parms = get_parms(experiment)
    parms.set('label', label)
    parms.set('proteins', pl)
    parms.set('bounds', [bd[p] for p in pl])
    parms.validate(VALID_PARAMETERS)
    f = EnsembleFit(parms)
    label = parms.get('label')
    print(f'Starting: {label}')
    result = f.fit()
    pprint(result)

    df.loc[len(df.index)] = [label, experiment.name, result['error'], result['WT error'], result['KO error']]

  # fn = os.path.join(OUTPUT_DIR, 'protein contributions.csv')
  # df.to_csv(fn)

  # Make a column which is the number of proteins
  def _n_proteins(row):
    l = row['protein list']
    row['number of proteins'] = l.count('+') + 1
    return row
  df = df.apply(_n_proteins, axis=1)

  # ... and plot!
  for error_type in ['WT', 'KO']:
    sns.boxplot(data=df, x='number of proteins', y=f'{error_type} survival error')
    fn = os.path.join(OUTPUT_DIR, f'{fig_name} {error_type}.{EXPORT_FILE_TYPE}')
    plt.savefig(fn)
    plt.close()


def plot_all_drug_fits(fig_label='?'):
  experiment = mr_95

  parms = get_parms(experiment)
  f = EnsembleFit(parms)
  control_weights = get_or_calculate_weights(f, experiment)

  conditions = ['0uM', '0.1uM MCL1i', '1uM MCL1i', '0.1uM BCL2i', '1uM BCL2i']
  fixed_proteins = [None, 'MCL1', 'MCL1', 'BCL2', 'BCL2']
  panels = ['c', 'e', 'f', 'j', 'k']
  survival_fig, _ = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))
  for condition, fixed_protein, panel in zip(conditions, fixed_proteins, panels):
    inhf = get_or_calculate_inhibition_factor(condition, experiment)
    fixed_protein_weights = control_weights.copy()
    if fixed_protein is not None:
      fixed_protein_weights[fixed_protein] *= inhf

    parms = get_parms(experiment)
    parms.set('condition', condition)
    parms.set('fit_from_time', 48)
    parms.validate(VALID_PARAMETERS)

    f = EnsembleFit(parms)
    f.predict(fixed_protein_weights)

    ensemble_fig = f.plot(ensemble_xlim=[-100, 300], logicle=True)
    fn = os.path.join(OUTPUT_DIR, f'{fig_label}{panel} {condition} ensembles.{EXPORT_FILE_TYPE}')
    ensemble_fig.savefig(fn)

    if fixed_protein in [None, 'BCL2']:
      f.plot(ensemble_xlim=[-100, 300], logicle=True, survival_only=True, fig=survival_fig, label_prefix=condition)

    histogram_fig = plot_histograms_from_weights(parms, fixed_protein_weights)
    fn = os.path.join(OUTPUT_DIR, f'{fig_label}{panel} {condition} histograms.{EXPORT_FILE_TYPE}')
    histogram_fig.savefig(fn)

  fn = os.path.join(OUTPUT_DIR, f'{fig_label}g BCL2i.{EXPORT_FILE_TYPE}')
  survival_fig.savefig(fn)

def plot_3_way(fig_label):
  ensemble_plots, histogram_plots = three_way_fit()
  for experiment, plot in ensemble_plots.items():
    plot.savefig(os.path.join(OUTPUT_DIR, f'{fig_label} ensemble - {experiment}.{EXPORT_FILE_TYPE}'))
  for experiment, plot in histogram_plots.items():
    plot.savefig(os.path.join(OUTPUT_DIR, f'{fig_label} histograms - {experiment}.{EXPORT_FILE_TYPE}'))

def plot_ratio_model(experiment, fig_name):
  fig, _ = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))
  dasheses = [[(1, 0), (2, 2), (2, 2)], [(1, 0), (4, 4), (4, 4)]]
  for model_type, dashes in zip(['linear', 'ratio'], dasheses):
    parms = get_parms(experiment)
    parms.set('model_type', model_type)
    if model_type == 'ratio':
      parms.set('bounds', [(0, 40), (0, 40), (0, 40)])
    parms.validate(VALID_PARAMETERS)
    f = EnsembleFit(parms)
    weights = get_or_calculate_weights(f, experiment, model_type=model_type)
    f.predict(weights)
    fig = f.plot(survival_only=True, fig=fig, dashes=dashes, label_prefix=model_type)

  outfn = os.path.join(OUTPUT_DIR, f'{fig_name}.{EXPORT_FILE_TYPE}')
  fig.savefig(outfn)
  plt.close()

def plot_drug_fitting(fig_label, panels):
  experiment = mr_95
  dasheses = [[(2, 2), (2, 2), (2, 2)], [(3, 3), (3, 3), (3, 3)], [(4, 4), (4, 4), (4, 4)]]
  label_prefixes = ['control', '0.1uM', '1uM']
  conditions = ['0uM', '0.1uM MCL1i', '1uM MCL1i']

  for condition, label_prefix, dashes, panel in zip(conditions, label_prefixes, dasheses, panels):
    inhf = get_or_calculate_inhibition_factor(condition, experiment)
    fig = plot_drug_fit_for_condition(experiment,
                                inhf={'MCL1': inhf},
                                condition=condition,
                                dashes=dashes,
                                label_prefix=label_prefix)
    outfn = os.path.join(OUTPUT_DIR, f'{fig_label}{panel}.{EXPORT_FILE_TYPE}')
    fig.savefig(outfn)
    plt.close()

def sensitivity(exp_name, mode='multiplication', fig_label='?'):
  if mode == 'multiplication':
    def _op(x, y): return x * y
    wfr = np.arange(0, 10.1, 0.1)
  else:
    def _op(x, y): return x + y
    wfr = np.arange(-5, 5.1, 0.1)

  if exp_name == '3 way':
    experiments = THREE_WAY_EXPERIMENTS
  else:
    experiments = [experiment_from_name(exp_name)]

  weights = weights_from_experiment(exp_name)

  big_df = pd.DataFrame()

  for experiment in experiments:
    for protein, w in weights.items():
      these_weights = weights.copy()
      wt_errors = []
      ko_errors = []
      hg_errors = []
      parms = get_parms(experiment)
      parms.validate(VALID_PARAMETERS)
      fitter = EnsembleFit(parms)
      for wf in wfr:
        these_weights[protein] = _op(wf, weights[protein])
        weight_vector = [these_weights[p] for p in PROTEINS]
        fitter.predict(these_weights)
        wt_e = fitter.wt_error()
        wt_errors.append(wt_e)
        ko_e = fitter.ko_error()
        ko_errors.append(ko_e)
        hg_e = fitter.histogram_error(weight_vector, None, fitter.ko_protein_levels)
        hg_errors.append(hg_e)
      df_hg = pd.DataFrame({'weight variation': wfr, 'error': hg_errors, 'source': 'histogram'})
      df_hg['experiment'] = exp_name
      df_hg['protein']    = protein
      df_ko = pd.DataFrame({'weight variation': wfr, 'error': ko_errors, 'source': 'KO'})
      df_ko['experiment'] = exp_name
      df_ko['protein']    = protein
      df_wt = pd.DataFrame({'weight variation': wfr, 'error': wt_errors, 'source': 'WT'})
      df_wt['experiment'] = exp_name
      df_wt['protein']    = protein
      big_df = pd.concat([big_df, df_wt, df_ko, df_hg])

  if exp_name == '3 way':
    def _rse(row):
      r = {
        'weight variation': row['weight variation'].iloc[0],
        'error': np.sqrt((row['error']**2).sum()),
        'source': row['source'].iloc[0],
        'experiment': 'combined',
        'protein': row['protein'].iloc[0]
      }
      return pd.DataFrame(r, index=[0])

    rse = big_df.groupby(['weight variation', 'source', 'protein']).apply(_rse)
    big_df = pd.concat([big_df, rse])

  source = 'histogram'
  _, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))
  df = big_df[(big_df['source'] == source) & (big_df['experiment'] == ('combined' if exp_name == '3 way' else exp_name))]
  ax = sns.lineplot(data=df, x='weight variation', y='error', hue='protein', ax=ax)
  title = f'{exp_name} - {mode} - {source} error'
  ax.set_title(title)
  fn = os.path.join(OUTPUT_DIR, f'{fig_label} - {exp_name} - {mode}.{EXPORT_FILE_TYPE}')
  plt.savefig(fn)
  plt.close()

def figure_routing(fig):
  if fig == '3':
    # Fit the 'hero' experiment and generate plots.
    single_fit(mr_2022_6, label={'ensembles': f'fig 3ce', 'histograms': f'fig 3f'})
    vary_protein_list(mr_2022_6, 'fig 3g')
    sensitivity(mr_2022_6.name, mode='addition', fig_label='fig 3h')
    return 0

  if fig == '4':
    # Fit the drug experiments
    plot_drug_fitting(fig_label='fig 4', panels=['c', 'd', 'e'])
    return 0

  if fig == '5':
    # T cells
    single_fit(mr_2022_11,
               label={'ensembles': 'fig 5df', 'histograms': 'fig 5g'},
               fit_from_time=96,
               normalise_method='PI stain')
    return 0

  if fig == 'S3':
    # Additional experiments:
    # 1. MR-70
    single_fit(mr_70, label={'ensembles': 'fig S3a1', 'histograms': 'fig S3a2'})
    vary_protein_list(mr_70, 'fig S3b')
    sensitivity(mr_70.name, mode='addition', fig_label='fig S3c')

    # # 2. MR-92a
    single_fit(mr_92a, label={'ensembles': 'fig S3d1', 'histograms': 'fig S3d2'})
    vary_protein_list(mr_92a, 'fig S3e')
    sensitivity(mr_92a.name, mode='addition', fig_label='fig S3f')

    # 3. Three simultaneous experiments
    plot_3_way(fig_label='fig S3ghij')
    sensitivity('3 way', mode='addition', fig_label='fig S3l')

    return 0

  if fig == 'S4':
    # The ratio ensemble model
    single_fit(mr_2022_6,
               label={'ensembles': 'fig S4a1', 'histograms': 'fig S4a2'},
               model_type='ratio')
    plot_ratio_model(mr_2022_6, 'fig S4a3')
    single_fit(mr_70,
               label={'ensembles': 'fig S4b1', 'histograms': 'fig S4b2'},
               model_type='ratio')
    plot_ratio_model(mr_70, 'fig S4b3')
    single_fit(mr_92a,
               label={'ensembles': 'fig S4c1', 'histograms': 'fig S4c2'},
               model_type='ratio')
    plot_ratio_model(mr_92a, 'fig S4c3')
    return 0

  if fig == 'S5':
    plot_all_drug_fits(fig_label='fig S5')
    return 0

  if fig == 'Unstim B':
    single_fit(mr_2023_5_unstim,
               label={'ensembles': 'fig unstim b ensembles', 'histograms': 'fig unstim b histogram'},
               fit_from_time=1)
    return 0

  print('Unrecognised figure')
  return 1

if __name__ == '__main__':
  opts = parse_args()
  if opts.figure is not None:
    rc = figure_routing(opts.figure)
    quit(rc)


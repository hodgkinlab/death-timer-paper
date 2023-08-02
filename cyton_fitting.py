import fnmatch
import os
import numpy as np
import pandas as pd
from pprint import pprint
import lmfit as lmf
from cyton.parse import parse_data
from cyton.model import Cyton2Model   # Full Cyton model. (Options: choose all variable to be either logN or N)
from scipy.stats import gmean, lognorm
import seaborn as sns
import matplotlib.pyplot as plt

rng = np.random.RandomState(seed=89907530)

FIGURE_MAP = {
  'MR-55 BCL2i Cohort template': 'fig 1b',
  'MR-55 MCL1i Cohort template': 'fig 1c',
  'MR-55 BCLxLi Cohort template': 'fig 1d',
  'MR-60 BIM ko Cohort template': 'fig 1e'
}

OUTPUT = 'outputs'
os.makedirs(OUTPUT, exist_ok=True)

DATA_DIR = os.path.join('data', 'For Cyton')
ITER_SEARCH = 100
LM_FIT_KWS = {      # [LMFIT/SciPy] Key-word arguments pass to LMFIT minimizer for Levenberg-Marquardt algorithm
  # 'ftol': 1E-10,  # Relative error desired in the sum of squares. DEFAULT: 1.49012E-8
  # 'xtol': 1E-10,  # Relative error desired in the approximate solution. DEFAULT: 1.49012E-8
  # 'gtol': 0.0,    # Orthogonality desired between the function vector and the columns of the Jacobian. DEFAULT: 0.0
  'epsfcn': 1E-4  # A variable used in determining a suitable step length for the forward-difference approximation of the Jacobian (for Dfun=None). Normally the actual step length will be sqrt(epsfcn)*x If epsfcn is less than the machine precision, it is assumed that the relative errors are of the order of the machine precision. Default value is around 2E-16. As it turns out, the optimisation routine starts by making a very small move (functinal evaluation) and calculating the finite-difference Jacobian matrix to determine the direction to move. The default value is too small to detect sensitivity of 'm' parameter.
}
DT = 0.5            # [Cyton Model] Time step
VARY_ROUND1 = {  # True = Subject to change; False = Lock parameter
  'mUns': False, 'sUns': False,
  'mDiv0': True, 'sDiv0': True,
  'mDD': True, 'sDD': True,
  'mDie': True, 'sDie': True,
  'm': True, 'p': False
}
VARY_ROUND2 = {  # True = Subject to change; False = Lock parameter
  'mUns': False, 'sUns': False,
  'mDiv0': False, 'sDiv0': False,
  'mDD': False, 'sDD': False,
  'mDie': True, 'sDie': True,
  'm': False, 'p': False
}


def residual_lm(pars, data, model):
  vals = pars.valuesdict()
  mUns, sUns = vals['mUns'], vals['sUns']
  mDiv0, sDiv0 = vals['mDiv0'], vals['sDiv0']
  mDD, sDD = vals['mDD'], vals['sDD']
  mDie, sDie = vals['mDie'], vals['sDie']
  m, p = vals['m'], vals['p']

  pred = model.evaluate(mUns, sUns, mDiv0, sDiv0, mDD, sDD, mDie, sDie, m, p)

  return (data - pred)

def do_fit_lm(params, paramExcl, y_cells, model):
  candidates = {'algo': [], 'result': [], 'residual': []}  # store fitted parameter and its residual
  for _ in range(ITER_SEARCH):
    # Random initial values
    for par in params:
      if par in paramExcl:
        pass  # Ignore excluded parameters
      else:
        par_min, par_max = params[par].min, params[par].max  # determine its min and max range
        params[par].set(value=rng.uniform(low=par_min, high=par_max))

    try:  # Some set of initial values is completely non-sensical, resulted in NaN errors
      mini_lm = lmf.Minimizer(residual_lm, params, fcn_args=(y_cells, model), **LM_FIT_KWS)
      res_lm = mini_lm.minimize(method='leastsq', max_nfev=None)  # Levenberg-Marquardt algorithm
      # res_lm = mini_lm.minimize(method='least_squares', max_nfev=MAX_NFEV)  # Trust Region Reflective method

      algo = 'LM'  # record algorithm name
      result = res_lm
      resid = res_lm.chisqr

      candidates['algo'].append(algo)
      candidates['result'].append(result)
      candidates['residual'].append(resid)
    except ValueError as _:
      pass # It's Python!

  fit_results = pd.DataFrame(candidates)
  fit_results.sort_values('residual', ascending=True, inplace=True)  # sort based on residual

  # Extract best-fit parameters
  best_result = fit_results.iloc[0]['result']
  best_fit = best_result.params.valuesdict()

  return best_fit

def fit_reps(df=None, reader=None, icnd=None, pars=None, vary=None):

  hts = reader.harvested_times[icnd]
  mgen = reader.generation_per_condition[icnd]

  ### ANALYSIS: ESTIMATE DIVISION PARAMETERS BY FITTING CYTON MODEL
  bounds = {
    'lb': {  # Lower bounds
      'mUns': 1E-4, 'sUns': 1E-3,
      'mDiv0': 0, 'sDiv0': 1E-2,
      'mDD': 0, 'sDD': 1E-2,
      'mDie': 0, 'sDie': 1E-2,
      'm': 5, 'p': 0
    },
    'ub': {  # Upper bounds
      'mUns': 1000, 'sUns': 2,
      'mDiv0': 200, 'sDiv0': 200,
      'mDD': 200, 'sDD': 200,
      'mDie': 400, 'sDie': 200,
      'm': 50, 'p': 1
    }
  }

  ### PREPARE DATA
  conv_df = pd.DataFrame(df['cgens']['rep'][icnd])
  conv_df.index = hts

  nreps = []
  for idx, row in conv_df.iterrows():
    nreps.append(len(row.dropna()))
  data = df['cgens']['rep'][icnd]  # n(g,t): number of cells in generation g at time t
  # Manually ravel the data. This allows asymmetric replicate numbers.
  all_x_gens, all_y_cells = [], []
  for datum in data:
    for irep, rep in enumerate(datum):
      for igen, cell in enumerate(rep):
        all_x_gens.append(igen)
        all_y_cells.append(cell)

  # Manually ravel the data. This allows asymmetric replicate numbers.
  y_cells = []
  init, _hts, _nreps = True, [], []
  for idx, row in conv_df.iterrows():
    irep = 0
    for cgen in row:
      if cgen is not None:
        for igen, cell in enumerate(cgen):
          y_cells.append(cell)
        _hts.append(idx)
        irep += 1
    # check if the row is empty
    if not all(v is None for v in row):
      _nreps.append(irep)
      if init:
        init = False
        avgN0 = np.array(row.dropna().values.tolist()).mean(axis=0).sum()
  _hts = np.unique(_hts)
  y_cells = np.asfarray(y_cells)

  params = lmf.Parameters()
  # LMFIT add parameter properties with tuples: (NAME, VALUE, VARY, MIN, MAX, EXPR, BRUTE_STEP)
  for par in pars:
    params.add(par, value=pars[par], min=bounds['lb'][par], max=bounds['ub'][par], vary=vary[par])
  paramExcl = [p for p in params if not params[p].vary]  # List of parameters excluded from fitting (i.e. vary=False)

  model = Cyton2Model(_hts, avgN0, mgen, DT, _nreps, True)

  best_fit = do_fit_lm(params, paramExcl, y_cells, model)

  return best_fit

def load_data():
  files = fnmatch.filter(os.listdir(DATA_DIR), '*.xlsx')
  df = parse_data(DATA_DIR, files)
  return df

def process_data(df):

  pars = { # Initial values
    'mUns': 1000,
    'sUns': 1E-3,  # Unstimulated death time (NOT USED HERE)
    'mDiv0': 30,
    'sDiv0': 10,  # Time to first division
    'mDD': 60,
    'sDD': 10,  # Time to division destiny
    'mDie': 80,
    'sDie': 10,  # Time to death
    'm': 10,
    'p': 1  # Subsequent division time & Activation probability (ASSUME ALL CELLS ACTIVATED)
  }

  out_df = pd.DataFrame()
  for experiment, df in df.items():
    reader = df['reader']
    conditions = reader.condition_names

    ## Round 1: largely unconstrained fit
    r1_df = pd.DataFrame()
    for icnd, condition in enumerate(conditions):
      fit = fit_reps(df=df, reader=reader, icnd=icnd, pars=pars, vary=VARY_ROUND1)
      r1_df = pd.concat([r1_df, pd.DataFrame(fit, index=[0])])

    ## Round 2: Fix some parameters to the gmeans from round 1 and refit
    means = r1_df.apply(gmean, axis=0)
    for k, v in VARY_ROUND2.items():
      if not v: pars[k] = means[k]

    for icnd, condition in enumerate(conditions):
      fit = fit_reps(df=df, reader=reader, icnd=icnd, pars=pars, vary=VARY_ROUND2)
      print(f'experiment: {experiment}, condition: {condition}')
      pprint(fit)

      fit['experiment'] = experiment
      fit['condition'] = condition
      out_df = pd.concat([out_df, pd.DataFrame(fit, index=[0])])



  return out_df

def _add_pdf_to_df(m, s, condition, tgt_df, negate):
  xmax = max(150, round(m / 100) * 100 * 2.5)
  x = np.linspace(0, xmax, 100)
  y = lognorm.pdf(x, s, scale=m)
  y = -y if negate else y
  df = pd.DataFrame({'time': x, 'y': y})
  df['condition'] = condition
  return pd.concat([tgt_df, df])

def _parms_from_df(df, condition):
  parm_names = ['mUns','sUns', 'mDiv0', 'sDiv0', 'mDD', 'sDD', 'mDie', 'sDie','m', 'p']
  df = df[df['condition'] == condition]
  parms = {}
  for p in parm_names:
    parms[p] = df[p].iloc[0]
  return parms

def plot_population(in_data, fitted_parms_df):
  experiments = list(fitted_parms_df['experiment'].unique())
  for experiment in experiments:
    experiment_df = fitted_parms_df[fitted_parms_df['experiment'] == experiment]
    exp_in = in_data[experiment]
    reader = exp_in['reader']
    all_cells = exp_in['cells']['avg']
    time = reader.harvested_times[0]
    ncnd = len(reader.condition_names)
    plot_df = pd.DataFrame()
    for icnd in range(ncnd):
      condition = reader.condition_names[icnd]
      cells = all_cells[icnd]
      cells = [c/cells[0] for c in cells] # normalise to initial cell count
      df = pd.DataFrame({'time': time, 'count': cells})
      df['condition'] = condition
      df['source'] = 'data'
      plot_df = pd.concat([plot_df, df])

      x = np.linspace(0, max(time), max(time)+1)
      parms = _parms_from_df(experiment_df, condition)
      b_model = Cyton2Model(time, cells[0], 10, 1, 1, True)
      b_extrapolate = b_model.extrapolate(x, parms)
      df = pd.DataFrame({'time': x, 'count': b_extrapolate['ext']['total_live_cells']})
      df['condition'] = condition
      df['source'] = 'cyton'
      plot_df = pd.concat([plot_df, df])

    ax = sns.lineplot(data=plot_df, x='time', y='count', hue='condition', style='source', markers=True)
    ax.set_title(experiment)
    figname = FIGURE_MAP[experiment]
    fn = os.path.join(OUTPUT, f'{figname} population fits.pdf')
    plt.savefig(fn)
    # plt.show()
    plt.close()


def plot_data(in_df, fitted_parms_df):
  plot_distributions(fitted_parms_df)
  plot_population(in_df, fitted_parms_df)

def plot_distributions(fitted_parms_df):
  experiments = list(fitted_parms_df['experiment'].unique())
  for experiment in experiments:
    experiment_df = fitted_parms_df[fitted_parms_df['experiment'] == experiment]
    conditions = list(experiment_df['condition'].unique())
    distributions = pd.DataFrame()
    for condition in conditions:
      ## Destiny distributions
      m = experiment_df[experiment_df['condition'] == condition]['mDie'].iloc[0]
      s = experiment_df[experiment_df['condition'] == condition]['sDie'].iloc[0]
      distributions = _add_pdf_to_df(m, s, condition, distributions, True)

      ## Division and destiny distributions for control
      if condition in ['0', 'WT']:
        m = experiment_df[experiment_df['condition'] == condition]['mDiv0'].iloc[0]
        s = experiment_df[experiment_df['condition'] == condition]['sDiv0'].iloc[0]
        distributions = _add_pdf_to_df(m, s, 'division 0', distributions, False)
        m = experiment_df[experiment_df['condition'] == condition]['mDD'].iloc[0]
        s = experiment_df[experiment_df['condition'] == condition]['sDD'].iloc[0]
        distributions = _add_pdf_to_df(m, s, 'destiny', distributions, False)

    ax = sns.lineplot(data=distributions, x='time', y='y', hue='condition')
    ax.set_title(experiment)
    figname = FIGURE_MAP[experiment]
    fn = os.path.join(OUTPUT, f'{figname} cyton fits.pdf')
    plt.savefig(fn)
    # plt.show()
    plt.close()

def main():
  in_df = load_data()
  csv = os.path.join(OUTPUT, 'cyton-fits.csv')
  if os.path.exists(csv):
    out_df = pd.read_csv(csv)
  else:
    print('starting cyon fitting')
    out_df = process_data(in_df)
    out_df.to_csv(csv)

  print('plotting cyton fits')
  plot_data(in_df, out_df)


if __name__ == '__main__':
  main()
  print('done')

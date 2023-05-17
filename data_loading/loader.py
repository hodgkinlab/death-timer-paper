"""
Low level utility functions for reading CSVs into data frames and extracting metadata.
"""
import fnmatch
import os
import re
import pandas as pd

if __name__ == '__main__':
  import sys
  sys.path = sys.path[1:]

# from data.experiments import EXPERIMENTS

DATA_DIR = 'data'
RENAME_COLS = [
  ('Comp-PE-Cy7-A', 'BCL2'),
  ('Comp-Alexa Fluor 647-A', 'MCL1'),
  ('Comp-PE-A', 'BCLxL'),
  ('Comp-FITC-A', 'BIM'),
  ('Comp-780_60 (YG)-A', 'BCL2'),
  ('Comp-670_30 (R)-A', 'MCL1'),
  ('Comp-586_15 (YG)-A', 'BCLxL'),
  ('Comp-525_50 (B)-A', 'BIM'),
]
PROTEINS = ['BCL2', 'BCLxL', 'MCL1', 'BIM'] # Order is important!!
KEEP_CHANNELS = []
GENOTYPES = ['WT', 'BaxBak KO']

def load_file(fn):
  df = pd.read_csv(fn, index_col=None)

  cols = list(df.columns.values)
  col_rename = {}
  for old, new in RENAME_COLS:
    if old in cols:
      col_rename[old] = new

  df = df.rename(columns=col_rename)
  cols = list(df.columns.values)
  df = df.drop(columns=list(set(cols)-set(PROTEINS + KEEP_CHANNELS)))
  cols = list(df.columns.values)

  if set(cols) != set(PROTEINS + KEEP_CHANNELS):
    print(fn)
    print(df.columns)
    raise Exception('Unexpected columns')

  return df

def add_B_md(df, root, replicate):
  dps = root.split(os.sep)
  df['experiment'] = dps[2]
  df['genotype'] = dps[3]
  tm = dps[-1]
  m = re.search('[^\d]*(\d+)h\.*', tm)
  if len(dps) == 6:
    condition = dps[4]
  else:
    condition = '-'
  df['condition'] = condition
  df['time'] = int(m.group(1))
  df['replicate'] = replicate
  return df

def add_T_md(df, root, replicate):
  dps = root.split(os.sep)
  md = dps[-1]
  time = int(md[:-1])
  condition = dps[-2]
  genotype = dps[-3]

  df['time'] = time
  df['condition'] = condition
  df['genotype']  = genotype
  df['replicate'] = replicate

  return df

def _condition_renamer(df):
  condition = df['condition'].iat[0]
  rep = condition[-2:]
  if rep == '.1':
    rep = 2
    new_cond = condition[:-2]
  elif rep == '.2':
    rep = 3
    new_cond = condition[:-2]
  else:
    rep = 1
    new_cond = condition

  df['condition'] = new_cond
  df['replicate'] = rep

  return df

def load_T_counts(fn, conditions=None):
  df = pd.read_csv(fn)
  df2 = df.melt('time')
  df2 = df2.rename(columns={'variable': 'condition', 'value': 'counts.py'})
  df3 = df2.groupby(['condition']).apply(_condition_renamer).reset_index()
  if conditions is not None:
    df3 = df3[df3['condition'].isin(conditions)]

  return df3

def load_counts(experiment):
  fl = f'{experiment.directory()}*csv'
  dfs = []
  for root, _, files in os.walk(DATA_DIR):
    for csv in sorted(fnmatch.filter(files, fl)):
      df = pd.read_csv(os.path.join(root, csv))
      df['experiment'] = experiment.name
      if 'condition' not in df.columns:
        df['condition'] = '-'
      dfs.append(df)

  return pd.concat(dfs).reset_index()

def load_protein_levels(experiments, cell_type):
  dfs = []
  experiment_dirs = [e.directory() for e in experiments]
  for root, _, files in os.walk(os.path.join(DATA_DIR, cell_type)):
    replicate = 1
    for csv in sorted(fnmatch.filter(files, "export*.csv")):
      if experiments:
        experiment = root.split(os.sep)[2]
        if experiment not in experiment_dirs:
          continue
      df = load_file(os.path.join(root, csv))
      df = add_B_md(df, root, replicate)
      dfs.append(df)
      replicate += 1
  return pd.concat(dfs).reset_index()

def load_T_protein_levels(cell_dir, conditions=None):
  dfs = []
  data_dir = os.path.join(DATA_DIR, 'T cells', cell_dir)

  for root, _, files in os.walk(data_dir):
    replicate = 1
    for csv in sorted(fnmatch.filter(files, "export*.csv")):
      df = load_file(os.path.join(root, csv))
      df = add_T_md(df, root, replicate)
      if conditions is None or df['condition'].iat[0] in conditions:
        dfs.append(df)
      replicate += 1

  return pd.concat(dfs).reset_index()

def protein_validator(proteins):
  for p in proteins:
    if p not in PROTEINS:
      return False
  return True


if __name__ == '__main__':
  # df = load_T_counts('data/T cells/total cell number (corrected).csv')
  df = load_T_protein_levels('Small cells')
  print(df)
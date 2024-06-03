import pandas as pd

from data_loading.experiments import BcellExperiment, TcellExperiment

mr_70 = BcellExperiment('MR-70',    144, 144, survival_parameters={'-': {'survival_mean': 4.432441237, 'survival_std': 0.472062243}})
mr_92a = BcellExperiment('MR-92A',  144, 144, survival_parameters={'-': {'survival_mean': 4.516374903, 'survival_std': 0.543239955}})
mr_2022_6 = BcellExperiment('MR2022.6', 168, 168, survival_parameters={'-': {'survival_mean': 4.52141594808779, 'survival_std': 0.582651979380123}})

ac = {
  '0.1uM MCL1i': ['MCL1'],
  '1uM MCL1i': ['MCL1'],
  '0.1uM BCL2i': ['BCL2'],
  '1uM BCL2i': ['BCL2'],
  '1uM combination': ['BCL2', 'MCL1'],
}
survival_parameters = {
  '0uM': {'survival_mean': 4.86551950193066, 'survival_std': 0.457404775994445},
  '0.1uM MCL1i': {'survival_mean': 4.58095612008668, 'survival_std': 0.475210321622452},
  '1uM MCL1i': {'survival_mean': 4.27557699899987, 'survival_std': 0.446606521973684},
  '0.1uM BCL2i': {'survival_mean': 4.55446905857973, 'survival_std': 0.387783829781481},
  '1uM BCL2i': {'survival_mean': 4.50679135728438, 'survival_std': 0.345840763083007},
  '1uM combination': {'survival_mean': 4.02180889572659, 'survival_std': 0.484798639221193}
}

mr_95 = BcellExperiment('MR-95', 144, 144, default_condition='0uM', available_conditions=ac, survival_parameters=survival_parameters)
def _ex_div0():
  # The cohort numbers without div 0
  time = [48, 72, 96, 120, 144]
  cn = [
    [13029, 13622, 13757],
    [1377, 1412, 1523],
    [939, 981, 986],
    [615, 624, 604],
    [118, 132, 164],
    [43,  39, 45]
  ]
  cn = [sum(c)/len(c) for c in cn]
  cn = [c/cn[0] for c in cn]
  death = [1-c for c in cn]
  return pd.DataFrame({'time': time, 'death': death[1:], 'source': ['ex div 0']*len(time)})

mr_95.ex_div0 = _ex_div0()

ac = {
  '0.1uM MCL1i': ['MCL1'],
  '1uM MCL1i': ['MCL1'],
  '0.1uM BCL2i': ['BCL2'],
  '1uM BCL2i': ['BCL2'],
  '0.1uM combination': ['BCL2', 'MCL1'],
}
survival_parameters = {
  '0uM': {'survival_mean': 4.77311862102, 'survival_std': 0.346924697905687},
  '0.1uM MCL1i': {'survival_mean': 4.58299882367104, 'survival_std': 0.386467062759514},
  '1uM MCL1i': {'survival_mean': 4.20041232340473, 'survival_std': 0.486650815468128},
  '0.1uM BCL2i': {'survival_mean': 4.57760493989843, 'survival_std': 0.334459491278304},
  '1uM BCL2i': {'survival_mean': 4.50566612925524, 'survival_std': 0.303672066063934},
  '0.1uM combination': {'survival_mean': 4.27223186725899, 'survival_std': 0.46666337112474}
}
mr_2022_8 = BcellExperiment('MR2022.8', 144, 144, default_condition='0uM', available_conditions=ac, survival_parameters=survival_parameters)

mr_2023_5_unstim = BcellExperiment('MR2023.5 unstimulated', 71, 71, directory='MR2023.5',
                                   survival_parameters={'-': {'survival_mean': 3.484729434570168, 'survival_std': 0.962686659822257}},
                                   ensemble_xlim_hint=[-1e3, 1e3])

survival_parameters = {'stimulated': {'survival_mean': 4.757030480861173, 'survival_std': 0.48091217875822234},
                  'unstimulated': {'survival_mean': 3.062947118371283, 'survival_std': 1.0752873419882492}}
mr_2023_12 = BcellExperiment('MR23.12', 999, 999, available_conditions=['stimulated', 'unstimulated'],
                             default_condition='stimulated', directory='MR-23.12', survival_parameters=survival_parameters.copy(),
                             ensemble_xlim_hint=[-1e7, 1e7])

survival_parameters = {'stimulated': {'survival_mean': 4.7387811172670204, 'survival_std': 0.5668107030944444},
                  'unstimulated': {'survival_mean': 3.2930707696948596, 'survival_std': 0.9536752607617667}}
mr_2023_13 = BcellExperiment('MR23.13', 999, 999, available_conditions=['stimulated', 'unstimulated'],
                             default_condition='stimulated', directory='MR-23.13', survival_parameters=survival_parameters.copy(),
                             ensemble_xlim_hint=[-1e7, 1e7])

survival_parameters = {'stimulated': {'survival_mean': 4.734195069055054, 'survival_std': 0.5598017443823999},
                  'unstimulated': {'survival_mean': 3.3779371699357523, 'survival_std': 0.8671460101299735}}
mr_2023_14 = BcellExperiment('MR23.14', 999, 999, available_conditions=['stimulated', 'unstimulated'],
                             default_condition='stimulated', directory='MR-23.14', survival_parameters=survival_parameters.copy(),
                             ensemble_xlim_hint=[-1e7, 1e7])

B_EXPERIMENTS = [mr_70, mr_92a, mr_95, mr_2022_6, mr_2022_8, mr_2023_5_unstim, mr_2023_12, mr_2023_13, mr_2023_14]

mr_2022_11  = TcellExperiment('MR2022.11', 144, 144, default_condition='96h wash', directory='MR2022.11')
T_EXPERIMENTS = [mr_2022_11]

ALL_EXPERIMENTS = B_EXPERIMENTS + T_EXPERIMENTS
def experiment_from_name(name):
  for exp in ALL_EXPERIMENTS:
    if exp.name == name:
      return exp
  return None

def experiment_validator(experiments):
  if type(experiments) != list:
    experiments = [experiments]

  if len(experiments) == 0:
    return False

  for experiment in experiments:
    if experiment not in ALL_EXPERIMENTS:
      return False

  return True
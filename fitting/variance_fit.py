from matplotlib import pyplot as plt

from fitting.fit import Fit
from models.variance import VarianceModel
from data_loading.counts import Bcounts
from data_loading.levels import BProteinLevels


class VarianceFit(Fit):
  def __init__(self,
               experiment=None,
               fitting_algorithm='differential evolution',
               bounds=[(0, 50), (0, 50), (0, 50), (-50, 0)]):
    Fit.__init__(self,
                 experiment=experiment,
                 fitting_algorithm=fitting_algorithm,
                 bounds=bounds)

    do = BProteinLevels(experiment.name, replace_negative_with=1)
    gstds = do.gstds()
    gstds = gstds[(gstds['genotype'] == 'BaxBak KO') &
                  (gstds['time'] >= 72) &
                  (gstds['time'] <= experiment.t_max_protein)]

    survival_df = Bcounts(experiment.name).survival(survival_method='peak')
    self.survival_df = survival_df[(survival_df['time'] >= 72) & (survival_df['time'] <= experiment.t_max_counts)]
    self.survival = self.survival_df['fraction'].to_numpy()

    self.model = VarianceModel(variances=gstds,
                               fitting_algorithm=self.fitting_algorithm)
    self.run_label = f'Final fit: {self.experiment.name} - variance model'

  def predict(self, parms):
    class FakeResult:
      x = parms
    self.result = FakeResult()
    self.model.predict_survival(parms)
    return self.model.predicted_survival

  def plot(self, save_fig=True):
    fig, ax = plt.subplots(nrows=1, constrained_layout=True)
    self.model.plot_survival(self.survival_df, ax=ax)
    fig.suptitle(self.run_label)
    if save_fig:
      plt.savefig(f'{self.run_label}.png')
    plt.show()
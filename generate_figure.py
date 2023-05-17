import os
import argparse

FIGURES = {
  '1': 'cyton_fitting',
  '3': 'et_fitting',
  '4': 'et_fitting',
  '5': 'et_fitting',
  '6': 'et_fitting',
  'S6': 'et_fitting',
  'S7': 'et_fitting',
  'S8': 'et_fitting',
  'S9': 'et_fitting',
  'S11': 'et_fitting'
}

def parse_args():
  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--figures',
                      dest='figures',
                      choices=list(FIGURES.keys())+['all'],
                      nargs='*',
                      help='plot a specific figure or set of figures')

  return parser.parse_args()


def generate_fig(fig):
  if fig == '1':
    rc = os.system('python -u cyton_fitting.py')
  else:
    rc = os.system(f'python -u et_fitting.py --figure {fig}')
  if rc != 0:
    print(f'generation of fig {fig} may have failed. rc={rc}')

if __name__ == '__main__':
  opts = parse_args()
  figs = opts.figures

  if 'all' in figs:
    figs = FIGURES.keys()

  for fig in figs:
    print(f'generating fig {fig}')
    generate_fig(fig)

  print('all done')

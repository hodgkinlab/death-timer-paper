class Parameters:
  def __init__(self, parms={}):
    self._parms = parms.copy()

  def set(self, parm, value):
    self._parms[parm] = value

  def has_key(self, k):
    return k in self._parms

  def get(self, parm, first=False):
    if parm in self._parms:
      v = self._parms[parm]
      if first and type(v) == list:
        return v[0]
      else:
        return v
    raise Exception(f'No value set for {parm}')

  def validate(self, validation, exception=True, silent=False):
    valid = True
    for key, validator in validation.items():
      if key in self._parms:
        v = self._parms[key]
        if not validator(v):
          valid = False
          if not silent:
            print(f'{v} is not a valid value for {key}.')
      else:
        valid = False
        if not silent:
          print(f'{key} has not been set')

    for key in self._parms.keys():
      if key not in validation:
        if not silent:
          print(f'{key} is not a valid parameter.')
        valid = False

    if not valid and exception:
      raise Exception('Parameters are not valid.')

    return valid

  def copy(self):
    return Parameters(self._parms)
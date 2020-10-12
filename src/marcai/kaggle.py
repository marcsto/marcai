import os
def is_kaggle():
  """ Returns true if running on Kaggle environment (i.e. kernel).
      Useful if you want to set different paths to data directories when running
      locally vs on a Kaggle kernel
  """
  return 'PWD' in os.environ and os.environ['PWD'] == '/kaggle/working'

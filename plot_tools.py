import matplotlib.pyplot as plt

show_plots = False

def magic_plot(ax, plot_func, title, file, xlabel, ylabel, show=False):
  '''
  wrapper for generic plot
  '''

  ax.set_title(title)
  ax.set(xlabel=xlabel)
  ax.set(ylabel=ylabel)
  
  plot_func()

def plot_export(file=None, show=None):
  pass
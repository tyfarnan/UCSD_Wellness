import matplotlib.pyplot as plt

def magic_plot(ax, plot_func, title, file, xlabel, ylabel, show=False):
  '''
  wrapper for generic plot
  '''

  ax.set_title(title)
  ax.set(xlabel=xlabel)
  ax.set(ylabel=ylabel)
  
  plot_func()

from .smooth_hot import *
import matplotlib.pyplot as plt

def imshow(*args, **kwargs):
	kwargs['cmap'] = smooth_hot()
	if 'ax' not in kwargs.keys():
		kwargs['ax'] = plt.gca()
	ax = kwargs.pop('ax')
	ax.imshow(*args, **kwargs)
	ax.set_xticks([])
	ax.set_yticks([])

def plot_condfidence_1d(x, mu, stds, ax=None, **kwargs):
	x = x.reshape(-1)
	mu = mu.reshape(-1)
	stds = stds.reshape(-1)

	if ax is None:
		ax = plt.gca()
	ax.fill_between(x, mu + stds, mu - stds, **kwargs)

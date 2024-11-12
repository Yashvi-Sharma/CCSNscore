import plotly
import plotly.graph_objs as go
# import plotly.express as px

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import itertools
from sklearn.metrics import roc_curve, auc, det_curve


def make_confusion_matrix(df,classnames):
	# df = self.result
	conmat = np.zeros((len(classnames), len(classnames) + 1))
	for i, truelabel in enumerate(classnames):
		dff = df[df['ytest'] == truelabel]
		for j, predlabel in enumerate(classnames):
			conmat[i, j] = len(dff[dff['pred_class_names'] == predlabel])
		conmat[i, j + 1] = len(dff[dff['pred_class_names'] == 'ambi'])

	accuracy = np.trace(conmat) / np.sum(conmat).astype('float')
	misclass = 1 - accuracy
	precisions, recalls = [], []
	for i in range(len(classnames)):
		precisions.append(conmat[i, i] / np.sum(conmat[:, i]))
		recalls.append(conmat[i, i] / np.sum(conmat[i, :]))
	precisions = np.array(precisions)
	recalls = np.array(recalls)
	f1s = 1. / (0.5 * (1 / precisions + 1 / recalls))
	stats = [precisions, recalls, f1s]
	stats_df = pd.DataFrame(columns=classnames, data=stats)
	stats_df.index = ['Precision', 'Recall', 'f1score']
	return conmat, stats_df

def plot_confusion_matrix(cm,
						  target_names,
						  savename,
						  ax=None,
						  title='Confusion matrix',
						  cmap=None,
						  normalize=False,
						  noshowplot=False,
						  fontscale=1):
	"""
	given a sklearn confusion matrix (cm), make a nice plot

	Arguments
	---------
	cm:           confusion matrix from sklearn.metrics.confusion_matrix
	target_names: given classification classes such as [0, 1, 2]
				  the class names, for example: ['high', 'medium', 'low']
	savename:     name of the file to save the plot
	ax:           matplotlib Axes object
	title:        the text to display at the top of the matrix
	cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
				  see http://matplotlib.org/examples/color/colormaps_reference.html
				  plt.get_cmap('jet') or plt.cm.Blues
	normalize:    If False, plot the raw numbers
				  If True, plot the proportions
	noshowplot:   If False, show the plot
	fontscale:    Scale the font size of the text in the plot

	Citiation
	---------
	http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
	"""
	accuracy = np.trace(cm) / np.sum(cm).astype('float')
	misclass = 1 - accuracy
	if cmap is None:
		cmap = plt.get_cmap('Blues')

	if ax is None:
		fig, ax = plt.subplots(1,1,figsize=(2*len(target_names), 2*len(target_names)),tight_layout=True)
	ax.imshow(cm, interpolation='nearest', cmap=cmap)
	ax.set_title(title)

	if target_names is not None:
		tick_marks = np.arange(len(target_names))
		ax.set_yticks(tick_marks, target_names,fontsize=12*fontscale)
		if cm.shape[1] > len(target_names):
			tick_marks2 = np.arange(len(target_names)+1)
			ax.set_xticks(tick_marks2, np.concatenate([target_names,['Ambi']]), rotation=45, fontsize=12*fontscale)
		else:
			ax.set_xticks(tick_marks, target_names, rotation=45, fontsize=12*fontscale)

	if normalize:
		cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


	thresh = cm.max() / 1.5 if normalize else cm.max() / 2
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		if normalize:
			ax.text(j, i, "{:0.4f}".format(cm[i, j]),
					 horizontalalignment="center",
					 color="white" if cm[i, j] > thresh else "black",fontsize=16*fontscale)
		else:
			ax.text(j, i, "{:,}".format(cm[i, j]),
					 horizontalalignment="center",
					 color="white" if cm[i, j] > thresh else "black",fontsize=16*fontscale)


	ax.set_ylabel('True label',fontsize=15)
	ax.set_xlabel('Predicted label\nAccuracy={:0.1f}%'.format(accuracy*100),fontsize=12*fontscale)
	plt.tight_layout()
	if savename is not None:
		plt.savefig(savename,dpi=200)
	if not noshowplot:
		plt.show()
	return ax


def plot_histograms(df, classnames, savename=None, noshowplot=False, binsize=20, ax=None):
	"""
	Plot histograms of the predicted probabilities for each class
	:param df: dataframe containing the predicted probabilities
	:param classnames: list of class names
	:param savename: name of the file to save the plot
	:param noshowplot: if True, do not show the plot
	:param binsize: number of bins for the histogram
	:param ax: matplotlib Axes object
	:param plotstyle: type of plot (bar or step)
	:param alpha: transparency of the histogram
	:return: matplotlib Axes object
	"""
	ytest = np.array(df['ytest'])
	if ax is None:
		fig, ax = plt.subplots(1,len(classnames),figsize=(15,4))
	for i,colname in enumerate(classnames):
		truebool = ytest == colname
		bins = np.linspace(0,1,binsize)
		ax[i].hist(df.iloc[truebool][colname],label=f'{colname}',color='blue',bins=bins,alpha=0.8, density=True,
				   histtype='bar',rwidth=0.8)
		ax[i].hist(df.iloc[~truebool][colname],label=f'Not {colname}',color='red',bins=bins,alpha=0.8, density=True,
				   histtype='bar',rwidth=0.8)
		ax[i].legend()
		ax[i].set_xlabel('Probability',fontsize=15)
		ax[i].set_title(colname)
	ax[0].set_ylabel('Density',fontsize=15)
	if savename is not None:
		plt.savefig(savename,dpi=200)
	if not noshowplot:
		plt.show()
	return ax

# Plot ROC curve
def plot_roc_curve(df, classnames, savename=None, noshowplot=False, ax=None):
	"""
	Plot ROC curve for each class
	:param df: dataframe containing the predicted probabilities
	:param classnames: list of class names
	:param savename: name of the file to save the plot
	:param noshowplot: if True, do not show the plot
	:param ax: matplotlib Axes object
	:return: matplotlib Axes object
	"""
	ytest = np.array(df['ytest'])
	if ax is None:
		fig, ax = plt.subplots(1,1,figsize=(6,6))
	for i,classname in enumerate(classnames):
		fpr, tpr, _ = roc_curve(ytest==classname, df[classname])
		roc_auc = auc(fpr, tpr)
		ax.plot(fpr, tpr, label=f'{classname}')
	ax.set_xlabel('False Positive Rate',fontsize=15)
	ax.set_ylabel('True Positive Rate',fontsize=15)
	ax.set_title('ROC curve',fontsize=15)
	ax.legend()
	if savename is not None:
		plt.savefig(savename,dpi=200)
	if not noshowplot:
		plt.show()
	return ax


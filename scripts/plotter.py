import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import itertools
from sklearn.metrics import roc_curve, auc
import matplotlib.gridspec as gridspec
import os, json


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

def plot_reports(testtable, results, outpath, testmeta=None):
	resdfs = []
	for j, res in enumerate(results):
		resdfs.append(pd.read_csv(res))

	for i in range(len(testtable)):
		row = testtable.loc[i]
		## read spectra
		spec = pd.read_csv(row['specfilename'], sep='\s+', header=None, comment='#')
		## two plots in left column, one in right
		fig = plt.figure(figsize=(10, 6), tight_layout=True)
		gs = gridspec.GridSpec(2, 2, figure=fig)
		ax1 = fig.add_subplot(gs[0, 0])
		ax2 = fig.add_subplot(gs[1, 0])
		ax3 = fig.add_subplot(gs[:, 1])
		ax3.axis('off')
		fig.suptitle(f'{os.path.basename(row["specfilename"])}', fontsize=14)
		## plot spectrum in ax1
		if not pd.isna(row['z']):
			ax1.plot(np.array(spec[0]) / (1 + row['z']), np.array(spec[1]), color='black', label='z=' + str(row['z']))
			ax1.set_xlabel('Rest wavelength (Angstroms)', fontsize=12)
		else:
			ax1.plot(np.array(spec[0]), np.array(spec[1]), color='black', label='z=' + str(row['z']))
			ax1.set_xlabel('Observed wavelength (Angstroms)', fontsize=12)
		y1,y2 = ax1.get_ylim()
		### hydrogen lines
		ax1.axvline(6563, color='red', linestyle='--', alpha=0.7)
		ax1.axvline(4861, color='red', linestyle='--', alpha=0.7)
		ax1.annotate(r'H$\alpha$', xy=(6563, 0.9*y2), color='red', fontsize=9, rotation=90)
		ax1.annotate(r'H$\beta$', xy=(4861, 0.9*y2), color='red', fontsize=9, rotation=90)
		### helium lines
		ax1.axvline(5876, color='blue', linestyle='--', alpha=0.7)
		ax1.axvline(6678, color='blue', linestyle='--', alpha=0.7)
		ax1.axvline(7065, color='blue', linestyle='--', alpha=0.7)
		ax1.annotate(r'HeI', xy=(5876, 0.9*y2), color='blue', fontsize=9, rotation=90)
		ax1.annotate(r'HeI', xy=(6678, 0.9*y2), color='blue', fontsize=9, rotation=90)
		ax1.annotate(r'HeI', xy=(7065, 0.9*y2), color='blue', fontsize=9, rotation=90)
		ax1.set_ylabel('Normalized flux', fontsize=12)
		ax1.legend(fontsize=9)
		ax1.set_title(f'{row["name"]} spectrum', fontsize=12)

		## read light curve
		if not pd.isna(row['lcfilename']):
			try:
				lc = pd.read_csv(f'data/gp_lightcurves/{row["name"]}.csv')
				## plot light curve in ax2
				selr = lc['filter']=='r'
				selg = lc['filter']=='g'
				ax2.errorbar(lc[selr]['phase'],lc[selr]['mag'],lc[selr]['emag'], color='red', label='r', fmt='o', ls='')
				ax2.errorbar(lc[selg]['phase'],lc[selg]['mag'],lc[selg]['emag'], color='green', label='g', fmt='o', ls='')
				ax2.invert_yaxis()
				ax2.set_xlabel('Phase (days)', fontsize=12)
				ax2.set_ylabel('Magnitude', fontsize=12)
				ax2.set_title(f'{row["name"]} lightcurve', fontsize=12)
				ax2.legend(fontsize=9)
			except:
				ax2.axis('off')

		ax3.text(0.1, 0.9, f"CCSNscore results:", fontsize=12)
		for j,res in enumerate(results):
			channel = os.path.basename(res).split('_')[-2]
			layer = os.path.basename(res).split('_')[-3]
			resdf = resdfs[j]
			sub = resdf[resdf['id']==i].iloc[0]
			## annotate in ax3
			ax3.text(0.1, 0.9-(j+1)*0.05,
					 f"{layer} - {channel}: {sub['pred_class_names']}, "
					 f"P = {np.round(100*sub['pred_conf'],1)} %, "
					 f"P_unc = {np.round(100*sub['pred_std'],1)} %", fontsize=12)

		ax3.text(0.1, 0.58, f"--------------------------------------------------", fontsize=12)
		if testmeta is not None:
			metarow = testmeta.loc[i]
			snid = metarow['snid']
			sniascore = metarow['SNIascore']
			quality = metarow['quality']
			numsnid = metarow['numsnid']
			ax3.text(0.1, 0.55, f"Quality: {quality}, # SNID matches: {numsnid}", fontsize=12)
			snid = json.loads(snid.replace("'",'"'))
			ax3.text(0.1, 0.5, f"SNID: {snid['match']}, z={snid['redshift']}, age={snid['age']}, rlap={snid['rlap']}",
					 fontsize=12)
			sniascore = json.loads(sniascore.replace("'", '"'))
			ax3.text(0.1, 0.45, f"SNIascore: {sniascore['SNIascore']}, Unc={sniascore['SNIascore_err']}",
						 fontsize=12)
		plt.savefig(f'{outpath}{os.path.basename(row["specfilename"]).split(".")[0]}.png', dpi=200)
		plt.close()







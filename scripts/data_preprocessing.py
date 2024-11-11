import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, argparse, json
from sklearn import gaussian_process as gp
import warnings
from sklearn.preprocessing import MinMaxScaler
from scipy.interpolate import CubicSpline
from scipy.ndimage import median_filter
import tensorflow as tf

warnings.filterwarnings('ignore')

global gplc_path
gplc_path = '../data/gp_lightcurves/'

def pwd_for(a):
	"""
		Compute pairwise differences with for loops
	"""
	return np.array([a[j] - a[i] for i in range(len(a)) for j in range(i + 1, len(a))])

def compute_dmdt(jd, mag):
	'''
	Compute dmdt for lightcurve
	Input : jd array and magnitude array
	Output : dmdt (2d array)
	'''
	dmints = [-4.5, -3, -2.5, -2, -1.5, -1.25, -0.75, -0.5, -0.3, -0.2, -0.1, -0.05, 0,
			  0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.25, 1.5, 2, 2.5, 3, 4.5]
	dtints = [0.0, 1.0 / 24, 4./24, 0.25, 0.5, 0.75, 1, 2, 3, 6, 9, 12,
			  15, 24, 33, 48, 63, 72, 96, 126, 153, 180, 216, 255, 300]
	jd_diff = pwd_for(jd)
	mag_diff = pwd_for(mag)

	hh, ex, ey = np.histogram2d(jd_diff, mag_diff, bins=[dtints, dmints])
	dmdt = hh
	dmdt = np.transpose(dmdt)
	dmdt = 25*25*dmdt/np.sum(dmdt)
	dmdt[np.isnan(dmdt)] = 0
	return dmdt

def cont_removal(img,npoints=32,res=256):
	# Continuum removal for spectra (not used)
	x = np.linspace(3790, 9208.75, num=res, endpoint=True)
	y = img
	selinds = np.arange(0,res,npoints)
	cs = CubicSpline(x[selinds],y[selinds])
	ycont = cs(x)
	return y/ycont - 1.0

def spectrum_preproc(img,median_window=3):
	'''
	Input : 1d spectrum
	Output : Normalized 1d spec with median smoothening
	'''
	img = np.nan_to_num(img)
	img = img/np.median(img[50:200])
	img[(img<-5) | (img>100)] = 0
	img = np.nan_to_num(img)
	img = median_filter(img, median_window, mode='constant')
	return MinMaxScaler().fit_transform(np.array([img]).transpose())

def magtoflam(mag,emag,lam):
	# Convert magnitude to flux (erg/s/cm2/Angstrom)
	flam = 10**((mag-8.9)/(-2.5))/(3.34e4*(lam**2))
	eflam = abs((np.log(10)*flam/2.5)*emag)
	return flam, eflam

def flamtomag(flam,eflam,lam):
	# Convert flux to magnitude
	if flam!=0:
		mag = -2.5*np.log10(flam*3.34e4*(lam**2))+8.9
		emag = abs((np.log(10)*flam/2.5)*eflam)
	else:
		mag, emag = 0,0
	return mag, emag

def fix_shape_of_Xtrain(x):
	newx = [tf.convert_to_tensor(list(x[i])) for i in range(len(x))]
	return newx

# @jit()
def forecast_interpolation(df,fluxcol='mag',scale=1e16):
	'''
	Interpolate lightcurve using Gaussian Process
	:param df:
	:param scale:
	:return: interpolated lightcurve
	'''
	if len(df)==0:
		return df[['phase',fluxcol,f'e{fluxcol}','filter']]

	# ## if dataframe length is more than 300, take only 300 points
	# if len(df)>300:
	# 	df = df.iloc[0:300]

	x = np.array(df['phase']).reshape(-1,1)
	y = np.array(df[fluxcol]).reshape(-1,1)*scale
	yerr = np.array(df[f'e{fluxcol}']).reshape(-1,1)*scale
	kernel =  (1*gp.kernels.RBF(length_scale=100,length_scale_bounds=(50,150)) +
			   gp.kernels.WhiteKernel(noise_level=1e-4, noise_level_bounds=(1e-10,1e-1)))
	gplc = gp.GaussianProcessRegressor(kernel=kernel,n_restarts_optimizer=10)
	gplc.fit(x,y)
	# Print kernel length scale
	print(gplc.kernel_)

	xp = np.arange(x.min(),x.max()+0.5,0.5).reshape(-1,1)
	yp,yerrp = gplc.predict(xp,return_std=True)

	ndf = pd.DataFrame({'phase':xp.flatten(),fluxcol:yp.flatten()/scale,f'e{fluxcol}':yerrp.flatten()/scale,
						'filter':df['filter'].iloc[0]})
	ndf = ndf[(ndf[fluxcol]>np.min(df[fluxcol])-1) & (ndf[fluxcol]<np.max(df[fluxcol])+1)].reset_index(drop=True)

	return ndf

def plot_lc(interpdf, rawdf, objname, save=False):

	intdfr = interpdf[(interpdf['filter']=='r') & (interpdf['mag']/interpdf['emag']>=3)]
	intdfg = interpdf[(interpdf['filter']=='g') & (interpdf['mag']/interpdf['emag']>=3)]
	dfr = rawdf[rawdf['filter']=='r']
	dfg = rawdf[rawdf['filter']=='g']
	plt.figure(figsize=(6,4))
	plt.errorbar(dfr['phase'],dfr['mag'],yerr=dfr['emag'],marker='o',ls='',color='red')
	plt.errorbar(dfg['phase'],dfg['mag'],yerr=dfg['emag'],marker='o',ls='',color='green')
	plt.errorbar(intdfr['phase'],intdfr['mag'],yerr=intdfr['emag'],marker='.',ls='',color='darkred',elinewidth=0.5)
	plt.errorbar(intdfg['phase'],intdfg['mag'],yerr=intdfg['emag'],marker='.',ls='',color='darkgreen',elinewidth=0.5)
	plt.gca().invert_yaxis()
	plt.xlabel('Phase')
	plt.ylabel('Magnitude')
	plt.title(f'{objname}')
	if save:
		plt.savefig(f'../data/interpolation_figures/{objname}.png',dpi=200)
	plt.show()

def generate_lc(objname, lcpath, plot=False):
	## Load lightcurve
	if not os.path.exists(lcpath):
		print('COULD NOT GET LC!!! ',objname)
		return {'status':0,'name':objname,'message':'Could not load raw LC'}

	tbl = pd.read_csv(lcpath)
	## Check required columns
	required_columns = ['mjd','mag','emag','filter']
	if not all([col in tbl.columns for col in required_columns]):
		return {'status':0,'name':objname,'message':'Failed, required columns do not exist'}
	else:
		df = tbl[(~pd.isnull(tbl['mag'])) & (tbl['mag']!=99.00)].reset_index(drop=True)
		df = df[(df['mag']/df['emag']>=3)].reset_index(drop=True)
		if len(df)<2:
			return {'status':0,'name':objname,'message':'Failed, not enough detections in LC'}
		else:
			minmjd = min(df['mjd'])
			df['phase'] = df['mjd'] - minmjd
			## Setting up datafrane for GP forecast
			dfg = df[df['filter']=='g'].reset_index(drop=True)
			fluxg, efluxg = magtoflam(np.array(dfg['mag']),np.array(dfg['emag']),4805.0)
			dfg['flam'] = fluxg
			dfg['eflam'] = efluxg

			dfr = df[df['filter']=='r'].reset_index(drop=True)
			fluxr, efluxr = magtoflam(np.array(dfr['mag']),np.array(dfr['emag']),6390.0)
			dfr['flam'] = fluxr
			dfr['eflam'] = efluxr
			dfsave = pd.concat([dfg,dfr]).sort_values(by='phase').reset_index(drop=True)
			dfsave = dfsave[['phase','mag','emag','filter','flam','eflam']]
			dfsave.to_csv(f'../data/semiprocessed_lightcurves/{objname}.csv',index=False)

			ndfg = forecast_interpolation(dfg,fluxcol='mag',scale=1)
			ndfr = forecast_interpolation(dfr,fluxcol='mag',scale=1)

			# Convert mag to flam for dfg and dfr
			fluxg, efluxg = magtoflam(np.array(ndfg['mag']),np.array(ndfg['emag']),4805.0)
			fluxr, efluxr = magtoflam(np.array(ndfr['mag']),np.array(ndfr['emag']),6390.0)
			ndfg['flam'] = fluxg
			ndfg['eflam'] = efluxg
			ndfr['flam'] = fluxr
			ndfr['eflam'] = efluxr

			ndf = pd.concat([ndfg,ndfr])
			ndf = ndf[ndf['flam']/ndf['eflam']>=3].reset_index(drop=True)
			ndf.to_csv(f'{gplc_path}{objname}.csv',index=False)
			if plot:
				plot_lc(ndf, dfsave, objname, save=True)
			return {'status':1,'name':objname,'message':'Successfully saved processed LC'}

def run_lc_preprocessing(df):
	failed = []
	processed = []
	for i in range(len(df)):
		row = df.iloc[i]
		if row['name'] in np.array(processed):
			continue
		print(i,row['name'])
		res = generate_lc(row['name'], row['lcfilename'], plot=False)
		if res['status']==0:
			failed.append(res['name'])
		else:
			processed.append(res['name'])
	print('LC preprocessing failed for',', '.join(failed))
	return failed

import cudf
from concurrent.futures import ThreadPoolExecutor, as_completed

def run_lc_preprocessing_parallel(df):
    failed = []
    processed = []

    def process_row(row):
        if row['name'] in np.array(processed):
            return None
        print(row.name)
        res = generate_lc(row['name'], row['lcfilename'], plot=False)
        return res

    df = cudf.DataFrame.from_pandas(df)

    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(process_row, df.iloc[i]): i for i in range(len(df))}
        for future in as_completed(futures):
            res = future.result()
            if res:
                if res['status'] == 0:
                    failed.append(res['name'])
                else:
                    processed.append(res['name'])

    print('LC preprocessing failed for', ', '.join(failed))
    return failed

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('tablepaths', nargs='+', help='Run GP LC interpolation for these tables')
	parser.add_argument('--parallel', action='store_true', help='Run in parallel')
	args = parser.parse_args()

	for tablename in args.tablepaths:
		df = pd.read_csv(tablename)
		if args.parallel:
			failed = run_lc_preprocessing_parallel(df)
		else:
			failed = run_lc_preprocessing(df)
		print('Failed for',failed)


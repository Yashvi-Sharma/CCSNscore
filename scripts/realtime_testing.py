import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import keras
import glob2,os,subprocess,sys,argparse,json
from matplotlib.backends.backend_pdf import PdfPages
from ztfquery import marshal, fritz
from astropy.time import Time
import tune_train_test as ttt 
import data_preprocessing as dp
from numba import jit, vectorize, set_num_threads
set_num_threads(10)

def make_test_table(specnames,imgpath):
	dat = []
	for i,spec in enumerate(specnames):
		# try:
		sp = os.path.basename(spec).split('_')
		name = sp[-1].split('.')[0]
		print(name)
		if('ZTF' not in name):
			continue
		obj = fritz.api('GET',f'https://fritz.science/api/sources/{name}')
		photstats = fritz.api('GET',f'https://fritz.science/api/sources/{name}/phot_stat')
		z = obj['redshift']
		tps = [c['classification'] for c in obj['classifications']]
		if(len(np.unique(tps))==1):
			tp = np.unique(tps)[0]
		elif(len(np.unique(tps))==0):
			tp = 'unclassified'
		else:
			print('Multiple classes, skipping')
			# tp = input('Enter one class from above: ')
			tp = 'multiple'
		hostz = None
		datetime = sp[7][3:7]+'-'+sp[7][7:9]+'-'+sp[7][9:11]+' '+sp[8]+':'+sp[9]+':'+sp[10]
		specjd = Time(datetime,format='iso',scale='utc').jd
		maxjd = Time(photstats['peak_mjd_global'],format='mjd').jd
		phase = specjd - maxjd
		fname = spec
		inst = 'P60'
		row = [name,tp,z,hostz,specjd,maxjd,phase,fname,5,inst,None,None,None,None]
		print(row)
		print('------------------------')
		dat.append(row)
		# except:
		# 	print('Error in collecting data, check code!!!')
	rtdf = pd.DataFrame(columns=['name', 'type', 'z', 'hostz', 'specjd', 'maxjd',
			'phase', 'fname','flag', 'instrument', 'ntype', 'deljd_r', 'deljd_g', 'deljd'],
			data=dat)
	rtdf.to_csv(f'{imgpath}testtable.csv',index=False)
	return rtdf,f'{imgpath}testtable.csv'

if __name__ == '__main__':
	## Get date as argument in YYMMDD format
	parser = argparse.ArgumentParser()
	parser.add_argument('date',help='Enter date to run tests on in YYMMDD format')
	parser.add_argument('--no-deredshifting',action='store_true',default=False,help='Select to not de-redshift the spectra for training')
	parser.add_argument('--resolution',type=float,default=256,help='Number of spectral data points per spectrum')
	parser.add_argument('--lcphase',type=float,default=200,help='Phase of light curves from discovery to include')
	parser.add_argument('--overwrite',action='store_true',help='Select to overwrite existing test json')
	parser.add_argument('--copy',action='store_true',help='Select to copy final pdf to public folder')
	args = parser.parse_args()

	today = args.date
	if(len(today)!=8):
		print('Incorrect date format')
		sys.exit()

	## Make that date's folder
	if glob2.glob(f'../ML_spectra/realtime_{today}')==[]:
		os.mkdir(f'../ML_spectra/realtime_{today}')
	imgpath = f'../ML_spectra/realtime_{today}/'

	## Download data from pharos
	if glob2.glob(f'{imgpath}spec*.txt')==[]:
		subprocess.call(f'sshpass -p 4sedmr3dux rsync -av sedmdrp@minar.caltech.edu:/data/sedmdrp/redux/{today}/spec_auto*.txt {imgpath}',shell=True)
	print('Data download finished')

	## Make dataframe similar to testtable.csv
	rtdf,rtdfpath = make_test_table(sorted(glob2.glob(f'{imgpath}*.txt')),imgpath)

	## Process raw light curves, interpolation
	failed_targets = dp.run_lc_preprocessing(rtdf,'test')

	## Run testing
	testobj = ttt.Testing(rtdfpath,imgpath,dered=~args.no_deredshifting,res=args.resolution,lcphase=args.lcphase,
						  overwrite=args.overwrite)

	### Layer 2
	testdf = testobj.load_test(layer='2',channels=['spec','lcr','lcg','lcrdmdt','lcgdmdt'])
	testobj.test_model(testdf, modelpath='../ML_spectra/allgpmodels/gold/')
	print('Predicted layer 2')

	### Layer 3a
	testdf = testobj.load_test(layer='3a',channels=['spec','lcr','lcg','lcrdmdt','lcgdmdt'])
	testobj.test_model(testdf, modelpath='../ML_spectra/allgpmodels/gold/')
	print('Predicted layer 3a')

	### Layer 3b
	testdf = testobj.load_test(layer='3b',channels=['spec'])
	testobj.test_model(testdf, modelpath='../ML_spectra/allgpmodels/onlyspec/')
	print('Predicted layer 3b')

	## Plot into pdf

	print(f'Saved predictions in ../ML_spectra/realtime_{today}/prediction_plots.pdf')

	# ## Copy pdf to public folder on gayatri
	# if args.copy:
	# 	subprocess.call(f'sshpass -p rajom$yashvi7 rsync -av ../ML_spectra/realtime_{today}/prediction_plots.pdf yssharma@gayatri.caltech.edu:/home/yssharma/public_html/realtime_{today}_prediction_plots.pdf',shell=True)
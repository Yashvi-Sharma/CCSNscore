import numpy as np
import pandas as pd
import os, json, warnings, sys, glob2, argparse
import multiprocessing as mp
from scipy.interpolate import interp1d
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
tf.config.experimental.set_memory_growth(device=gpus[0], enable=True)
print(gpus)
from tensorflow import keras
import keras_tuner as kt

import plotter
from models import create_model
import data_preprocessing as dp

warnings.filterwarnings('ignore')

class Source:
    def __init__(self, sourcedata, sourcetype):
        self.id = int(sourcedata.name)
        self.sourcedata = dict(sourcedata)
        self.name = self.sourcedata['name']
        self.spec = None
        self.lcg = None
        self.lcr = None
        self.lcgdmdt = None
        self.lcrdmdt = None
        self.redshift = self.sourcedata['z']
        self.type = None
        self.label = None
        self.sourcetype = sourcetype

        if 'TYPE_COLUMN' in config.keys():
            if config['TYPE_COLUMN'] in self.sourcedata.keys():
                self.type = self.sourcedata[config['TYPE_COLUMN']]
                for key, value in config['LABELS'].items():
                    if self.type in value:
                        self.label = key
                        break
        if 'specjd' in self.sourcedata.keys():
            self.specjd = self.sourcedata['specjd']
        if 'maxjd' in self.sourcedata.keys():
            self.maxjd = self.sourcedata['maxjd']
        if 'instrument' in self.sourcedata.keys():
            self.instrument = self.sourcedata['instrument']


    def get_spectrum(self, dered=True, res=256, median_window=3):
        self.wavelength = np.linspace(4200, 9200, num=res, endpoint=True)
        try:
            spec = pd.read_csv(self.sourcedata['specfilename'], header=None, sep='\s+', comment='#')
        except:
            print('Error reading spectrum for ', self.sourcedata['name'], ' in ', self.sourcetype)
            return

        if dered and (not pd.isna(self.sourcedata['z'])):
            wavelengths = np.array(spec[0]).astype('float') / (1 + self.sourcedata['z'])
        else:
            wavelengths = np.array(spec[0]).astype('float')
        fluxes = np.array(spec[1]).astype('float')
        specdf = pd.DataFrame({'wave': wavelengths, 'flux': fluxes})
        specdf = specdf.drop_duplicates(subset='wave')
        try:
            fspec = interp1d(specdf['wave'], specdf['flux'], kind='cubic', fill_value=0, bounds_error=False,
                             assume_sorted=False)
            newfluxes = fspec(self.wavelength)
            self.spec = dp.spectrum_preproc(newfluxes, median_window)
            self.spec = self.spec.tolist()
            self.wavelength = self.wavelength.tolist()
        except:
            print('Error in spectrum interpolation for ', self.sourcedata['specfilename'], ' in ', self.sourcetype)

    def get_lightcurve(self, lcphase):
        if glob2.glob(config['GPLC_DIR'] + self.sourcedata['name'] + '.csv') == []:
            print('Processed light curve missing, trying to generate')
            if 'lcfilename' not in self.sourcedata.keys():
                print('No light curve file names exist, not using light curves')
            elif pd.isna(self.sourcedata['lcfilename']):
                print(f'No light curve file names exist for {self.sourcedata["name"]}, not using light curves')
            elif 'ZTF' not in self.sourcedata['name']:
                print(f'Light curve file names exist for {self.sourcedata["name"]}, but not ZTF, not using light curves')
            else:
                print(self.sourcedata['lcfilename'])
                res = dp.generate_lc(self.sourcedata['name'], self.sourcedata['lcfilename'], config['GPLC_DIR'],
                                     plot=False)
                print(res['message'])

        if glob2.glob(config['GPLC_DIR'] + self.sourcedata['name'] + '.csv') == []:
            phaserange = np.arange(0, lcphase, 1)
            self.lcr = np.zeros([lcphase, 2]).tolist()
            self.lcg = np.zeros([lcphase, 2]).tolist()
            self.lcrdmdt = dp.compute_dmdt(phaserange, np.repeat(20.5, len(phaserange)))[..., tf.newaxis].tolist()
            self.lcgdmdt = dp.compute_dmdt(phaserange, np.repeat(20.5, len(phaserange)))[..., tf.newaxis].tolist()
            return
        else:
            lc = pd.read_csv(config['GPLC_DIR'] + self.sourcedata['name'] + '.csv')
            if 'flam' not in lc.columns:
                res = dp.generate_lc(self.sourcedata['name'], self.sourcedata['lcfilename'], config['GPLC_DIR'],
                                     plot=False)
                print(res['message'])
                lc = pd.read_csv(config['GPLC_DIR'] + self.sourcedata['name'] + '.csv')
            phaserange = np.arange(0, lcphase + 1, 1)
            lcr = lc[(lc['filter'] == 'r')]
            lcg = lc[(lc['filter'] == 'g')]
            dfr = lcr[['phase', 'flam', 'eflam']].groupby(pd.cut(lcr['phase'], phaserange)).mean()
            dfg = lcg[['phase', 'flam', 'eflam']].groupby(pd.cut(lcg['phase'], phaserange)).mean()
            dfr.loc[pd.isnull(dfr['flam']), 'flam'] = 0
            dfr.loc[pd.isnull(dfr['eflam']), 'eflam'] = 0
            dfg.loc[pd.isnull(dfg['flam']), 'flam'] = 0
            dfg.loc[pd.isnull(dfg['eflam']), 'eflam'] = 0

            flamarr = np.hstack([np.array(dfr['flam']), np.array(dfg['flam'])])*1e16  ## IMPORTANT
            lcflux = MinMaxScaler().fit_transform(flamarr[..., tf.newaxis]).flatten()
            lcrflux = lcflux[0:lcphase]
            lcrflux_err = np.nan_to_num(
                [lcrflux[i] * np.array(dfr['eflam'])[i] / np.array(dfr['flam'])[i] for i in range(len(lcrflux))])
            self.lcr = np.nan_to_num(np.vstack([lcrflux, lcrflux_err])).transpose().tolist()

            lcgflux = lcflux[lcphase:]
            lcgflux_err = np.nan_to_num(
                [lcgflux[i] * np.array(dfg['eflam'])[i] / np.array(dfg['flam'])[i] for i in range(len(lcgflux))])
            self.lcg = np.nan_to_num(np.vstack([lcgflux, lcgflux_err])).transpose().tolist()

            self.lcrdmdt = dp.compute_dmdt(list(lcr['phase']), list(lcr['mag']))[..., tf.newaxis].tolist()
            self.lcgdmdt = dp.compute_dmdt(list(lcg['phase']), list(lcg['mag']))[..., tf.newaxis].tolist()
            return


class BaseSet:
    def __init__(self, tablepath, dered=True, res=256, lcphase=200, median_window=3):
        self.table = pd.read_csv(tablepath)
        self.dered = dered
        self.res = res
        self.lcphase = lcphase
        self.median_window = median_window
        self.wavelength = np.linspace(3700, 9200, num=res, endpoint=True).tolist()
        self.sourcetype = 'base'

    def create_set(self):
        global make_source
        def make_source(source, dered, res, lcphase, median_window):
            if glob2.glob(source.sourcedata['specfilename']) != []:
                source.get_spectrum(dered, res, median_window)
                if source.spec is not None:
                    source.get_lightcurve(lcphase)
                    source.__dict__.pop('sourcedata')
                    return source.__dict__
                else:
                    return {}
            else:
                print('Spectra file missing for '+source.sourcedata['name'])
                return {}

        savename = imgpath + self.sourcetype + 'data.json'
        with mp.Manager() as manager:
            with manager.Pool(6) as pool:
                args = [(Source(self.table.loc[i], self.sourcetype), self.dered, self.res, self.lcphase,
                         self.median_window) for i in range(len(self.table))]
                listdict = pool.starmap(make_source, args)
        listdict = [l for l in listdict if l != {}]
        self.wavelength = listdict[0]['wavelength']
        with open(savename, 'w') as savefile:
            json.dump(listdict, savefile)
        print(f'Created {self.sourcetype} dataset')


class TrainSet(BaseSet):
    def __init__(self, tablepath, dered=True, res=256, lcphase=200, median_window=3, overwrite=False):
        super().__init__(tablepath, dered, res, lcphase, median_window)
        self.sourcetype = 'train'
        if glob2.glob(imgpath + 'traindata.json') == [] or overwrite:
            self.create_set()

    def augment_train(self, typename, traindf, auglimit=5000):
        global make_augmented

        def make_augmented(dummy_index, rows, dered, res, wavelength):
            row0 = rows.loc[0]
            row1 = rows.loc[1]
            wt = np.random.rand(1)[0]
            if (typename == 'CV'):
                z0, z1 = 0, 0
            else:
                z0 = row0['redshift']
                z1 = row1['redshift']
            z = wt * z0 + (1 - wt) * z1
            if dered:
                flux = wt * np.array(row0['spec'])[:, 0] + (1 - wt) * np.array(row1['spec'])[:, 0]
            else:
                dewave0 = np.array(wavelength) / (1 + z0)
                dewave1 = np.array(wavelength) / (1 + z1)
                minwv = np.max([dewave0[0], dewave1[0]])
                maxwv = np.min([dewave0[-1], dewave1[-1]])
                tempwavelengths = np.linspace(minwv, maxwv, num=res, endpoint=True)
                fspec0 = interp1d(wavelength, np.array(row0['spec'])[:, 0], kind='cubic', fill_value=0,
                                  bounds_error=False, assume_sorted=False)
                flux0 = fspec0(tempwavelengths)
                fspec1 = interp1d(wavelength, np.array(row1['spec'])[:, 0], kind='cubic', fill_value=0,
                                  bounds_error=False, assume_sorted=False)
                flux1 = fspec1(tempwavelengths)
                flux = wt * flux0 + (1 - wt) * flux1
                wave = tempwavelengths * (1 + z)
                fspec = interp1d(wave, flux, kind='cubic', fill_value=0, bounds_error=False, assume_sorted=False)
                flux = fspec(wavelength)
            if flux is None:
                return {}
            spec = dp.spectrum_preproc(flux)
            spec = spec.tolist()
            lcr = np.array(row0['lcr'])
            lcg = np.array(row0['lcg'])
            if (z0 != 0 and z != 0):
                lcr = lcr * (z0 * z0 * (1 + z0)) / (z * z * (1 + z))
                lcg = lcg * (z0 * z0 * (1 + z0)) / (z * z * (1 + z))

            flamarr = np.hstack([lcr[:, 0], lcg[:, 0]])
            lcflux = MinMaxScaler().fit_transform(flamarr[..., tf.newaxis]).flatten()
            lcrflux = lcflux[0:self.lcphase]
            lcrflux_err = np.nan_to_num([lcrflux[i] * lcr[i, 1] / lcr[i, 0] for i in range(len(lcrflux))])
            fakelcr = np.nan_to_num(np.vstack([lcrflux, lcrflux_err])).transpose().tolist()

            lcgflux = lcflux[self.lcphase:]
            lcgflux_err = np.nan_to_num([lcgflux[i] * lcg[i, 1] / lcg[i, 0] for i in range(len(lcgflux))])
            fakelcg = np.nan_to_num(np.vstack([lcgflux, lcgflux_err])).transpose().tolist()

            fakedict = {'id': dummy_index, 'name': row0['name'] + '_' + row1['name'],
                        'spec': spec,
                        'lcg': fakelcg, 'lcr': fakelcr, 'lcgdmdt': row0['lcgdmdt'],
                        'lcrdmdt': row0['lcrdmdt'],
                        'type': typename, 'redshift': z, 'specjd': None, 'maxjd': None, 'instrument': None,
                        'label': row0['label'], 'sourcetype': 'fake'}
            return fakedict

        tdf = traindf[~pd.isnull(traindf['redshift'])].reset_index(drop=True)
        tdf['phase'] = tdf['specjd'] - tdf['maxjd']
        savename = imgpath + typename + '_fakedata.json'

        if (typename == 'Ib') | (typename == 'Ic') | (typename == 'Ic-BL') | (typename == 'Ia'):
            phasegroups = [[-100, 0], [0, 50]]
        elif (typename == 'IIP') | (typename == 'IIn') | (typename == 'II'):
            phasegroups = [[-100, 0], [0, 100]]
        elif typename == 'IIb':
            phasegroups = [[-100, -5], [0, 50]]

        subdf0 = tdf[(tdf['type'] == typename) & (tdf['phase'] >= phasegroups[0][0]) & (
                tdf['phase'] <= phasegroups[0][1])].reset_index(drop=True)
        subdf1 = tdf[(tdf['type'] == typename) & (tdf['phase'] >= phasegroups[1][0]) & (
                tdf['phase'] <= phasegroups[1][1])].reset_index(drop=True)
        print('Found ', len(subdf0), ' objects in ', phasegroups[0], ' and ', len(subdf1), ' objects in ',
              phasegroups[1])
        indlist0 = subdf0.index
        indlist1 = subdf1.index
        auglist = []
        if len(indlist0) != 0:
            for ind in range(0, int(0.3 * auglimit)):
                rows = subdf0.loc[np.random.choice(indlist0, 2, p=None, replace=False)].reset_index(drop=True)
                fakedict = make_augmented(ind, rows, self.dered, self.res, self.wavelength)
                if fakedict != {}:
                    auglist.append(fakedict)
            augstartind = int(0.3 * auglimit)
        else:
            augstartind = 0
        if len(indlist1) != 0:
            for ind in range(augstartind, auglimit):
                rows = subdf1.loc[np.random.choice(indlist1, 2, p=None, replace=False)].reset_index(drop=True)
                fakedict = make_augmented(ind, rows, self.dered, self.res, self.wavelength)
                if fakedict != {}:
                    auglist.append(fakedict)

        with open(savename, 'w') as savefile:
            json.dump(auglist, savefile)
        print('Augmented ' + typename + ' with ' + str(len(auglist)) + ' fake objects')


class TestSet(BaseSet):
    def __init__(self, tablepath, dered=True, res=256, lcphase=200, median_window=3, overwrite=False):
        super().__init__(tablepath, dered, res, lcphase, median_window)
        self.sourcetype = 'test'
        if glob2.glob(imgpath + 'testdata.json') == [] or overwrite:
            self.create_set()


class Training:
    def __init__(self, trainpath, dered=True, res=256, lcphase=200, median_window=3, augtypes=[], auglimit=2000,
                 overwrite=False):
        self.trainpath = trainpath
        self.dered = dered
        self.res = res
        self.lcphase = lcphase
        self.median_window = median_window
        self.auglimit = auglimit
        self.augtypes = augtypes

        trset = TrainSet(self.trainpath, self.dered, self.res, self.lcphase, self.median_window, overwrite)
        tdf = pd.read_json(open(imgpath + 'traindata.json', 'r'))
        for typename in self.augtypes:
            if glob2.glob(imgpath + typename + '_fakedata.json') == [] or overwrite:
                trset.augment_train(typename, tdf, self.auglimit)

    def load_train(self, channels, augment=True, augconstant=2000):
        self.augment = augment
        self.augconstant = augconstant
        self.channels = channels

        traindf = pd.read_json(open(imgpath + 'traindata.json', 'r'))
        traindf = traindf[~pd.isna(traindf['label'])]

        if self.augment:
            for typename in self.augtypes:
                augnum = self.augconstant - len(traindf[traindf['type'] == typename])
                fakedata = pd.read_json(open(imgpath + typename + '_fakedata.json', 'r'))[:augnum]
                traindf = pd.concat([traindf, fakedata]).reset_index(drop=True)

        traindf = traindf.sample(frac=1).reset_index(drop=True)
        print('Samples in training set: ', len(traindf))
        return traindf

    def get_models(self, weights=False, lr=0.001):
        layermodels = {}
        for modelname in self.modelnames:
            model = keras.models.load_model(f'{imgpath}BC_{modelname}', compile=False)
            if not weights:
                newmodel = keras.models.clone_model(model)
            else:
                newmodel = model
            optim = keras.optimizers.Adam(learning_rate=lr)
            newmodel.compile(optimizer=optim,
                             loss=tf.keras.losses.BinaryCrossentropy(), metrics=['accuracy', 'Precision'])
            layermodels[modelname] = newmodel
        return layermodels

    def train(self, traindf, modelnames, loadmodel=False, withweights=False, lr=0.001, patience=7,
              epochs=20, batchsize=32):
        self.modelnames = modelnames
        self.models = {}
        inp_shapes = [np.shape(x[0]) for x in
                      np.array(traindf[['spec', 'lcr', 'lcg', 'lcrdmdt', 'lcgdmdt']]).transpose()]
        print(inp_shapes)

        if loadmodel:
            if withweights:
                layermodels = self.get_models(weights=True, lr=lr)
            else:
                layermodels = self.get_models(weights=False, lr=lr)
        else:
            layermodels = None

        for num, modelname in enumerate(self.modelnames):
            print('Training model ', modelname)
            # if modelname=='Ia':
            #     continue
            xtrain = traindf[np.array(self.channels[modelname])].transpose().values

            bin_ytrain = np.array(traindf['label'] == modelname).astype(float)
            class_inds = np.where(bin_ytrain == 1)[0]
            notclass_inds = np.where(bin_ytrain == 0)[0]
            # print(np.unique(bin_ytrain, return_counts=True))

            if layermodels is None:
                model = create_model(inp_shapes, 1, np.array(self.channels[modelname]), mc=False,
                                     tuning=False)
                optim = tf.keras.optimizers.Adam(lr=lr)
                model.compile(optimizer=optim, loss=tf.keras.losses.BinaryCrossentropy(),
                              metrics=['accuracy', tf.keras.metrics.Precision()])
            else:
                model = layermodels[modelname]
            earlystop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience,
                                                                  restore_best_weights=True)
            tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=f"{imgpath}logs_{modelname}")
            np.random.seed(7)

            if len(notclass_inds) >= 0.8 * len(class_inds):
                count = 0
                while (len(notclass_inds) >= 0.8 * len(class_inds) and count < 2):
                    chosen_inds = np.random.choice(notclass_inds, size=len(class_inds))
                    chosen_indlocs = [np.where(notclass_inds == ci)[0][0] for ci in chosen_inds]
                    notclass_inds = np.delete(notclass_inds, chosen_indlocs)
                    totrain_inds = shuffle(np.concatenate([class_inds, chosen_inds]), random_state=7)
                    xtrain_sub = []
                    for i, x in enumerate(xtrain):
                        xtrain_sub.append(x[totrain_inds])
                    ytrain_sub = bin_ytrain[totrain_inds]
                    xtrain_sub = dp.fix_shape_of_Xtrain(xtrain_sub)
                    model.fit(xtrain_sub, ytrain_sub, validation_split=0.33,
                              epochs=epochs, batch_size=batchsize, callbacks=[earlystop_callback, tensorboard_callback],
                              verbose=1, shuffle=True, use_multiprocessing=True, workers=6)
                    count = count + 1
            else:
                model.fit(dp.fix_shape_of_Xtrain(xtrain), bin_ytrain, epochs=epochs, batch_size=batchsize,
                          validation_split=0.33,callbacks=[earlystop_callback, tensorboard_callback],
                          verbose=1, shuffle=True, use_multiprocessing=True, workers=6)

            self.models[modelname] = model
            model.save(f'{imgpath}RT_{modelname}')
            print('Model saved at ', f'{imgpath}RT_{modelname}')

    def tune(self, traindf, patience=5, epochs=10, batchsize=32):
        inp_shapes = [np.shape(x[0]) for x in
                      np.array(traindf[['spec', 'lcr', 'lcg', 'lcrdmdt', 'lcgdmdt']]).transpose()]

        for num, modelname in enumerate(self.modelnames):
            xtrain = traindf[np.array(self.channels[modelname])].transpose().values

            bin_ytrain = np.array(traindf['label'] == modelname).astype(float)
            class_inds = np.where(bin_ytrain == 1)[0]
            notclass_inds = np.where(bin_ytrain == 0)[0]
            np.random.seed(7)
            chosen_inds = np.random.choice(notclass_inds, size=len(class_inds))
            totrain_inds = shuffle(np.concatenate([class_inds, chosen_inds]), random_state=7)
            xtrain_sub = []
            for i, x in enumerate(xtrain):
                xtrain_sub.append(x[totrain_inds])
            ytrain_sub = bin_ytrain[totrain_inds]
            xtrain_sub = dp.fix_shape_of_Xtrain(xtrain_sub)

            model_for_tuner = create_model(inp_shapes, 1, np.array(self.channels[modelname]), mc=False,
                                           tuning=True)
            tuner = kt.BayesianOptimization(model_for_tuner,
                                            objective='val_accuracy',
                                            max_trials=10,
                                            directory=f'{imgpath}tuner',
                                            project_name=f'{modelname}')
            stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience,
                                                          restore_best_weights=True)
            print('Starting tuner')
            tuner.search(xtrain_sub, ytrain_sub, epochs=epochs, batch_size=batchsize,
                         validation_split=0.3, callbacks=[stop_early])

            # Get the optimal hyperparameters
            best_hps = tuner.get_best_hyperparameters()[0]
            best_model = tuner.get_best_models()[0]

            best_model.save(f'{imgpath}BC_{modelname}')
            print(f'Saved tuned model for model {modelname} at {imgpath}BC_{modelname}')


class Testing:
    def __init__(self, testpath, dered=True, res=256, lcphase=200, median_window=3, overwrite=False):
        self.testpath = testpath
        self.dered = dered
        self.res = res
        self.lcphase = lcphase
        self.median_window = median_window
        self.wavelength = np.linspace(3700, 9200, num=res, endpoint=True).tolist()

        TestSet(self.testpath, self.dered, self.res, self.lcphase, self.median_window, overwrite)

    def load_test(self, channels):
        self.channels = channels
        testdf = pd.read_json(open(imgpath + 'testdata.json', 'r'))
        if len(testdf[pd.isna(testdf['type'])])!=len(testdf):
            testdf = testdf[~pd.isna(testdf['label'])]
        testdf = testdf.reset_index(drop=True)
        print('Samples in testing set: ', len(testdf))
        return testdf

    def get_trained_models(self, modelpaths):
        layermodels = {}
        for modelname in self.modelnames:
            model = keras.models.load_model(modelpaths[modelname], compile=False)
            optim = keras.optimizers.Adam(learning_rate=0.001)
            model.compile(optimizer=optim, loss=tf.keras.losses.BinaryCrossentropy(),
                          metrics=['accuracy', 'Precision'])
            layermodels[modelname] = model
        return layermodels

    def decide_finalclass(self, df, testdf):
        df['ytest'] = list(testdf['label'])
        df['pred_class_names'] = np.repeat(None, len(df))
        df['pred_conf'] = np.repeat(None, len(df))
        df['pred_std'] = np.repeat(None, len(df))

        for i in range(len(df)):
            row = df.loc[i][self.modelnames]
            row_unc = df.loc[i][[x + '_std' for x in self.modelnames]]
            srow = row[np.argsort(-row)]
            srow_unc = row_unc[np.argsort(-row)]
            fc = srow.index[0]
            ps = srow[0]
            ps_unc = srow_unc[0]
            if ((srow[0]-srow_unc[0]) - (srow[1]+srow_unc[1]) < 0.01):
                fc = 'ambi'
                ps = 0
                ps_unc = 0

            df.loc[i, 'pred_class_names'] = fc
            df.loc[i, 'pred_conf'] = ps
            df.loc[i, 'pred_std'] = ps_unc
        return df

    def plot_cm_and_hist(self, noshowplot):
        conmat, stats = plotter.make_confusion_matrix(self.result, self.modelnames)
        plotter.plot_confusion_matrix(conmat, self.modelnames, savename=f'{imgpath}{taskname}_conmat.png',
                                      noshowplot=noshowplot)
        plotter.plot_histograms(self.result, self.modelnames, savename=f'{imgpath}{taskname}_histograms.png',
                                noshowplot=noshowplot)
        plotter.plot_roc_curve(self.result, self.modelnames, savename=f'{imgpath}{taskname}_roc.png',
                               noshowplot=noshowplot)
        return stats

    def mc_dropout_predict_batch(self, model, xtest, num_samples=100):
        predictions = [model(xtest, training=True).numpy().flatten() for i in range(num_samples)]
        return np.array(predictions)

    def test(self, testdf, modelnames, modelpaths=None, noshowplot=False):
        self.modelnames = modelnames

        class_probabs = {}
        if modelpaths is None:
            modelpaths = {}
            for modelname in self.modelnames:
                modelpaths[modelname] = f'{imgpath}RT_{modelname}'
        layermodels = self.get_trained_models(modelpaths)

        for num, modelname in enumerate(self.modelnames):
            xtest = testdf[np.array(self.channels[modelname])].transpose().values
            xtest = dp.fix_shape_of_Xtrain(xtest)
            model = layermodels[self.modelnames[num]]
            ## Predict with MC dropout enabled
            scores = self.mc_dropout_predict_batch(model, xtest, num_samples=20)
            print(scores.shape)
            predictions = np.mean(scores, axis=0)
            # print(predictions.shape)
            uncertainties = np.std(scores, axis=0)
            # print(uncertainties.shape)
            class_probabs[modelname] = predictions
            class_probabs[modelname + '_std'] = uncertainties

        df = pd.DataFrame(class_probabs)
        df['id'] = testdf['id']
        df = self.decide_finalclass(df, testdf)
        self.result = df
        self.result.to_csv(f'{imgpath}{taskname}_results.csv', index=False)
        if len(testdf[pd.isna(testdf['type'])])!=len(testdf):
            self.stats = self.plot_cm_and_hist(noshowplot=noshowplot)


if __name__ == '__main__':

    # Read arguments
    parser = argparse.ArgumentParser(description='Train or test models for SEDM data')
    parser.add_argument('config', type=str, help='config file path w.r.t. CCSNscore/')
    args = parser.parse_args()

    # Read config file
    global config, imgpath, taskname
    config = json.load(open(args.config, 'r'))
    print(config)
    imgpath = config['OUTPUT_DIR']
    taskname = config['TASKNAME']

    if imgpath[-1] != '/':
        imgpath = imgpath + '/'
    if not os.path.exists(imgpath):
        os.makedirs(imgpath)

    # Copy config file to output directory
    os.system(f'cp {args.config} {imgpath}config_{taskname}')

    if config['MODE'] == 'tune' or config['MODE'] == 'train':
        print(f'Begin program, loading data')
        trainobj = Training(config['TRAINTABLEPATH'], dered=config['DEREDSHIFT'], res=config['RESOLUTION'],
                            lcphase=config['LCPHASE'], median_window=config['MEDIAN_WINDOW'],
                            augtypes=config['TYPES_TO_AUGMENT'], auglimit=config['AUGLIMIT'],
                            overwrite=config['OVERWRITE'])
        testobj = Testing(config['TESTTABLEPATH'], dered=config['DEREDSHIFT'], res=config['RESOLUTION'],
                          lcphase=config['LCPHASE'], median_window=config['MEDIAN_WINDOW'],
                          overwrite=config['OVERWRITE'])

        print(f'Starting task {config["TASKNAME"]}')
        traindf = trainobj.load_train(channels=config['CHANNELS'], augment=config['AUGMENT'],
                                      augconstant=config['AUGMENT_CONSTANT'])
        testdf = testobj.load_test(channels=config['CHANNELS'])

        if config['MODE']=='tune':
            print(f'Tuning is selected, starting keras tuner')
            trainobj.tune(traindf, patience=config['PATIENCE'], epochs=config['EPOCHS'], batchsize=config['BATCHSIZE'])
            print(f'Finished tuning, exiting')

        elif config['MODE']=='train':
            trainobj.train(traindf, config['MODELS'], loadmodel=config['LOAD_TUNED_MODELS'],
                           withweights=config['LOAD_WEIGHTS'], lr=config['LEARNING_RATE'], patience=config['PATIENCE'],
                           epochs=config['EPOCHS'], batchsize=config['BATCHSIZE'])
            print(f'Finished training, now testing')
            testobj.test(testdf, config['MODELS'], modelpaths=None, noshowplot=config['NOSHOWPLOT'])
            print(f'Finished testing, exiting')

    elif config['MODE'] == 'test':
        print(f'Begin program, loading data')
        testobj = Testing(config['TESTTABLEPATH'], dered=config['DEREDSHIFT'], res=config['RESOLUTION'],
                          lcphase=config['LCPHASE'], median_window=config['MEDIAN_WINDOW'],
                          overwrite=config['OVERWRITE'])

        print(f'Starting task {config["TASKNAME"]}')
        testdf = testobj.load_test(channels=config['CHANNELS'])

        if 'TRAINED_MODELPATH' in config.keys():
            if config['TRAINED_MODELPATH'] is None:
                print(f'No trained model path specified, will look for models in default place')
                modelpaths = None
            else:
                print(f'Trained model path is different')
                modelpaths = config['TRAINED_MODELPATH']
        else:
            modelpaths = None
        testobj.test(testdf, config['MODELS'], modelpaths=modelpaths, noshowplot=config['NOSHOWPLOT'])
        print(f'Finished testing, exiting')
    else:
        print('Invalid mode')

    print('Finished with specified task, exiting')
    sys.exit(0)

# SEDM-spectral-classifier

A deep learning based hierarchical classifier program for classification of core collapse supernovae (SN) from ultra low resolution spectra (SEDM) and ZTF r and g band lightcurves.

## Data and pre-processing
All training and testing data are in ML_spectra/, spectra/ contains the ascii SEDM and openSN catalog spectral files, lightcurves/ contains curated ZTF lightcurve csv files for SNe. The SN list and metadata are in largetraintable.csv and largetesttable.csv, the column 'type' has oriignal classification from user GUIs (fritz and marshal), the column 'ntype' has the simplified type used to create labels.

## How to use
tune_train_test.py is the main script that handles lightcurve pre-processing, creates training and testing samples, tunes and trains the model.
realtime_testing.py is used for making predictions on new data, and it also handles the preprocessing of the new data and nightly data from SEDM.

The code allows for the following options related to data processing. `--no-deredshifting` disables changing spectral range from observed to rest wavelengths. `--resolution` specifies the number of data points per spectrum, default is 256, increasing it only makes sense for openSN spectra. `--lcphase` specifies the length of light curve in days to include. `--auglimit` specifies the number of augmented spectra that should be created. 

The following options pertain to the model training. There are 3 hierarchical levels called layers, the top layer (1) deals with classification between SN,AGN and CV. The second layer (2) classifies between hydrogen rich vs hydrogen poor SNe, and the third layer has two branches, one for H-rich and one for H-poor and deals with sub-type classifications. `--layers` option can be used to choose to train one or more than one layers. `--exclude-augmented` option is used to exclude all augmented data but it will lead to highly unbalanced training set. 

In each layer, for each label there is one binary classifier model. The final classes are decided based on the predictions of all the parallel binary classifiers in the layer. if any two type probabilities are too close to each other, the sample is labeled ambiguous. Each binary classifier model has channels for multiple inputs and any of the input channels can be disabled. `--channels` option allows you to choose which inputs to use. The model created will only have the RNN layers corresponding to the chosen inputs so keep in mind when using this option only for predictions.

`--preprocess` option can be used to generate GP interpolated light curves from raw light curves. The output is saved in ML_spectra/processed_lightcurves folder and already contains all LCs for the data in ML_spectra. This option is in case new data is added.


```
usage: tune_train_test.py [-h] [--traintablepath TRAINTABLEPATH]
                          [--testtablepath TESTTABLEPATH] [--imgpath IMGPATH]
                          [--no-deredshifting] [--resolution RESOLUTION]
                          [--lcphase LCPHASE] [--auglimit AUGLIMIT]
                          [--layers {1,2,3a,3b} [{1,2,3a,3b} ...]]
                          [--exclude-augmented]
                          [--channels {spec,specdmdt,lcr,lcg,lcrdmdt,lcgdmdt}]
                          [--usetunedmodel] [--usetrainedmodel]
                          [--epochs EPOCHS] [--batch BATCH] [--opt {adam,sgd}]
                          [--preprocess] [--tune] [-ow] [--noshowplot]
                          [--onlytest]

optional arguments:
  -h, --help            show this help message and exit
  --traintablepath TRAINTABLEPATH
                        Path to table containing list of training spectra and
                        associated information
  --testtablepath TESTTABLEPATH
                        Path to table containing list of testing spectra and
                        associated information
  --imgpath IMGPATH     Path to save the training instance, include full path,
                        give a folder name that identifies training specifics
                        (like resolution)
  --no-deredshifting    Select to not de-redshift the spectra for training
  --resolution RESOLUTION
                        Number of spectral data points per spectrum
  --lcphase LCPHASE     Phase of light curves from discovery to include
  --auglimit AUGLIMIT   If including augmented data, how many samples of each
                        type to generate
  --layers {1,2,3a,3b} [{1,2,3a,3b} ...]
  --exclude-augmented   Select to not train with augmented data
  --channels {spec,specdmdt,lcr,lcg,lcrdmdt,lcgdmdt}
                        Select which channels to include in training
  --usetunedmodel       Select to use tuned model for training new data
  --usetrainedmodel     Select to use trained model for training new data
  --epochs EPOCHS       Number of epochs to train
  --batch BATCH         Batch size
  --opt {adam,sgd}      Optimizer, adam or sgd
  --preprocess          Select if train/test lightcurves need to be processed
                        to generate GP interpolated lightcurves, raw lc
                        path=../ML_spectra/lightcurves, processed lc
                        path=../ML_spectra/processed_lightcurves
  --tune                Select to tune model hyperparameters
  -ow, --overwrite      Select if want to overwrite training and testing json
                        files in imgpath
  --noshowplot          To disable plotting
  --onlytest            Only run the testing set
```

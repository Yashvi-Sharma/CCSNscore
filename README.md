# CCSNscore

A deep learning based program with parallel binary classifiers for classification of core collapse supernovae (SN) from ultra low resolution spectra (SEDM) and lightcurves. The trained models are used hierarchically (first set of models to classify SN as hydrogen rich or poor (layer 2), then one set of models to classify hydrogen-rich subtypes and other for hydrogen-poor subtypes (layers 3a and 3b)).

NOTE: The models provided in trained_models/ have been trained on a mixture of public and private dataset (described in this paper) hence all the spectra and lightcurves are not made available here. But the csv files in data/ contain the metadata information for training and testing sets used for the trained_models/. The non-SEDM spectra can be queried from the Open SN Catalog. The public SEDM spectra can be obtained from the Transient Name Server. 


## Usage

The code allows user to set up any SN classification task and train a set of parallel binary classifiers for their desired categories (see example_config). The following instructions should be followed to prepare the dataset in correct format for CCSNscore.

1. Metadata CSV: Create separate csv files for training and testing dataset in the format of sampletable.csv. The required columns that should be present in the csvs are - ['name', 'type' ,'z', 'specjd', 'maxjd', 'instrument', 'specfilename', 'lcfilename']. 'z' is for redshift, 'type' is for the transient classification type, 'specjd' is for JD date of spectrum observation, 'maxjd' is for JD date of lightcurve maximum, 'specfilename' is for full path of the spectral file ('data/spectra/filename.ascii'), 'lcfilename' is for full path of the raw lightcurve file ('data/lightcurves/filename.csv')

2. Spectra: The raw spectra files (from 'specfilename' column in metadata csv) should be in ascii format. Comments denoted using '#', the first column for wavelength and the second for flux, columns separated by space. If using trained_models/ wavelengths should be in Angstrom and in the range 3000-10000 \AA. The default directory to keep raw 1D spectra ascii files is data/spectra/ (change this path in config if it is different). 

3. Lightcurves: Is using LC channels, the raw lightcurve files (from 'lcfilename' column in metadata csv) should be supplied in csv format. The following columns are necessary - 'MJD', 'mag', 'emag' (magnitude error), 'filter' (r or g). The raw LC should contain 3 sigma detections (upper limits are thrown away). The default directory to keep raw LC files is data/lightcurves/ (change this path in config if it is different). The code processes these files and saves the interpolated LCs in data/gp_lightcurves/. To separately run gp LC processing for all lightcurves at once, check out `python data_preprocessing.py --help`.


### Config file

An example config is provided (example_config) to set up a 'Ia' vs. 'notIa' classification task. One binary classifier is trained per entry in the `MODEL` key. `LABELS` key is used to indicate which 'type (from metadata 'type' column) belongs to which output category. 

- `DEREDSHIFT` specifies whether to change spectral range from observed to rest wavelengths. 
- `RESOLUTION` specifies the number of data points per spectrum sampling), default is 256, increasing it only makes sense for openSN spectra. 
- `LCPHASE` specifies the length of light curve in days to include. 
- `TYPES_TO_AUGMENT` specifies the transient types for which fake samples should be generated
- `AUGLIMIT` specifies the number of fake samples to be generated per specified type
- `AUGMENT` If true the fake samples are used in training
- `AUGMENT_CONSTANT` If `AUGMENT` is true, the number of samples per specified type are equalized to this number by augmenting with fake samples. 


For each label there is one binary classifier model. The final predictions are decided based on the predictions of all the parallel binary classifiers. if any two classifier probabilities are too close to each other, the sample is labeled ambiguous. Each binary classifier model has channels for multiple inputs, `CHANNELS` key allows you to choose which inputs to use per classifier. 


```

```

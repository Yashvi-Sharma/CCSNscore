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
# Example config file (remove the comments before using)
{
  "SPECTRA_DIR": "data/spectra/",       # Path to the directory containing the raw spectra
  "LC_DIR": "data/lightcurves/",        # Path to the directory containing the raw lightcurves
  "GPLC_DIR": "data/gp_lightcurves/",   # Path to the directory containing the Gaussian Process interpolated LCs

  "TRAINTABLEPATH": "data/train.csv",   # Path to table from which training and validation sets will be created
  "TESTTABLEPATH": "data/test.csv",     # Path to table from which test set will be created
  "OUTPUT_DIR": "workdirs/Binary_Ia/",  # Working directory where model, logs, and results will be saved

  "TASKNAME": "binIa",                       # Prefix for output figures
  "MODE": "train",                         # Mode to run the code (tune, train, test), train will also test the model

  "MODELS": ["Ia", "notIa"],             # Parallel binary classifiers to train
  "LOAD_TUNED_MODELS": false,            # Whether to use the tuned models or not (saved with BC_ prefix)
  "LOAD_WEIGHTS": false,                 # False to just use the tuned architecture, True to load the weights

  "LABELS": {
    "Ia": ["Ia", "Ia-91bg", "Ia-91T", "Ia-pec", "Ia-02cx", "Iax", "Ia-norm"],
             "notIa": ["IIP", "IIL", "IIn", "II", "IIb", "Ib", "Ic", "Ic-BL", "SLSN-I", "SLSN-II", "AGN", "CV"]
            },                           # Transient classes that belong to the positive class of each binary classifier
  "CHANNELS": {"Ia": ["spec", "lcr", "lcg", "lcrdmdt", "lcgdmdt"],
               "notIa": ["spec", "lcr", "lcg", "lcrdmdt", "lcgdmdt"]
              },                         # Channels to use for each binary classifier

  "DEREDSHIFT": true,                    # Whether to apply redshift correction to the spectra
  "RESOLUTION": 256,                     # Resolution of the spectra
  "LCPHASE": 200,                        # Number of days for the lightcurve length
  "TYPES_TO_AUGMENT": ["Ib", "Ic"],      # Transient classes to generate augmented samples for
  "AUGLIMIT": 2000,                      # Maximum number of augmented samples per class to generate
  "AUGMENT": true,                        # Boolean to augment the training set or not
  "AUGMENT_CONSTANT": 2000,              # Total number of samples per TYPES_TO_AUGMENT
  "OVERWRITE": false,                    # Whether to overwrite traindata.json and testdata.json or not

  "EPOCHS": 100,                         # Number of epochs to train the model
  "BATCHSIZE": 32,                       # Batch size for training
  "LEARNING_RATE": 0.001,                # Learning rate for the optimizer
  "PATIENCE": 7,                         # Number of epochs to wait before early stopping
  "NOSHOWPLOTS": false,                  # Whether to show plots or not
}




```

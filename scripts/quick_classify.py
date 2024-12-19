'''
Script that takes a list of spectral files, redshifts and optionally light curve files,
makes a config file required to run test mode with tune_train_test.py, and runs the test mode.
'''

import os, subprocess, sys, json
import numpy as np
import pandas as pd
import argparse
from astropy.time import Time
from plotter import plot_reports

def create_config(args, basepath, outpath):
    # Read in the example config json
    config = json.loads(open(f"{basepath}example_config").read())
    # Modify the config
    config['TESTTABLEPATH'] = f'{outpath}test.csv'
    config['OUTPUT_DIR'] = outpath
    config['MODE'] = 'test'
    _ = config.pop('TYPE_COLUMN')
    _ = config.pop('LABELS')
    config['DEREDSHIFT'] = True
    config['TYPES_TO_AUGMENT'] = []
    config['AUGMENT'] = False

    ## Layer 1 config
    if 'layer1' in args.layers:
        config['OVERWRITE'] = True
        config['MODELS'] = ['Hrich', 'Hpoor']
        if 'onlyspec' in args.channels:
            config['TASKNAME'] = args.taskname + '_layer1_onlyspec'
            config['TRAINED_MODELPATH'] = {'Hrich': f"{basepath}trained_models/onlyspec/RT_Hrich",
                                           'Hpoor': f"{basepath}trained_models/onlyspec/RT_Hpoor"}
            config['CHANNELS'] = {'Hrich': ['spec'], 'Hpoor': ['spec']}
            json.dump(config, open(f"{outpath}config_{config['TASKNAME']}", 'w'), indent=4)
        elif 'onlyspeclcs' in args.channels:
            config['TASKNAME'] = args.taskname + '_layer1_onlyspeclcs'
            config['TRAINED_MODELPATH'] = {'Hrich': f"{basepath}trained_models/onlyspeclcs/RT_Hrich",
                                           'Hpoor': f"{basepath}trained_models/onlyspeclcs/RT_Hpoor"}
            config['CHANNELS'] = {'Hrich': ['spec', 'lcr', 'lcg'], 'Hpoor': ['spec', 'lcr', 'lcg']}
            json.dump(config, open(f"{outpath}config_{config['TASKNAME']}", 'w'), indent=4)

    ## Layer 2a config
    if 'layer2a' in args.layers:
        config['OVERWRITE'] = False
        config['MODELS'] = ["II", "IIb-H", "IIn"]
        if 'onlyspec' in args.channels:
            config['TASKNAME'] = args.taskname + '_layer2a_onlyspec'
            config['TRAINED_MODELPATH'] = {"II": f"{basepath}trained_models/onlyspec/RT_II",
                                           "IIb-H": f"{basepath}trained_models/onlyspec/RT_IIb-H",
                                           "IIn": f"{basepath}trained_models/onlyspec/RT_IIn"}
            config['CHANNELS'] = {"II": ['spec'], "IIb-H": ['spec'], "IIn": ['spec']}
            json.dump(config, open(f"{outpath}config_{config['TASKNAME']}", 'w'), indent=4)
        elif 'onlyspeclcs' in args.channels:
            config['TASKNAME'] = args.taskname + '_layer2a_onlyspeclcs'
            config['TRAINED_MODELPATH'] = {"II": f"{basepath}trained_models/onlyspeclcs/RT_II",
                                           "IIb-H": f"{basepath}trained_models/onlyspeclcs/RT_IIb-H",
                                           "IIn": f"{basepath}trained_models/onlyspeclcs/RT_IIn"}
            config['CHANNELS'] = {"II": ['spec', 'lcr', 'lcg'], "IIb-H": ['spec', 'lcr', 'lcg'],
                                  "IIn": ['spec', 'lcr', 'lcg']}
            json.dump(config, open(f"{outpath}config_{config['TASKNAME']}", 'w'), indent=4)


    ## Layer 2b config
    if 'layer2b' in args.layers:
        config['OVERWRITE'] = False
        config['MODELS'] = ["Ib", "Ic", "Ic-BL"]
        if 'onlyspec' in args.channels:
            config['TASKNAME'] = args.taskname + '_layer2b_onlyspec'
            config['TRAINED_MODELPATH'] = {"Ib": f"{basepath}trained_models/onlyspec/RT_Ib",
                                           "Ic": f"{basepath}trained_models/onlyspec/RT_Ic",
                                           "Ic-BL": f"{basepath}trained_models/onlyspec/RT_Ic-BL"}
            config['CHANNELS'] = {"Ib": ['spec'], "Ic": ['spec'], "Ic-BL": ['spec']}
            json.dump(config, open(f"{outpath}config_{config['TASKNAME']}", 'w'), indent=4)
        elif 'onlyspeclcs' in args.channels:
            config['TASKNAME'] = args.taskname + '_layer2b_onlyspeclcs'
            config['TRAINED_MODELPATH'] = {"Ib": f"{basepath}trained_models/onlyspeclcs/RT_Ib",
                                           "Ic": f"{basepath}trained_models/onlyspeclcs/RT_Ic",
                                           "Ic-BL": f"{basepath}trained_models/onlyspeclcs/RT_Ic-BL"}
            config['CHANNELS'] = {"Ib": ['spec', 'lcr', 'lcg'], "Ic": ['spec', 'lcr', 'lcg'],
                                  "Ic-BL": ['spec', 'lcr', 'lcg']}
            json.dump(config, open(f"{outpath}config_{config['TASKNAME']}", 'w'), indent=4)


def main():
    parser = argparse.ArgumentParser('Running predictions with trained CCSNscore models on a list of spectra')
    parser.add_argument("spectra", help="List of spectra files (full paths)", nargs='+')
    parser.add_argument("--redshifts", help="List of redshifts", nargs='+', default=None)
    parser.add_argument("--lightcurves", help="List of light curve files (full paths)", nargs='+',
                        default=None)
    parser.add_argument("--names", help="List of names for the spectra", nargs='+', default=None)
    parser.add_argument("--taskname", help="To be used for output filename prefixes",
                        default=Time.now().strftime('%Y%m%d_%H%M%S')+'_ccsnscore')
    parser.add_argument("--layers", help="Which hierarchical tasks to predict, layer1 - (Hrich vs Hpoor),"
                                         " layer2a - (II vs IIb-H vs IIn), layer2b - (Ib vs Ic vs Ic-BL), "
                                         "can choose multiple layers",
                        choices=['layer1', 'layer2a', 'layer2b'], default='layer1',nargs='+')
    parser.add_argument("--channels", help="Which input channel configuration to use for prediction, "
                                           "onlyspec - (spectra only), onlyspeclcs - (spectra and light curves),"
                                           "can choose both", choices=['onlyspec', 'onlyspeclcs'],
                        default='onlyspec',nargs='+')
    parser.add_argument("--noplot", help="Don't plot the results", action='store_true')

    args = parser.parse_args()

    # Check if the number of spectra and redshifts match
    if args.redshifts is not None and len(args.spectra) != len(args.redshifts):
        print("Number of spectra and redshifts do not match")
        sys.exit(1)
    if args.lightcurves is not None and len(args.spectra) != len(args.lightcurves):
        print("Number of spectra and light curves do not match")
        sys.exit(1)

    # Create a directory to store the output files
    ## get the current working directory
    cwd = os.getcwd()
    ## get path from cwd until CCSNscore
    basepath = cwd.split('CCSNscore')[0]+'CCSNscore/'
    outpath = basepath+'workdirs/'+args.taskname+'/'
    if not os.path.exists(outpath):
        os.mkdir(outpath)

    # Input df
    if args.names is not None:
        names = args.names
    else:
        names = [os.path.basename(spec).split('.')[0] for spec in args.spectra]
    testtable = pd.DataFrame({'name':names,
                              'specfilename': args.spectra, 'z': np.array(args.redshifts).astype(float),
                              'lcfilename': args.lightcurves})
    testtable.to_csv(f'{outpath}test.csv', index=False)

    # Create the config files
    ## consistency check
    if args.lightcurves is None and 'onlyspeclcs' in args.channels:
        print("No light curves provided, cannot use onlyspeclcs channel configuration")
        sys.exit(1)
    create_config(args, basepath, outpath)

    # Run the test mode
    results = []
    for layer in args.layers:
        for suff in args.channels:
            subprocess.run( f' python {basepath}scripts/tune_train_test.py '
                            f'{outpath}config_{args.taskname}_{layer}_{suff}',shell=True)
            results.append(f'{outpath}{args.taskname}_{layer}_{suff}_results.csv')

    # Plot the report
    if not args.noplot:
        plot_reports(testtable, results, outpath)

if __name__ == '__main__':
    main()
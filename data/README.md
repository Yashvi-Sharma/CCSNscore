The `train_metadata.csv` and `test_metadata.csv` files list all the samples used for training and evaluating models saved in trained_models/. 
They contain the following columns:
- `name`: Name of the supernova
- `ntype`: Type of the supernova
- `z`: Redshift of the supernova
- `specjd`: Julian date of the spectrum observation
- `maxjd`: Julian date of the maximum light of the supernova
- `peakmag`: Peak magnitude of the supernova
- `peakfilt`: Filter of the peak magnitude
- `instrument`: Instrument used to observe the spectrum
Additionally `test_metadata.csv` contains the following columns:
- `host_contaminated`: Whether the spectrum is host galaxy contaminated (if Type~Ibc)
- `num_snid`: Number of SNID matches with rlap>4 and 'good' quality
- `tag`: Gold or Bronze tag for the test set sample

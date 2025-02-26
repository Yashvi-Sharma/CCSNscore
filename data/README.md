The `train_metadata.csv` and `test_metadata.csv` files list all the samples used for training and evaluating models saved in trained_models/. The performance of these models is described in the [CCSNscore](https://arxiv.org/abs/2412.08601) paper. 
The tables contain the following columns:
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

The CCSNscore models in trained_models/onlyspec (models trained only using the spectral data) are currently used for real-time reporting of supernova classifications to the Transient Name Server ([TNS](https://www.wis-tns.org)). The following threshold cuts are currently applied to the scores and `numSNID` (see CCSNscore paper) to determine the eligibility for TNS reporting: 
- For `numSNID` > 30: `CCSNscore` > 0.8 and `CCSNscore_err` < 0.05 
- For `numSNID` > 20 and `numSNID` < 30: `CCSNscore` > 0.9 and `CCSNscore_err` < 0.05
- For `numSNID` < 20: `CCSNscore` > 0.95 and `CCSNscore_err` < 0.05
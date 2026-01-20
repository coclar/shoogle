# shoogle

Shoogle is a python package for timing gamma-ray pulsars using Fermi-LAT data. It uses introduces sets of latent variables that transform the gamma-ray timing likelihood into more a tractable Gaussian distribution that can be fit analytically using standard pulsar-timing Gaussian process approximations, and uses the technique of Gibbs Sampling to marginalise over these latent variables.

The full method is described in the shoogle paper:

_Timing Gamma-ray Pulsars using Gibbs Sampling_  
Clark, C. J., Valtolina, S., Nieder, N. and van Haasteren, R.  
(2026), A&A, submitted, [arXiv:2601.07592](https://arxiv.org/abs/2601.07592)

Please cite our paper if you use shoogle in your work.

## Installation

```
pip install shoogle-pulsar
```

(ideally after creating and/or activating a dedicated virtual environment)

## Usage

After installation, the scripts `run_shoogle` and `plot_shoogle` will be added to your PATH, and should be available whenever your python environment is activated.

### Running the sampler

To run the sampler, you will need:
* An "FT1" file, which is a .fits file containing the Fermi photon data. These files can be obtained from the [FSSC](https://fermi.gsfc.nasa.gov/ssc/data/). You will need to run `gtsrcprob` from the [Fermitools](https://fermi.gsfc.nasa.gov/ssc/data/analysis/) to generate a column in the FT1 file containing photon probability weights. Pre-made FT1 files containing these weights for pulsars included in the Third Pulsar Catalog are also available from the [FSSC 3PC page](https://fermi.gsfc.nasa.gov/ssc/data/access/lat/3rd_PSR_catalog/).

* An "FT2" file: the "spacecraft" fits file containing telemetry. This can also be downloaded from the [FSSC](https://heasarc.gsfc.nasa.gov/FTP/fermi/data/lat/mission/spacecraft/)

* Optionally, a PINT-compatible .tim file containing radio ToAs for the same pulsar, if you want to try the experimental joint radio/gamma-ray fitting capabilities.

* A PINT-compatible .par file containing the pre-fit pulsar timing model. See below for shoogle-specific syntax.

* A file describing the pre-fit pulse profile template. This can be obtained from the FT1 file (assuming the PULSE_PHASE column has been populated using the initial timing model by PINT or tempo2) using the `itemplate.py` script from [GeoTOA-2.0](https://fermi.gsfc.nasa.gov/ssc/data/analysis/user/).

The sampler is run via the `run_shoogle` command, with the above inputs. It will start by tuning MCMC samplers specific to the pulse profile template (and optional noise hyperparameters), before beginning the actual Gibbs sampling. It will monitor the autocorrelation time of the resulting sample chains, running until `--ntau` autocorrelation times (approximately the number of statistically independent samples) have been accumulated.

`run_shoogle` will periodically output an output file containing these sampling chains. This can be analysed with the `plot_shoogle` command, which will output the best-fitting timing model and uncertainties, and some plots showing the results of the sampling process. Please see the help message for these scripts (`run_shoogle -h`) for more details. 

The [examples](examples) directory contains ready-made inputs, scripts and a jupyter notebook that can be used as a guideline.

Notes:
* The timing model should be in a PINT-compatible .par file, with lines following:
  ```
  [PARAMETER] [Initial value] [free] [prior uncertainty]
  ```

* ### IMPORTANT:
  Uncertainties provided in the .par file are interpreted
  as the widths of Gaussian priors,
  so BE VERY CAREFUL about including these!

  For example:
  ```
  PMRA     0    1   10
  ```
  Will place a zero-mean Gaussian prior on PMRA, with width 10 mas/yr,

  while
  ```
  PMRA     0    1
  ```
  will assume an unbounded uniform prior on PMRA.

  After sampling, shoogle outputs .par files with the posterior
  uncertainty in this fourth column.
  This means that you *cannot immediately use the output from a shoogle
  run to start a new one*, as the posterior uncertainty will be treated
  as a prior, and so the likelihood will be double-counted!
  (we plan to fix this unhelpful behaviour soon...)

* The following extra parameters (not known to PINT) can be added to the .par file to define a noise model:

  `TN_REDAMP`: log10(amplitude) of timing noise.
             Amplitude is in units of yr^{3/2},
             at a reference frequency of 1 yr^-1

  `TN_REDFC`:  log10(corner frequency / 1 yr^-1)

  `TN_REDGAM`: high-frequency spectral index,
             PSD(f) ~ f^{-gamma} at high frequencies,
             so TN_REDGAM should be positive

  `TN_REDKAPPA`: (optional) flat tail level, in units of yr^{3/2}.

  For orbital period variations, replace `TN_` with `OPV_`

  Additional components can be added to the noise model by adding,
  `TN2_REDAMP`, `TN2_REDFC`, `TN2_REDGAM`, `TN3_REDAMP` ... etc.

  .par file syntax for these hyperparameters is:

  ```
  [HYPERPARAMETER] [Initial value] [free]
  ```
  e.g.:
  ```
  TN_REDAMP        -14              1
  TN_REDFC         -2               0
  TN_REDGAM         4.333           0
  ```
  will fit for a GWB-like power-law-like spectrum (with a corner frequency of 1/100 yr, well below the sensitive band), where only the amplitude is free to vary.

* To make use of these hyperparameters, the parameters for the relevant basis functions need to be included in the .par file.

  For timing noise, these are PINT's WAVEX parameters.
  For orbital period variations, these are PINT'S ORBWAVE parameters.

  For performance reasons, if you include the ORBWAVES model,
  then do not mark these as free variables in the .par file.
  PINT's design matrix evaluation is very slow for these parameters,
  so this is handled automatically by shoogle.

## License

shoogle is available under the BSD-3-Clause Licence. By downloading, cloning
or forking this repository or its artifacts you, as the licensee, agree to the
additional license terms as defined in the "Agreement on the provision of
free-of-charge open-source software via the Internet" ([DE](OSSVereinbarung.pdf)/[EN](OSSAgreement.pdf))

## Contributing

Please see [CONTRIBUTING.md](CONTRIBUTING.md)
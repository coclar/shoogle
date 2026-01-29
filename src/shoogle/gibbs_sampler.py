import sys
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import corner
import emcee
from tqdm.auto import tqdm
import time

from astropy import units as u

from scipy.stats import norm, multivariate_normal as mvn, Covariance, dirichlet
from scipy.linalg import cholesky, cho_solve, solve

from pint.templates.lctemplate import LCTemplate, prim_io
from pint.templates.lceprimitives import LCGaussian
from pint import toa
from pint.eventstats import hmw
from pint.models import get_model
from pint.fermi_toas import get_Fermi_TOAs
from pint.observatory.satellite_obs import SatelliteObs
import pint.logging

pint.logging.setup(level="WARNING")

from shoogle.plot_gibbs_results import GibbsResults, fermi_lc
from shoogle.utils import *
from shoogle.noise_models import BrokenPowerLaw, FlatTailBrokenPowerLaw
from shoogle.conditional_samplers import *

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp


def read_input_ft1_file(infile, FT2, weightfield, wmin, ephem_str):

    try:
        SatelliteObs(name="Fermi", ft2name=FT2)
    except ValueError:
        pass

    toas = get_Fermi_TOAs(
        infile,
        weightcolumn=weightfield,
        minweight=wmin,
        include_bipm=False,
        planets=True,
        ephem=ephem_str,
    )

    weights = np.array([float(t["weight"]) for t in toas.table["flags"]])
    energies = np.array([float(t["energy"]) for t in toas.table["flags"]])

    return toas, weights, energies


class Gibbs(object):
    """
    This is shoogle's main class, which runs the Gibbs sampling for a pulsar.
    """

    def __init__(
        self,
        parfile=None,
        ft1file=None,
        ft2file=None,
        weightfield="MODEL_WEIGHT",
        templatefile=None,
        wmin=0.05,
        timfile=None,
        Edep=False,
    ):
        """

        Initialises the Gibbs sampler by:
        * Uses PINT to load in the Fermi FT1 and FT2 files,
          and (optionally) a .tim file containing ToAs.
        * Uses PINT to load the timing model from a .par file,
          folds the data to obtain an initial set of (pre-fit) phases,
          and evaluates the design matrix.
        * Reads the .par file to identify noise-model hyperparamters to fit for.

        Parameters
        ----------
        parfile      : str
                       .par file containing the timing model
        ft1file      : str
                       .fits file containing photon data
        ft2file      : str
                       .fits file containing spacecraft telemetry.
                       Obtainable from the FSSC.
        weightfield  : str
                       Name of column in FT1 file containing photon weights
        templatefile : str
                       Name of file containing template pulse profile parameters
                       Can be created using "itemplate.py" from GeoTOA-2.0
        wmin         : float, default = 0.05
                       Minimum photon weight to include.
                       Typically this should be around 0.01--0.05.
                       Higher values speed up processing but decrease sensivity.
        timfile      : str, optional
                       .tim file containing radio ToAs
        Edep         : bool, default=False
                       Fit for energy-dependence in the gamma-ray pulse profile
                       Template peak parameters (amp, position, width) will be
                       linearly interpolated as a function of log10(E)

        """

        self.parfile = parfile
        self.timing_model = get_model(self.parfile)

        print("Loading FT1 file")

        self.photon_toas, self.w, energies = read_input_ft1_file(
            ft1file, ft2file, weightfield, wmin, self.timing_model.EPHEM.value
        )
        self.t = self.photon_toas.get_mjds().value

        self.log10E = np.log10(energies)

        self.nphot = len(self.t)

        self.has_TN = "WXSIN_0001" in self.timing_model.params
        self.fit_TN = False

        if (
            "ORBWAVEC0" in self.timing_model.params
            and self.timing_model.ORBWAVEC0.value is not None
        ):
            self.has_OPV = True
        else:
            self.has_OPV = False
        self.fit_OPV = False

        self.phi = np.asarray(
            self.timing_model.phase(self.photon_toas).frac, dtype=float
        )
        if timfile is not None:
            self.radio_toas = toa.get_TOAs(timfile, planets=True)
            self.radio_toa_uncerts = np.asarray(
                (self.radio_toas.get_errors() * self.timing_model.F0.quantity).to(
                    u.dimensionless_unscaled
                ),
                dtype=float,
            )
            self.radio_t = self.radio_toas.get_mjds().value
            self.nradio = len(self.radio_t)

        # check for additional phase shifts
        extra_phase = None
        with open(self.parfile, "r") as par:
            lines = par.readlines()
            for l in lines:
                if l.split()[0] == "LATPHASE":
                    extra_phase = float(l.split()[1])

        self.Edep = Edep
        if self.Edep:
            self.tau_sampler = EdepTemplateSampler(templatefile, self.w, self.log10E)
        else:
            self.tau_sampler = TemplateSampler(templatefile, self.w)

        self.npeaks = self.tau_sampler.npeaks

        if self.Edep:
            self.zm_sampler = LatentVariableSampler(
                self.w, self.npeaks, log10E=self.log10E
            )
        else:
            self.zm_sampler = LatentVariableSampler(self.w, self.npeaks)

        print("Constructing design matrix")
        self._make_design_matrix()

        self.noise_models = []

        self.Tobs = self.t.max() - self.t.min()
        if not hasattr(self, "WXfreqs"):
            self._make_wavex_frequencies()

        if self.has_TN:

            nc = 1

            # Add components to the noise model, until there are none left in the .par file
            while True:

                if nc == 1:
                    prefix = "TN"
                else:
                    prefix = f"TN{nc}"

                logA0 = None
                A_free = False
                logFC0 = None
                FC_free = False
                GAM0 = None
                GAM_free = False
                KAPPA0 = None
                KAPPA_free = False

                for line in lines:
                    if len(line.split()) < 2:
                        continue
                    if line.split()[0] == f"{prefix}_REDAMP":
                        logA0 = float(line.split()[1])
                        A_free = int(line.split()[2])
                    if line.split()[0] == f"{prefix}_REDFC":
                        logFC0 = float(line.split()[1])
                        FC_free = int(line.split()[2])
                    if line.split()[0] == f"{prefix}_REDGAM":
                        GAM0 = float(line.split()[1])
                        GAM_free = int(line.split()[2])
                    if line.split()[0] == f"{prefix}_REDKAPPA":
                        KAPPA0 = float(line.split()[1])
                        KAPPA_free = int(line.split()[2])

                if logA0 is not None:
                    self.Tobs = (self.WXfreqs[0] * u.d) ** (-1)

                    if A_free or FC_free or GAM_free or KAPPA_free:
                        self.fit_TN = True

                    if KAPPA0 is not None:
                        self.noise_models.append(
                            FlatTailBrokenPowerLaw(
                                self.WXfreqs,
                                self.WXSinds,
                                self.WXCinds,
                                np.array(
                                    [A_free, FC_free, GAM_free, KAPPA_free], dtype=bool
                                ),
                                self.Tobs,
                                np.array([logA0, logFC0, GAM0, KAPPA0]),
                                prefix,
                            )
                        )

                    else:
                        self.noise_models.append(
                            BrokenPowerLaw(
                                self.WXfreqs,
                                self.WXSinds,
                                self.WXCinds,
                                np.array([A_free, FC_free, GAM_free], dtype=bool),
                                self.Tobs,
                                np.array([logA0, logFC0, GAM0]),
                                prefix,
                            )
                        )
                    if A_free == 2:
                        self.noise_models[-1].linear_amp_prior = True
                    nc += 1
                else:
                    break

        if self.has_OPV:
            nc = 1

            while True:

                if nc == 1:
                    prefix = "OPV"
                else:
                    prefix = f"OPV{nc}"

                logA0 = None
                logFC0 = None
                GAM0 = None
                A_free = False
                FC_free = False
                GAM_free = False

                for line in lines:
                    if len(line.split()) < 2:
                        continue
                    if line.split()[0] == f"{prefix}_REDAMP":
                        logA0 = float(line.split()[1])
                        A_free = int(line.split()[2])
                    if line.split()[0] == f"{prefix}_REDFC":
                        logFC0 = float(line.split()[1])
                        FC_free = int(line.split()[2])
                    if line.split()[0] == f"{prefix}_REDGAM":
                        GAM0 = float(line.split()[1])
                        GAM_free = int(line.split()[2])

                if logA0 is not None:
                    if not hasattr(self, "OPVfreqs"):
                        self._make_opv_frequencies()
                    if A_free or FC_free or GAM_free:
                        self.fit_OPV = True

                    self.noise_models.append(
                        BrokenPowerLaw(
                            self.OPVfreqs,
                            self.OPVSinds,
                            self.OPVCinds,
                            [A_free, FC_free, GAM_free],
                            self.Tobs,
                            np.array([logA0, logFC0, GAM0]),
                            prefix,
                        )
                    )
                    nc += 1

                else:
                    break

        self.nhyp = 0
        for component in self.noise_models:
            self.nhyp += component.nfree

        if self.has_OPV:
            self._orbital_design_matrix()

        if hasattr(self, "radio_toas"):
            self.radio_resids = np.asarray(
                self.timing_model.phase(self.radio_toas).frac, dtype=float
            )
            self.radio_resids -= np.mean(self.radio_resids)

        if self.nhyp > 0:
            if self.has_OPV:
                PB0 = self.PB0
            else:
                PB0 = None
            self.timing_sampler = NoiseAndTimingModelSampler(
                self.phi,
                self.M,
                self.theta_prior,
                self.timing_parameter_uncertainties,
                self.noise_models,
                self.parameter_scales,
                PB0,
            )
        else:
            self.timing_sampler = TimingModelSampler(self.phi, self.M, self.theta_prior)

        return

    def _lambda_emcee_logL_wrapper(self, x):

        hyp = np.array([])

        s = 0
        log_prior = 0.0
        for component in self.noise_models:
            n = component.nfree

            p = component.all_parameters(x[s : s + n])
            if p is None:
                return -np.inf

            s += n
            hyp = np.append(hyp, p)

            log_prior += component.log_prior(p)

        self._make_psd_cov(hyp)
        post_cov_inv = self.MT_Sigma_inv_M + self.inv_prior_cov

        try:
            post_cov_inv_U = cholesky(post_cov_inv, lower=False)
        except np.linalg.LinAlgError:
            return None

        self.post_cov_inv_U = post_cov_inv_U

        self.theta_opt = cho_solve(
            (post_cov_inv_U, False),
            self.MT_Sigma_inv_resids + self.inv_prior_cov @ self.theta_prior,
        )

        prior_chi2 = self.theta_prior @ (self.inv_prior_cov @ self.theta_prior)
        post_chi2 = self.theta_opt @ (post_cov_inv @ self.theta_opt)
        self.logdet_post = -2 * np.sum(np.log(np.diag(post_cov_inv_U)))
        logL = -0.5 * (self.logdet_prior + prior_chi2 - self.logdet_post - post_chi2)

        # plt.plot(self.WXfreqs, 1.0/np.diag(self.inv_prior_cov)[6::2])
        # plt.yscale('log')
        # plt.xscale('log')
        # plt.show()
        # raise ValueError

        # print(self.theta_prior)
        # print(np.diag(self.inv_prior_cov))
        # print(x,
        #       "prior_chi2:", prior_chi2,
        #       "logdet_prior:", self.logdet_prior,
        #       "post_chi2:", post_chi2,
        #       "logdet_post:", self.logdet_post)

        return logL + log_prior

    def _tune_lambda_sampler(self, hyp0, progress=True):

        x0 = np.array([])
        for component in self.noise_models:
            x0 = np.append(x0, component.x0[component.free])

        plt.ioff()
        nwalkers = 32

        ndim = len(x0)
        cov = np.eye(ndim) * 1e-2

        sampler = emcee.EnsembleSampler(nwalkers, ndim, self._lambda_emcee_logL_wrapper)
        start = mvn.rvs(size=nwalkers, mean=x0, cov=cov)
        if ndim == 1:
            start = start.reshape(len(start), 1)

        sampler.run_mcmc(
            start,
            2000,
            progress=progress,
            progress_kwargs={"desc": "Tuning hyperpar sampler, step 1 of 2"},
        )

        chain = sampler.get_chain(thin=25, discard=100, flat=True)

        cov = np.cov(chain, rowvar=False)

        sampler2 = emcee.EnsembleSampler(
            1,
            ndim,
            self._lambda_emcee_logL_wrapper,
            live_dangerously=True,
            moves=[emcee.moves.GaussianMove(cov)],
        )
        x0 = chain[-1, :]
        sampler2.run_mcmc(
            x0,
            20000,
            progress=progress,
            skip_initial_state_check=True,
            progress_kwargs={"desc": "Tuning hyperpar sampler, step 2 of 2"},
        )

        chain2 = sampler2.get_chain(discard=1000, flat=True)

        cov = np.cov(chain2, rowvar=False)

        self.lambda_autocorr_time = int(
            np.ceil(np.max(sampler2.get_autocorr_time(discard=1000)))
        )
        self.lambda_emcee_cov = cov

    def _sample_lambda_given_z_tau(self, x0):

        ndim = len(x0)

        sampler = emcee.EnsembleSampler(
            1,
            ndim,
            self._lambda_emcee_logL_wrapper,
            live_dangerously=True,
            moves=[emcee.moves.GaussianMove(self.lambda_emcee_cov)],
        )
        sampler.run_mcmc(
            x0,
            self.lambda_autocorr_time,
            progress=False,
            skip_initial_state_check=True,
        )

        lambda_new = sampler.get_chain(flat=True)[-1, :]

        # Re-compute Cholesky matrix with last accepted sample
        # (might not be the same as the last attempted sample, since that could have been rejected)
        self._lambda_emcee_logL_wrapper(lambda_new)

        c = 0
        new_hyp = np.array([])
        for component in self.noise_models:
            n = component.nfree
            new_hyp = np.append(new_hyp, lambda_new[c : c + n])
            c += n

        return new_hyp

    def _sample_theta_given_lambda_z_tau(self):

        p = mvn.rvs(mean=np.zeros_like(self.theta_opt))
        theta = self.theta_opt + solve(self.post_cov_inv_U, p)
        phase_shifts = self.M @ theta
        mean_phase_shift = np.mean(phase_shifts)
        phase_shifts -= mean_phase_shift

        if hasattr(self, "radio_toas"):
            radio_phase_shifts = self.Mradio @ (theta - self.theta_prior)
            resids = np.append(resids, self.radio_resids - radio_phase_shifts)
            radio_phase_shifts -= mean_phase_shift

        return theta, phase_shifts

    def _setup_leastsq_given_z_tau(self, mu_z, sigma_z):

        npar = self.n_timing_pars

        if self.has_OPV:
            npar += self.nOPVfreqs * 2

        # Setting up WLS with photons assigned to peaks
        pulsed_photons = np.where(sigma_z > 0)[0]
        M = np.copy(self.M[pulsed_photons])
        resids = np.asarray(
            self.phi[pulsed_photons] - mu_z[pulsed_photons], dtype=float
        )
        precisions = 1.0 / sigma_z[pulsed_photons] ** 2

        if hasattr(self, "radio_toas"):
            resids = np.append(resids, self.radio_resids)
            precisions = np.append(precisions, self.radio_toa_uncerts**-2)
            M = np.append(M, self.Mradio, axis=0)

        MT_Sigma_inv = np.einsum("ij,j->ij", M.T, precisions, optimize="greedy")
        self.resids = resids
        self.Sigma_inv = precisions
        self.logdet_Sigma = -np.sum(np.log(self.Sigma_inv))
        self.MT_Sigma_inv_M = MT_Sigma_inv @ M
        self.MT_Sigma_inv_resids = MT_Sigma_inv @ resids
        self.sum_precisions = np.sum(precisions)

    def _Htest_logL(self, tau, phase_shifts):

        phi = np.mod(self.phi - phase_shifts, 1.0)

        if self.Edep:
            profile = self.tau_sampler.template(tau, phi, self.log10E)
        else:
            profile = self.tau_sampler.template(tau, phi)

        logL = np.sum(np.log((1 - self.w) + self.w * profile))

        return hmw(phi, self.w), logL

    def _make_wavex_frequencies(self):

        nWXfreqs = 0
        for par in self.timing_model.params:
            if "WXCOS" in par or "WXSIN" in par:
                freq = int(par.split("_")[-1])
                if freq > nWXfreqs:
                    nWXfreqs = freq

        self.WXfreqs = np.array(
            [self.timing_model[f"WXFREQ_{f:04d}"].value for f in range(1, nWXfreqs + 1)]
        ) * (1 / u.d)

        self.nWXfreqs = nWXfreqs
        self.WXCinds = []
        self.WXSinds = []

        for f in range(nWXfreqs):
            self.WXCinds.append(
                np.where(self.timing_parameter_names == f"WXCOS_{f+1:04d}")[0][0]
            )
            self.WXSinds.append(
                np.where(self.timing_parameter_names == f"WXSIN_{f+1:04d}")[0][0]
            )

    def _make_opv_frequencies(self):

        self.OPVfmin = self.timing_model.ORBWAVE_OM.quantity.to_value("radian/d") / (
            2 * np.pi
        )

        nw = 0
        while hasattr(self.timing_model, f"ORBWAVEC{nw}"):
            nw += 1

        self.OPVfmax = self.OPVfmin * nw
        self.OPVfreqs = np.linspace(self.OPVfmin, self.OPVfmax, nw, dtype="float") * (
            1 / u.d
        )

        self.OPVCinds = []
        self.OPVSinds = []
        for f in range(nw):
            self.OPVCinds.append(self.n_timing_pars + 2 * f)
            self.OPVSinds.append(self.n_timing_pars + 2 * f + 1)

        self.nOPVfreqs = len(self.OPVfreqs)

    def _make_psd_cov(self, hyp):

        # Taking values/uncertainties in .par file as priors
        K = self.timing_parameter_uncertainties**2

        s = 0
        for component in self.noise_models:
            n = component.npar
            prior_variance = (
                component.powspec(hyp[s : s + n])
                * component.freqs.to_value("1/d").min()
            )

            if "OPV" in component.prefix:
                prior_variance /= self.PB0.to_value("s") ** 2

            K[component.Sinds] += prior_variance
            K[component.Cinds] += prior_variance
            s += n

        # Ignore divisions by zero, which are dealt with by np.where
        with np.errstate(divide="ignore"):
            # Invert, taking "0" uncertainty to mean no prior
            Kinv = np.where(K > 0.0, 1.0 / K, 0.0)

        Kinv /= self.parameter_scales**2

        self.inv_prior_cov = np.diag(Kinv)

        with np.errstate(divide="ignore"):
            self.logdet_prior = -np.sum(np.where(Kinv > 0, np.log(Kinv), 0.0))

    def _make_design_matrix(self):

        M, names, units = self.timing_model.designmatrix(
            self.photon_toas, incoffset=True
        )
        base_timing_params = []

        for n in names:
            base_timing_params.append(n)

        self.timing_parameter_names = np.array(base_timing_params)
        self.n_timing_pars = len(self.timing_parameter_names)

        if hasattr(self, "radio_toas"):
            Mradio_tmp, names_radio, units_radio = self.timing_model.designmatrix(
                self.radio_toas, incoffset=True
            )
            Mradio = np.zeros_like(Mradio_tmp)
            for jj, n in enumerate(names):
                ii = np.where(self.timing_parameter_names == n)[0][0]
                Mradio[:, ii] = Mradio_tmp[:, jj]

            M = np.append(M, np.zeros(self.nphot)[:, None], axis=1)
            Mradio = np.append(
                Mradio,
                np.ones(self.nradio)[:, None] / self.timing_model.F0.value,
                axis=1,
            )

            self.n_timing_pars += 1
            self.timing_parameter_names = np.append(
                self.timing_parameter_names, "RG_Offset"
            )

        self.timing_parameter_values = np.zeros(self.n_timing_pars)
        self.timing_parameter_uncertainties = np.zeros(self.n_timing_pars)

        pint_pars = self.timing_model.get_params_dict()
        for i, name in enumerate(self.timing_parameter_names):
            try:
                # This fails for "Offset" parameters
                par = pint_pars[name]
                self.timing_parameter_values[i] = par.value
                self.timing_parameter_uncertainties[i] = par.uncertainty.value
            except:
                self.timing_parameter_values[i] = 0.0
                self.timing_parameter_uncertainties[i] = 0.0

        if self.has_OPV:
            dd_dTASC = self.timing_model.get_deriv_funcs("DelayComponent")["TASC"][0]
            dtoa_dTasc = dd_dTASC(self.photon_toas, "TASC", None)
            if hasattr(self, "radio_toas"):
                radio_dtoa_dTasc = dd_dTASC(self.radio_toas, "TASC", None)

            PB = self.timing_model.PB.quantity
            if PB is None:
                PB = 1.0 / self.timing_model.FB0.quantity
            self.PB0 = PB

            ii = 0
            orbwaves = np.array([])
            WOM = self.timing_model.ORBWAVE_OM.quantity.to_value("radian/d")
            WEPOCH = self.timing_model.ORBWAVE_EPOCH.value
            while hasattr(self.timing_model, f"ORBWAVES{ii}"):
                orbwaves = np.append(
                    orbwaves, getattr(self.timing_model, f"ORBWAVES{ii}").value
                )
                orbwaves = np.append(
                    orbwaves, getattr(self.timing_model, f"ORBWAVEC{ii}").value
                )

                dtoa_dORBWAVES = -(
                    PB * dtoa_dTasc * np.sin(WOM * (ii + 1) * (self.t - WEPOCH))
                ).to_value("s")
                dtoa_dORBWAVEC = -(
                    PB * dtoa_dTasc * np.cos(WOM * (ii + 1) * (self.t - WEPOCH))
                ).to_value("s")
                M = np.append(M, dtoa_dORBWAVES[:, None], axis=1)
                M = np.append(M, dtoa_dORBWAVEC[:, None], axis=1)

                if hasattr(self, "radio_toas"):
                    dtoa_dORBWAVES = -(
                        PB
                        * radio_dtoa_dTasc
                        * np.sin(WOM * (ii + 1) * (self.radio_t - WEPOCH))
                    ).to_value("s")
                    dtoa_dORBWAVEC = -(
                        PB
                        * radio_dtoa_dTasc
                        * np.cos(WOM * (ii + 1) * (self.radio_t - WEPOCH))
                    ).to_value("s")
                    Mradio = np.append(Mradio, dtoa_dORBWAVES[:, None], axis=1)
                    Mradio = np.append(Mradio, dtoa_dORBWAVEC[:, None], axis=1)
                ii += 1

        # PINT returns dTOA/dParam, we want dPhase/dParam.
        M *= self.timing_model.F0.value

        if self.has_OPV:
            tasc_idx = np.where(self.timing_parameter_names == "TASC")[0][0]
            self.dphi_dtasc = M[:, tasc_idx] / 86400.0

        self.M = np.asarray(M, dtype=float)

        if hasattr(self, "radio_toas"):
            Mradio *= self.timing_model.F0.value
            self.radio_dphi_dtasc = Mradio[:, tasc_idx] / 86400.0
            self.Mradio = np.asarray(Mradio, dtype=float)
            self.parameter_scales = np.maximum(
                np.max(np.abs(self.M), axis=0), np.max(np.abs(self.Mradio), axis=0)
            )
        else:
            self.parameter_scales = np.max(np.abs(self.M), axis=0)

        if self.has_OPV:
            self.theta_prior = np.concatenate((np.zeros(self.n_timing_pars), -orbwaves))
            self.timing_parameter_uncertainties = np.concatenate(
                (self.timing_parameter_uncertainties, np.zeros(len(orbwaves)))
            )
        else:
            self.theta_prior = np.zeros(self.n_timing_pars)

        for i, par in enumerate(self.timing_parameter_names):
            if par[:2] == "WX":
                self.theta_prior[i] = -self.timing_parameter_values[i]

        # Scale things so max derivative is unity - ensures matrices have nice condition numbers
        self.parameter_scales = np.where(
            self.parameter_scales == 0.0, 1.0, self.parameter_scales
        )

        self.M /= self.parameter_scales[None, :]

        if hasattr(self, "radio_toas"):
            self.Mradio /= self.parameter_scales[None, :]

        self.theta_prior *= self.parameter_scales
        self.theta_prior = np.asarray(self.theta_prior, dtype=float)

    def _orbital_design_matrix(self):

        # This needs some thought

        self.Mo = np.zeros_like(self.M)

        if hasattr(self, "radio_toas"):
            self.Mo_radio = np.zeros_like(self.Mradio)

        c = self.timing_model.components

        b = c[[x for x in c.keys() if x.startswith("Binary")][0]]
        o = b.binary_instance.orbits_cls

        tasc_idx = np.where(self.timing_parameter_names == "TASC")[0][0]
        self.Mo[:, tasc_idx] = 86400.0

        TASC = self.timing_model.TASC.quantity

        try:
            pb_idx = np.where(self.timing_parameter_names == "PB")[0][0]
            PB = self.timing_model.PB.quantity
            self.Mo[:, pb_idx] = ((self.t - TASC.tdb.mjd) * 86400 / PB).to_value("1/d")
            if hasattr(self, "radio_toas"):
                self.Mo_radio[:, pb_idx] = (
                    (self.radio_t - TASC.tdb.mjd) * 86400 / PB
                ).to_value("1/d")

        except:
            fb0_idx = np.where(self.timing_parameter_names == "FB0")[0][0]
            FB0 = self.timing_model.FB0.quantity
            PB = 1.0 / FB0
            self.Mo[:, fb0_idx] = -(
                (self.t - TASC.tdb.mjd) * 86400 * u.s / FB0
            ).to_value("s ** 2")

            if hasattr(self, "radio_toas"):
                self.Mo_radio[:, fb0_idx] = -(
                    (self.radio_t - TASC.tdb.mjd) * 86400 * u.s / FB0
                ).to_value("s ** 2")

        try:
            fb1_idx = np.where(self.timing_parameter_names == "FB1")[0][0]
            self.Mo[:, fb1_idx] = -0.5 * (
                ((self.t - TASC.tdb.mjd) * 86400 * u.s) ** 2 / FB0
            ).to_value("s ** 3")
            if hasattr(self, "radio_toas"):
                self.Mo_radio[:, fb1_idx] = -0.5 * (
                    ((self.radio_t - TASC.tdb.mjd) * 86400 * u.s) ** 2 / FB0
                ).to_value("s ** 3")
        except:
            pass

        if hasattr(self, "radio_toas"):
            self.Mo_radio[:, tasc_idx] = 86400.0
            #
            # try:
            #     self.Mo_radio[:, pb_idx] = (
            #         (self.radio_t - TASC.tdb.mjd) * 86400 / PB
            #     ).to_value("1/d")
            # except:
            #     self.Mo_radio[:, fb0_idx] = (
            #         (self.radio_t - TASC.tdb.mjd) * 86400 / PB
            #     ).to_value("1/d")

            self.Mo_radio /= self.parameter_scales[None, :]

        WOM = self.timing_model.ORBWAVE_OM.quantity.to_value("rad/d")
        WEPOCH = self.timing_model.ORBWAVE_EPOCH.value

        for nf in range(self.nOPVfreqs):
            ii = self.n_timing_pars + 2 * nf
            self.Mo[:, ii] = -PB.to_value("s") * np.sin(
                WOM * (nf + 1) * (self.t - WEPOCH)
            )
            self.Mo[:, ii + 1] = -PB.to_value("s") * np.cos(
                WOM * (nf + 1) * (self.t - WEPOCH)
            )
            if hasattr(self, "radio_toas"):
                self.Mo_radio[:, ii] = -PB.to_value("s") * np.sin(
                    WOM * (nf + 1) * (self.radio_t - WEPOCH)
                )
                self.Mo_radio[:, ii + 1] = -PB.to_value("s") * np.cos(
                    WOM * (nf + 1) * (self.radio_t - WEPOCH)
                )

        self.Mo /= self.parameter_scales[None, :]

    def sample(
        self,
        n_acor_target=100,
        update=100,
        max_iterations=1000000,
        plots=False,
        outputfile="chains",
        resume=False,
    ):
        """
        Runs the Gibbs sampling loop.

        The process will begin by running "tuning" MCMCs with emcee on the
        template pulse profile parameters and hyperparameters to find
        proposal covariance matrices and autocorrelation times for these
        parameters. This step will be skipped if these attributes already exist
        (i.e., if tuning has already been run):
        tau_emcee_cov, tau_autocorr_time,
        lambda_emcee_cov, lambda_autocorr_time

        After these tuning steps, the main sampling loop will run.

        Every so often (after "update" iterations), the autocorrelation of the
        sample chains time will be checked.

        The sampler will run until "ntau_target" autocorrelation times have
        been obtained, or "max_iterations" is reached.


        Parameters
        ----------

        ntau_target : int, default=100
                      Number of autocorrelation times to run for.
                      This determines the effective sample size of the chains.
                      Higher number = more robust results, but slower.

        update      : int, default=100
                      Number of samples between autocorrelation checks.
                      Should not be too small, as this can slow things down

        max_iterations : int, default=1000000
                         Max. number of iterations to run for, regardless of
                         autocorrelation time criterion.
                         Default is very large, and should never really be met!

        plots        : bool, default=False
                       If true, progress plots will be shown,
                       and more text will be output to stdout

        outputfile   : str, default="chains"
                       Results are stored in "<outputfile>.npz"

        resume       : bool, default=False
                       If resume=True, the sampler will load  <outputfile>.npz,
                       and will automatically resume from where it left off.
                       It is up to the user to ensure that the timing model
                       is consistent with the earlier run that is being resumed!

        """

        # template alignment check
        fig, ax = plt.subplots(2, 1, sharex=True, height_ratios=[1, 2], figsize=(5, 10))
        i = np.argsort(self.w)
        p = np.arange(0, 1, 0.001)
        ax[1].scatter(
            np.mod(self.phi, 1.0)[i], self.t[i], 5, c=self.w[i], cmap="binary"
        )
        fermi_lc(np.mod(self.phi, 1.0), self.w, ax=ax[0], xbins=100)
        S = np.sum(self.w**2) / 100
        B = np.sum(self.w * (1 - self.w)) / 100

        if self.Edep:
            template = self.tau_sampler.template(
                self.tau_sampler.tau_0,
                p,
                np.ones_like(p) * np.average(self.log10E, weights=self.w),
            )
        else:
            template = self.tau_sampler.template(self.tau_sampler.tau_0, p)
        ax[0].plot(p, B + S * template, color="orange")
        ax[0].set_xlim(0.0, 1.0)
        ax[0].set_ylim(0.0, B + S * template.max() * 1.05)
        ax[1].set_xlabel("Spin phase")
        ax[1].set_ylabel("Time (MJD)")
        try:
            plt.savefig(outputfile + "_template_alignment_check.png")
        except TypeError:
            pass

        if plots:
            plt.show()
        else:
            plt.close()

        key = None

        if resume is True:
            dat = np.load(outputfile + ".npz")
            self.template_chain = dat["template_chain"]

            tau = self.template_chain[-1]
            self.tau_sampler.tau_0 = np.array(tau)

            self.loglike_chain = dat["loglike_chain"]
            self.timing_chain = dat["timing_chain"]
            if self.has_OPV:
                self.opv_amps_chain = dat["opv_amps_chain"]

                theta = np.concatenate(
                    (self.timing_chain[-1, :], self.opv_amps_chain[-1, :])
                )
            phase_shifts = self.M @ theta
            phase_shifts -= np.mean(phase_shifts)

            if self.nhyp > 0:
                self.hyp_chain = dat["hyp_chain"]
                hyp = self.hyp_chain[-1]
                mu_z, sigma_z, key = self.zm_sampler.sample_z_given_theta_tau(
                    tau, jnp.array(self.phi - phase_shifts), key
                )
                self.timing_sampler.hyp_0 = hyp
            else:
                hyp = np.array([])
                for component in self.noise_models:
                    hyp = np.append(hyp, component.x0)
                self._make_psd_cov(hyp)
        else:
            self.timing_chain = np.zeros((update, self.n_timing_pars))
            tau = self.tau_sampler.tau_0

            if self.has_OPV:
                self.opv_amps_chain = np.zeros((update, self.nOPVfreqs * 2))

            if self.nhyp > 0:
                self.hyp_chain = np.zeros((update, self.nhyp))
                hyp = np.array([])
                for component in self.noise_models:
                    hyp = np.append(hyp, component.x0[component.free])
            else:
                hyp = np.array([])
                for component in self.noise_models:
                    hyp = np.append(hyp, component.x0)
                self._make_psd_cov(hyp)

            self.template_chain = np.zeros((update, len(self.tau_sampler.tau_0)))
            self.loglike_chain = np.zeros(update)
            phase_shifts = np.zeros(self.nphot)

        if not hasattr(self.tau_sampler, "kernel"):
            print("Setting up template sampler")
            tau, key = self.tau_sampler.setup_sampler(self.phi - phase_shifts)

        if self.nhyp > 0:
            self.timing_sampler.hyp_0 = jnp.array(hyp)
            if not hasattr(self.timing_sampler, "kernel"):
                print("Setting up hyper-parameter sampler")
                mu_z, sigma_z, key = self.zm_sampler.sample_z_given_theta_tau(
                    tau, self.phi - phase_shifts, key
                )
                hyp, key = self.timing_sampler.setup_sampler(mu_z, sigma_z)

        c = 0

        if key is None:
            key = jax.random.key(0)
        keys = jax.random.split(key, 10000)

        jphi = jnp.array(self.phi)
        if self.nhyp > 0:
            state = (phase_shifts, tau, hyp)

            @jax.jit
            def gibbs_sampling_loop(state, key):

                phase_shifts = state[0]
                tau = state[1]
                hyp = state[2]

                tau, key = self.tau_sampler.sample_tau_given_theta(
                    tau, jphi - phase_shifts, key
                )
                mu_z, sigma_z, key = self.zm_sampler.sample_z_given_theta_tau(
                    tau, jphi - phase_shifts, key
                )
                hyp, theta, phase_shifts, key = (
                    self.timing_sampler.sample_lambda_theta_given_tau_zm(
                        hyp, mu_z, sigma_z, key
                    )
                )

                return (phase_shifts, tau, hyp), (tau, hyp, theta)

        else:
            state = (phase_shifts, tau)

            inv_prior_cov = jnp.array(self.inv_prior_cov)

            @jax.jit
            def gibbs_sampling_loop(state, key):

                phase_shifts = state[0]
                tau = state[1]

                tau, key = self.tau_sampler.sample_tau_given_theta(
                    tau, jphi - phase_shifts, key
                )
                mu_z, sigma_z, key = self.zm_sampler.sample_z_given_theta_tau(
                    tau, jphi - phase_shifts, key
                )
                theta, phase_shifts, key = (
                    self.timing_sampler.sample_theta_given_tau_zm(mu_z, sigma_z, key)
                )

                return (phase_shifts, tau), (tau, theta)

        progress = tqdm(
            total=n_acor_target * update, desc="Gibbs sampling", smoothing=0.0
        )
        start_time = time.time()
        progress.start_t = start_time
        progress.last_print_t = start_time

        while (c < update or n_acor < n_acor_target) and (c < max_iterations):

            if c == len(self.loglike_chain):
                keys = jax.random.split(key, update + 1)
                key = keys[-1]

                self.loglike_chain = np.append(self.loglike_chain, np.zeros(update))
                self.timing_chain = np.append(
                    self.timing_chain, np.zeros((update, self.n_timing_pars)), axis=0
                )

                self.template_chain = np.append(
                    self.template_chain,
                    np.zeros((update, len(self.tau_sampler.tau_0))),
                    axis=0,
                )

                if self.has_OPV:
                    self.opv_amps_chain = np.append(
                        self.opv_amps_chain,
                        np.zeros((update, self.nOPVfreqs * 2)),
                        axis=0,
                    )
                if self.nhyp > 0:
                    self.hyp_chain = np.append(
                        self.hyp_chain, np.zeros((update, self.nhyp)), axis=0
                    )

            state, samples = gibbs_sampling_loop(state, keys[c % update])

            tau = samples[0]
            if self.nhyp > 0:
                hyp = samples[1]
                theta = samples[2]
            else:
                theta = samples[1]

            self.timing_chain[c] = theta[: self.n_timing_pars]
            self.template_chain[c] = tau

            if self.has_OPV:
                self.opv_amps_chain[c] = theta[self.n_timing_pars :]  # -self.npeaks]

            if self.nhyp > 0:
                self.hyp_chain[c] = hyp

            if c % update == update - 1:

                output_dict = {
                    "loglike_chain": self.loglike_chain[: c + 1],
                    "timing_chain": self.timing_chain[: c + 1],
                    "template_chain": self.template_chain[: c + 1],
                    "parameter_scales": self.parameter_scales,
                    "parameter_names": self.timing_parameter_names,
                    "timing_parameter_values": self.timing_parameter_values,
                    "timing_parameter_uncertainties": self.timing_parameter_uncertainties,
                }

                if self.has_OPV:
                    output_dict_opv = {
                        "opv_amps_chain": self.opv_amps_chain[: c + 1],
                        "opv_freqs": self.OPVfreqs,
                    }
                    output_dict.update(output_dict_opv)

                if self.nhyp > 0:
                    output_dict_hyp = {
                        "hyp_chain": self.hyp_chain[: c + 1],
                    }
                    output_dict.update(output_dict_hyp)

                np.savez(outputfile, **output_dict)

                start = 0

                all_chains = self.timing_chain[start:c]
                # np.concatenate(
                #    (self.timing_chain[start:c], self.template_chain[start:c]), axis=1
                # )

                if self.nhyp > 0:
                    all_chains = np.concatenate(
                        (
                            all_chains,
                            self.hyp_chain[start:c],
                        ),
                        axis=1,
                    )

                if self.has_OPV:
                    all_chains = np.concatenate(
                        (
                            all_chains,
                            self.opv_amps_chain[start:c],
                        ),
                        axis=1,
                    )

                if self.fit_TN:
                    all_chains = np.concatenate(
                        (
                            all_chains,
                            self.hyp_chain[start:c],
                        ),
                        axis=1,
                    )

                acor_time, j = acor(all_chains)

                if plots:
                    print(
                        c,
                        "Autocorr time = ",
                        acor_time,
                        "Estimated steps required:",
                        acor_time * n_acor_target,
                    )
                    sys.stdout.flush()
                else:
                    progress.reset(total=acor_time * n_acor_target)
                    progress.start_t = start_time
                    progress.last_print_t = start_time
                    progress.update(c)
                    progress.refresh()

                n_acor = (c - start) / acor_time

                if plots:
                    ax[0].clear()
                    ax[1].clear()
                    for i in range(self.n_timing_pars):
                        param = (
                            self.timing_chain[:c, i] - np.mean(self.timing_chain[:c, i])
                        ) / np.std(self.timing_chain[:c, i])
                        ax[0].plot(i * 5 + param, color="black")

                    for i in range(self.npeaks * 3):
                        param = (
                            self.template_chain[:c, i]
                            - np.mean(self.template_chain[:c, i])
                        ) / np.std(self.template_chain[:c, i])
                        ax[0].plot(-5 - i * 5 + param, color=f"C{i//self.npeaks}")

                    for i in range(self.nhyp):

                        param = (
                            self.hyp_chain[:c, i] - np.mean(self.hyp_chain[:c, i])
                        ) / np.std(self.hyp_chain[:c, i])
                        ax[0].plot(
                            -5 - i * 5 - (self.npeaks * 3) * 5 + param,
                            color=f"C{self.npeaks + i + 1}",
                        )

                    ax[0].set_xlabel("Iteration")
                    ax[0].set_ylabel("Chains")
                    ax[1].plot(self.loglike_chain[:c])
                    ax[1].set_xlabel("Iteration")
                    ax[1].set_ylabel("Loglikelihood")
                    ax[2].scatter(c, acor_time * n_acor_target, color="red")
                    ax[2].scatter(c, (c - start), color="green", marker="^")
                    ax[2].set_xlabel("Iteration")
                    ax[2].set_ylabel("Estimated iterations required")

                    plt.pause(0.01)
                    plt.draw()
                    plt.pause(0.01)

            c += 1
            if not plots:
                progress.update(1)
                progress.refresh()

        plt.ioff()
        plt.show()

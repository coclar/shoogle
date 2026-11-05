import numpy as np
from tqdm.auto import tqdm
import time, datetime

from astropy import units as u
from astropy.coordinates import Angle

from pint import toa
from pint.models import get_model
from pint.fermi_toas import get_Fermi_TOAs
from pint.observatory.satellite_obs import SatelliteObs
import pint.logging

pint.logging.setup(level="WARNING")

from shoogle.utils import *
from shoogle.noise_models import FlatTailBrokenPowerLaw
from shoogle.conditional_samplers import *

from blackjax.diagnostics import effective_sample_size as ESS_fn
import jax
import jax.numpy as jnp


def read_input_ft1_file(infile, FT2, weightfield, wmin, model):

    try:
        SatelliteObs(name="Fermi", ft2name=FT2)
    except ValueError:
        pass

    toas = get_Fermi_TOAs(infile, weightcolumn=weightfield, minweight=wmin, model=model)

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
        weightfield=None,
        templatefile=None,
        wmin=0.05,
        timfile=None,
        Edep=False,
        priorfile=None,
    ):
        """

        Initialises the Gibbs sampler by:
        * Uses PINT to load in the Fermi FT1 and FT2 files,
          and (optionally) a .tim file containing ToAs.
        * Uses PINT to load the timing model from a .par file,
          folds the data to obtain an initial set of (pre-fit) phases,
          and evaluates the design matrix.
        * Uses the .par file to detect noise-model components and reads the
          prior file to configure/fix/free their hyperparameters.

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
        priorfile    : str, optional
                       File specifying initial values, uncertainties/priors and
                       free/fixed flags for noise-model hyperparameters.
                       If None, no noise-model hyperparameters are included.

        """

        self.parfile = parfile
        self.timing_model = get_model(self.parfile)

        print("Loading FT1 file")

        self.photon_toas, self.w, energies = read_input_ft1_file(
            ft1file, ft2file, weightfield, wmin, self.timing_model
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
                parts = l.split()
                if parts and parts[0] == "LATPHASE":
                    extra_phase = float(parts[1])

        self.Edep = Edep
        if self.Edep:
            self.tau_sampler = EdepTemplateSampler(
                templatefile, self.w, self.log10E, extra_phase=extra_phase
            )
        else:
            self.tau_sampler = TemplateSampler(
                templatefile, self.w, extra_phase=extra_phase
            )

        self.npeaks = self.tau_sampler.npeaks

        if self.Edep:
            self.zm_sampler = LatentVariableSampler(
                self.w, self.npeaks, log10E=self.log10E
            )
        else:
            self.zm_sampler = LatentVariableSampler(self.w, self.npeaks)

        if priorfile is not None:
            with open(priorfile, "r") as pf:
                prior_lines = pf.readlines()
        else:
            prior_lines = []

        print("Constructing design matrix")
        self._make_design_matrix(priorfile)

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
                logKAPPA0 = -20.0
                KAPPA_free = False

                for line in prior_lines:
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
                        logKAPPA0 = float(line.split()[1])
                        KAPPA_free = int(line.split()[2])

                if logA0 is not None:
                    self.Tobs = (self.WXfreqs[0] * u.d) ** (-1)

                    if A_free or FC_free or GAM_free or KAPPA_free:
                        self.fit_TN = True

                    self.noise_models.append(
                        FlatTailBrokenPowerLaw(
                            self.WXfreqs,
                            self.WXSinds,
                            self.WXCinds,
                            np.array(
                                [A_free, FC_free, GAM_free, KAPPA_free], dtype=bool
                            ),
                            self.Tobs,
                            np.array([logA0, logFC0, GAM0, logKAPPA0]),
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
                logKAPPA0 = -20.0
                KAPPA_free = False

                for line in prior_lines:
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
                        logKAPPA0 = float(line.split()[1])
                        KAPPA_free = int(line.split()[2])

                if logA0 is not None:
                    if not hasattr(self, "OPVfreqs"):
                        self._make_opv_frequencies()
                    if A_free or FC_free or GAM_free:
                        self.fit_OPV = True

                    self.noise_models.append(
                        FlatTailBrokenPowerLaw(
                            self.OPVfreqs,
                            self.OPVSinds,
                            self.OPVCinds,
                            [A_free, FC_free, GAM_free, KAPPA_free],
                            self.Tobs,
                            np.array([logA0, logFC0, GAM0, logKAPPA0]),
                            prefix,
                        )
                    )
                    nc += 1

                    if A_free == 2:
                        self.noise_models[-1].linear_amp_prior = True

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

    def _make_design_matrix(self, priorfile=None):

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
        self.theta_prior = np.zeros(self.n_timing_pars)

        priors_dict = {}
        if priorfile is not None:
            with open(priorfile, "r") as pf:
                prior_lines = pf.readlines()
            for line in prior_lines:
                words = line.split()
                if len(words) < 4:
                    continue

                if line.split()[0][0] == "#":
                    continue

                if int(words[2]):

                    if words[0] == "RAJ":
                        value = Angle(words[1], unit="hourangle").hourangle
                        sigma = float(words[3]) / 3600.0
                    elif words[0] == "DECJ":
                        value = Angle(words[1], unit="deg").deg
                        sigma = float(words[3]) / 3600.0
                    else:
                        value = float(words[1])
                        sigma = float(words[3])

                    priors_dict[words[0]] = {"value": value, "sigma": sigma}

        pint_pars = self.timing_model.get_params_dict()
        for i, name in enumerate(self.timing_parameter_names):
            if name == "Offset":
                continue

            par = pint_pars[name]

            self.timing_parameter_values[i] = par.value
            if name in priors_dict:
                self.theta_prior[i] = priors_dict[name]["value"] - par.value
                self.timing_parameter_uncertainties[i] = priors_dict[name]["sigma"]

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
            # Explicitly set ORBWAVE priors to be centered on zero
            self.theta_prior = np.concatenate((self.theta_prior, -orbwaves))
            self.timing_parameter_uncertainties = np.concatenate(
                (self.timing_parameter_uncertainties, np.zeros(len(orbwaves)))
            )

        # Explicitly set WX priors to be centered on zero
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

    def _setup_sampler(self, seed=None, num_NUTS_steps=1):

        # Make sure template is phase-aligned with the data
        logL_max = 0
        logL_fn = jax.jit(self.tau_sampler._log_like)
        for dphi in np.arange(-0.5, 0.5, 0.0001):
            tau = np.copy(self.tau_sampler.tau_0)
            tau[self.npeaks : 2 * self.npeaks] += dphi
            tau[self.npeaks : 2 * self.npeaks] = np.mod(
                tau[self.npeaks : 2 * self.npeaks], 1.0
            )
            if self.Edep:
                tau[4 * self.npeaks : 5 * self.npeaks] += dphi
                tau[4 * self.npeaks : 5 * self.npeaks] = np.mod(
                    tau[4 * self.npeaks : 5 * self.npeaks], 1.0
                )

            logL = logL_fn(jnp.array(tau), jnp.array(self.phi))
            if logL > logL_max:
                logL_max = logL
                tau_opt = jnp.array(tau)

        self.tau_sampler.tau_0 = tau_opt

        print("Pre-fit log-likelihood:", logL_max)

        if seed is None:
            # 32-bit integer seed based on current POSIX microsecond
            seed = np.int32(
                datetime.datetime.now().timestamp() * 1e6 % np.iinfo(np.int32).max
            )
        key = jax.random.key(seed)

        tau = self.tau_sampler.tau_0

        if self.nhyp > 0:
            hyp = np.array([])
            for component in self.noise_models:
                hyp = np.append(hyp, component.x0[component.free])
        else:
            hyp = np.array([])
            for component in self.noise_models:
                hyp = np.append(hyp, component.x0)
            self._make_psd_cov(hyp)

        phase_shifts = jnp.zeros(self.nphot)

        if self.nhyp > 0:
            all_hyp = self.timing_sampler._fill_noise_pars(hyp)
            inv_prior_cov = self.timing_sampler._make_psd_cov(all_hyp)
        else:
            inv_prior_cov = jnp.array(self.inv_prior_cov)

        jphi = jnp.array(self.phi)

        @jax.jit
        def warmup_loop(phase_shifts, tau, key):

            mu_z, sigma_z, key = self.zm_sampler.sample_z_given_theta_tau(
                tau, jphi - phase_shifts, key
            )

            theta, phase_shifts, key = (
                self.timing_sampler.sample_theta_given_lambda_tau_zm(
                    mu_z, sigma_z, inv_prior_cov, key
                )
            )
            logL = self.tau_sampler._log_like(tau, jphi - phase_shifts)

            return phase_shifts, logL, key

        for it in range(100):
            phase_shifts, logL, key = warmup_loop(phase_shifts, tau, key)
            if it == 0:
                logL0 = logL
        print(f"Timing model fit with fixed template, logL : {logL0:.2f} -> {logL:.2f}")

        if not hasattr(self.tau_sampler, "kernel"):
            tau, key = self.tau_sampler.setup_sampler(jphi - phase_shifts, key)
            print()

        if self.nhyp > 0:
            for it in range(100):
                phase_shifts, logL, key = warmup_loop(phase_shifts, tau, key)
                if it == 0:
                    logL0 = logL
            print(
                f"Timing model fit with burnt-in template, logL : {logL0:.2f} -> {logL:.2f}"
            )

            mu_z, sigma_z, key = self.zm_sampler.sample_z_given_theta_tau(
                tau, jphi - phase_shifts, key
            )

            self.timing_sampler.hyp_0 = jnp.array(hyp)
            if not hasattr(self.timing_sampler, "kernel"):
                hyp, key = self.timing_sampler.setup_sampler(mu_z, sigma_z, key)
                print()

        if self.nhyp > 0:
            state = (phase_shifts, tau, hyp)

            @jax.jit
            def gibbs_sampling_loop(state, key):

                phase_shifts = state[0]
                tau = state[1]
                hyp = state[2]

                tau, key, tau_accept0, tau_diverged0, tau_step_size = (
                    self.tau_sampler.sample_tau_given_theta(
                        tau, jphi - phase_shifts, key, num_NUTS_steps
                    )
                )
                mu_z, sigma_z, key = self.zm_sampler.sample_z_given_theta_tau(
                    tau, jphi - phase_shifts, key
                )
                (
                    hyp,
                    theta,
                    phase_shifts,
                    key,
                    hyp_accept0,
                    hyp_diverged0,
                    hyp_step_size,
                ) = self.timing_sampler.sample_lambda_theta_given_tau_zm(
                    hyp, mu_z, sigma_z, key, num_NUTS_steps
                )
                logL = self.tau_sampler._log_like(tau, jphi - phase_shifts)

                return (phase_shifts, tau, hyp), (
                    tau,
                    tau_accept0,
                    tau_diverged0,
                    tau_step_size,
                    theta,
                    logL,
                    hyp,
                    hyp_accept0,
                    hyp_diverged0,
                    hyp_step_size,
                )

        else:
            state = (phase_shifts, tau)

            inv_prior_cov = jnp.array(self.inv_prior_cov)

            @jax.jit
            def gibbs_sampling_loop(state, key):

                phase_shifts = state[0]
                tau = state[1]

                tau, key, tau_accept0, tau_diverged0, tau_step_size = (
                    self.tau_sampler.sample_tau_given_theta(
                        tau, jphi - phase_shifts, key, num_NUTS_steps
                    )
                )
                mu_z, sigma_z, key = self.zm_sampler.sample_z_given_theta_tau(
                    tau, jphi - phase_shifts, key
                )
                theta, phase_shifts, key = (
                    self.timing_sampler.sample_theta_given_lambda_tau_zm(
                        mu_z, sigma_z, inv_prior_cov, key
                    )
                )
                logL = self.tau_sampler._log_like(tau, jphi - phase_shifts)

                return (phase_shifts, tau), (
                    tau,
                    tau_accept0,
                    tau_diverged0,
                    tau_step_size,
                    theta,
                    logL,
                )

        return state, gibbs_sampling_loop, key

    def sample(
        self,
        n_acor_target=1000,
        update=1000,
        max_iterations=1000000,
        plots=False,
        outputfile="chains",
        resume=False,
        num_NUTS_steps=1,
        seed=None,
        chains=16,
    ):
        """
        Runs the Gibbs sampling loop.

        The first call sets up the sampler via _setup_sampler: the template is
        optimised with LBFGS and blackjax NUTS window adaptation is used to tune
        the step size and mass matrix (this is skipped if setup has already run).

        The main loop then alternates template, latent-variable and
        timing/noise-model updates. After every `update` samples (rounded up
        to a multiple of `chains`), the minimum effective sample size (ESS)
        across parameter groups is computed. Sampling stops when that minimum
        ESS reaches `n_acor_target`, or after `max_iterations` samples.


        Parameters
        ----------

        n_acor_target : int, default=1000
                       Target for the minimum effective sample size (ESS)
                       across parameter groups.
                       Higher number = more robust results, but slower.

        update       : int, default=1000
                       Number of samples between ESS checks. Rounded up to a
                       multiple of `chains`.
                       Should not be too small, as this can slow things down

        max_iterations : int, default=1000000
                         Max. number of iterations to run for, regardless of
                         the ESS criterion.
                         Default is very large, and should never really be met!

        plots        : bool, default=False
                       Currently unused; retained for backwards compatibility

        outputfile   : str, default="chains"
                       Results are stored in "<outputfile>_<chain>.npz"

        resume       : bool, default=False
                       Currently unused; retained for backwards compatibility

        num_NUTS_steps : int, default=1
                       Number of NUTS iterations to perform in the intermediate
                       template/hyperparameter sampling steps. Higher numbers are slower
                       but may lead to shorter overall autocorrelation times

        chains       : int, default=16
                       Number of chains to run in parallel.

        seed         : int or None, default=None
                       Random seed. If None, a time-based seed is used.

        """

        if not hasattr(self, "gibbs_sampling_loop"):
            self.initial_state, self.gibbs_sampling_loop, self.initial_key = (
                self._setup_sampler(seed, num_NUTS_steps)
            )

        state = self.initial_state

        progress = tqdm(total=max_iterations, desc="Gibbs sampling", smoothing=0.0)

        def update_progress(steps):
            progress.update(int(steps))

        def sample_chains(states, keys):
            states, samples = jax.vmap(self.gibbs_sampling_loop)(states, keys)
            jax.debug.callback(update_progress, chains)
            return states, samples

        @jax.jit
        def sample_batch(states, keys):
            state, samples = jax.lax.scan(sample_chains, states, keys)
            return state, samples

        start_time = time.time()
        progress.start_t = start_time
        progress.last_print_t = start_time

        update = int(np.ceil(update / chains)) * chains
        max_len = int(np.ceil(max_iterations / update)) * update

        self.timing_chain = jnp.zeros((chains, max_len, self.n_timing_pars))
        self.template_chain = jnp.zeros((chains, max_len, len(self.tau_sampler.tau_0)))
        self.loglike_chain = jnp.zeros((chains, max_len))

        if self.nhyp > 0:
            self.hyp_chain = jnp.zeros((chains, max_len, self.nhyp))

        if self.has_OPV:
            self.opv_amps_chain = jnp.zeros((chains, max_len, 2 * self.nOPVfreqs))

        c = 0
        n_acor = 0

        next_keys = jax.random.split(self.initial_key, chains)

        states = jax.tree.map(
            lambda x: jnp.repeat(jnp.expand_dims(x, axis=0), chains, axis=0), state
        )

        tau_diverged = 0
        hyp_diverged = 0

        headings1 = ["Samples", "Eff. samples", "Eff. samples"]
        headings2 = ["", "(templates)", "(timing)"]
        if self.nhyp > 0:
            headings1 += ["Eff. samples"]
            headings2 += ["(hyperpars)"]
        if self.has_OPV:
            headings1 += ["Eff. samples"]
            headings2 += ["(OPVs)"]

        headings1 += ["Divergences"]
        headings2 += ["(templates)"]

        if self.nhyp > 0:
            headings1 += ["Divergences"]
            headings2 += ["(hyperpars)"]

        colfmt = "{:^14}"
        heading_line1 = "|" + "|".join(colfmt.format(head) for head in headings1) + "|"
        heading_line2 = "|" + "|".join(colfmt.format(head) for head in headings2) + "|"

        tqdm.write(heading_line1)
        tqdm.write(heading_line2)
        tqdm.write("-" * len(heading_line2))

        batch_steps = update // chains

        while (n_acor < n_acor_target) and (c < max_len):
            keys = (
                jax.vmap(jax.random.split, (0, None))(next_keys, (batch_steps + 1))
            ).reshape(batch_steps + 1, chains)
            next_keys = keys[-1]

            states, samples = sample_batch(states, keys[:-1])

            tau = samples[0].swapaxes(0, 1)
            if jnp.any(jnp.isnan(tau)):
                raise ValueError("Error: found a NaN in the template samples")

            tau_diverged += jnp.sum(samples[2])

            theta = samples[4].swapaxes(0, 1)
            logL = samples[5].swapaxes(0, 1)

            if self.nhyp > 0:
                hyp = samples[6].swapaxes(0, 1)
                hyp_diverged += jnp.sum(samples[8])

                if jnp.any(jnp.isnan(hyp)):
                    raise ValueError("Error: found a NaN in the hyperparameter samples")

            if jnp.any(jnp.isnan(theta)):
                raise ValueError("Error: found a NaN in the timing samples")

            self.timing_chain = self.timing_chain.at[:, c : c + batch_steps, :].set(
                theta[:, :, : self.n_timing_pars]
            )
            self.template_chain = self.template_chain.at[:, c : c + batch_steps, :].set(
                tau
            )
            self.loglike_chain = self.loglike_chain.at[:, c : c + batch_steps].set(logL)

            if self.has_OPV:
                self.opv_amps_chain = self.opv_amps_chain.at[
                    :, c : c + batch_steps, :
                ].set(theta[:, :, self.n_timing_pars :])

            if self.nhyp > 0:
                self.hyp_chain = self.hyp_chain.at[:, c : c + batch_steps, :].set(hyp)

            for chain in range(chains):
                output_dict = {
                    "seed": seed,
                    "chain_index": chain,
                    "loglike_chain": self.loglike_chain[chain, : c + batch_steps],
                    "timing_chain": self.timing_chain[chain, : c + batch_steps],
                    "template_chain": self.template_chain[chain, : c + batch_steps],
                    "parameter_scales": self.parameter_scales,
                    "parameter_names": self.timing_parameter_names,
                    "timing_parameter_values": self.timing_parameter_values,
                    "timing_parameter_uncertainties": self.timing_parameter_uncertainties,
                }

                if self.has_OPV:
                    output_dict_opv = {
                        "opv_amps_chain": self.opv_amps_chain[chain, : c + batch_steps],
                        "opv_freqs": self.OPVfreqs,
                    }
                    output_dict.update(output_dict_opv)

                if self.nhyp > 0:
                    output_dict_hyp = {
                        "hyp_chain": self.hyp_chain[chain, : c + batch_steps]
                    }
                    output_dict.update(output_dict_hyp)

                of = outputfile + f"_{chain:d}"
                np.savez(of, **output_dict)

            ESS_tau = int(ESS_fn(self.template_chain[:, : c + batch_steps]).min())
            ESS_theta = int(ESS_fn(self.timing_chain[:, : c + batch_steps]).min())

            vals = [(c + batch_steps) * chains, ESS_tau, ESS_theta]

            ESS_all = jnp.array([ESS_theta])

            if self.nhyp > 0:
                ESS_hyp = int(ESS_fn(self.hyp_chain[:, : c + batch_steps]).min())
                ESS_all = jnp.append(ESS_all, ESS_hyp)
                vals.append(ESS_hyp)

            if self.has_OPV:
                ESS_OPV = int(ESS_fn(self.opv_amps_chain[:, : c + batch_steps]).min())
                ESS_all = jnp.append(ESS_all, ESS_OPV)
                vals.append(ESS_OPV)

            vals.append(tau_diverged)
            if self.nhyp > 0:
                vals.append(hyp_diverged)
            data_fmt = "{:>14}"
            info_line = "|" + "|".join(data_fmt.format(val) for val in vals) + "|"
            tqdm.write(info_line)

            ESS_min = np.min(ESS_all)
            n_acor = ESS_min
            c += batch_steps

            acor_time = c / n_acor

            progress.reset(
                total=int(jnp.ceil(acor_time * n_acor_target / batch_steps))
                * batch_steps
                * chains
            )
            progress.start_t = start_time
            progress.last_print_t = start_time

            progress.update(c * chains)
            progress.refresh()

import numpy as np

from astropy import units as u
from scipy.special import gamma, beta
from sklearn.gaussian_process.kernels import Matern


def matern_psd(f, H, LAM, NU):

    # if NU == 0.0:
    #     model = (H ** 2 * np.sqrt(2.0 * np.pi)
    #              * LAM
    #              * np.exp(-2.0 * np.pi**2 * f ** 2 * LAM ** 2))

    # else:
    model = (
        H**2
        * 2
        * np.sqrt(np.pi)
        * gamma(NU + 1 / 2)
        * (2 * NU) ** NU
        / gamma(NU)
        / LAM ** (2 * NU)
        * (2.0 * NU / LAM**2 + 4.0 * np.pi**2 * f**2) ** -(NU + 1 / 2)
    )

    return model


class NoiseModel(object):

    # Needs to be specified by individual models
    npar = None

    def __init__(self, freqs, Sinds, Cinds, free, Tobs, x0, prefix):

        self.freqs = freqs
        self.Cinds = Cinds
        self.Sinds = Sinds
        self.Tobs = Tobs

        self.x0 = x0

        self.free = np.array(free, dtype="bool")
        self.nfree = sum(free)

        self.linear_amp_prior = False

        if len(self.x0) != len(self.free) != self.npar:
            raise ValueError("Incorrect number of hyperparameters specified")

        self.prefix = prefix
        self.linear_amp_prior = False

        return

    def cov(self, t, pars):

        raise NotImplementedError

    def powspec(self, pars):

        raise NotImplementedError

    def log_prior(self, pars):

        raise NotImplementedError

    def all_parameters(self, x):
        """
        Returns a full set of hyperparameters for this model,
        given only values for "free" parameters that need to be updated,
        taking the initial default values for the others.
        """

        if self.nfree == self.npar:
            p = np.copy(x)
         
        else:
            c = 0
            p = np.copy(self.x0)
            for i in range(self.npar):
                if self.free[i]:
                    p[i] = x[c]
                    c += 1

        for i in range(self.npar):
            if p[i] < self.bounds[i][0] or p[i] > self.bounds[i][1]:
                return None

        return p


class BPLMatern(NoiseModel):

    def __init__(self, *pars):
        self.npar = 3
        self.param_names = ["logH", "logLAM", "NU"]

        super().__init__(*pars)

        self.bounds = [[-6, 6], [1.5, np.log10(2 * self.Tobs)], [0.5, 5]]

    def cov(self, t, pars):

        H, LAM, NU = self.matern_hyp_log_to_lin(pars)

        t = np.asarray(np.atleast_1d(t)[:, None], dtype="float")

        d = t - t.T

        if NU == 0.0:
            K = H**2 * np.exp(-0.5 * ((t - t.T) / LAM) ** 2)
        else:
            try:
                C = Matern(length_scale=LAM, nu=NU)
                K = H**2 * C(t, t)
            except:
                C = Matern(length_scale=LAM, nu=1.0)
                K = H**2 * C(t, t)

        # White noise component (for stability with strong timing noise)
        if len(pars) == 4:
            W = pars[3]
            K += np.where((t - t.T) == 0.0, W**2, 0.0)

        return K

    def log_prior(self, pars):

        return 0.0

    def powspec(self, pars, f=None):

        H, LAM, NU = self.matern_hyp_log_to_lin(pars)

        if f is None:
            f = self.freqs

        # 2.0 for one-sided PSD
        psd = 2.0 * matern_psd(f.to_value("1/d"), H, LAM, NU)

        return psd

    def matern_hyp_log_to_lin(self, pars):

        logH = pars[0]
        logLAM = pars[1]
        NU = pars[2]

        LAM = 10**logLAM
        Hf = 10**logH

        ratio = np.sqrt(matern_psd(1.0 / self.Tobs, 1, LAM, NU)) / np.sqrt(
            matern_psd(0, 1, LAM, NU)
        )

        return Hf / ratio, LAM, NU

    def matern_hyp_lin_to_log(self, H, LAM, NU):

        ratio = np.sqrt(matern_psd(1.0 / self.Tobs, 1, LAM, NU)) / np.sqrt(
            matern_psd(0, 1, LAM, NU)
        )

        return np.log10(H * ratio), np.log10(LAM), NU


class BrokenPowerLaw(NoiseModel):

    def __init__(self, *pars):
        self.npar = 3
        self.param_names = ["REDAMP", "REDFC", "REDGAM"]

        super().__init__(*pars)

        self.bounds = [
            [-20, -10],
            [
                -np.log10(self.Tobs * (u.d / u.yr)) - 1,
                -np.log10(self.Tobs * (u.d / u.yr)) + 2,
            ],
            [1, 9],
        ]

        if "OPV" in self.prefix:
            self.bounds[0] = [-15, -5]
            self.bounds[2] = [1.0, 9.0]

    def cov(self, t, pars):

        H, LAM, NU = self.bpl_to_matern(pars)

        t = np.asarray(np.atleast_1d(t)[:, None], dtype="float")

        d = t - t.T

        if NU == 0.0:
            K = H**2 * np.exp(-0.5 * ((t - t.T) / LAM) ** 2)
        else:
            try:
                C = Matern(length_scale=LAM, nu=NU)
                K = H**2 * C(t, t)
            except:
                C = Matern(length_scale=LAM, nu=1.0)
                K = H**2 * C(t, t)

        return K

    def log_prior(self, pars):

        if self.linear_amp_prior:
            return pars[0]

        return 0.0

    def powspec(self, pars, f=None):

        fyr = 1.0 / u.yr

        A = 10.0 ** pars[0]
        fc = (10.0 ** pars[1]) / u.yr
        GAM = pars[2]

        norm = A**2 / (12 * np.pi**2 * fyr**3) * (fc / fyr) ** (-GAM)

        if f is None:
            f = self.freqs

        psd = norm * (1 + (f / fc) ** 2) ** (-GAM / 2)

        return psd.to_value("s ** 2 * d")

    def bpl_to_matern(self, pars):

        A = 10.0 ** pars[0]
        fc = (10.0 ** pars[1]) / u.yr
        GAM = pars[2]
        fyr = 1.0 / u.yr

        norm = A**2 / (12 * np.pi**2 * fyr**3) * (fc / fyr) ** (-GAM)

        NU = GAM / 2 - 0.5

        # Divided by 2 because PSD is one-sided
        H2 = norm * fc * beta(NU, 0.5) / 2.0
        H = np.sqrt(H2.to_value("s ** 2"))

        LAM = (np.sqrt(NU) / (np.sqrt(2) * np.pi * fc)).to_value("d")

        return H, LAM, NU


class FlatTailBrokenPowerLaw(NoiseModel):

    def __init__(self, *pars):
        self.npar = 4
        self.param_names = ["REDAMP", "REDFC", "REDGAM", "REDKAPPA"]

        super().__init__(*pars)

        self.bounds = [
            [-20, -10],
            [
                -np.log10(self.Tobs * (u.d / u.yr)) - 1,
                -np.log10(self.Tobs * (u.d / u.yr)) + 2,
            ],
            [1, 9],
            [-20, -9],
        ]

        if "OPV" in self.prefix:
            self.bounds[0] = [-15, -5]
            self.bounds[2] = [1.0, 9.0]


    """
    def cov(self, t, pars):

        H, LAM, NU = self.bpl_to_matern(pars)

        t = np.asarray(np.atleast_1d(t)[:, None], dtype="float")

        d = t - t.T

        if NU == 0.0:
            K = H**2 * np.exp(-0.5 * ((t - t.T) / LAM) ** 2)
        else:
            C = Matern(length_scale=LAM, nu=NU)
            K = H**2 * C(t, t)

        return K
    """

    def log_prior(self, pars):

        if self.linear_amp_prior:
            return pars[0]

        return 0.0

    def powspec(self, pars, f=None):

        fyr = 1.0 / u.yr

        A = 10.0 ** pars[0]
        fc = (10.0 ** pars[1]) / u.yr
        GAM = pars[2]
        log10_kappa = pars[3]

        norm = A**2 / (12 * np.pi**2 * fyr**3) * (fc / fyr) ** (-GAM)

        if f is None:
            f = self.freqs

        psd = norm * (1 + (f / fc) ** 2) ** (-GAM / 2)
        flat = 10 ** (2 * log10_kappa) * u.yr**3
        psd = np.maximum(psd, flat)

        return psd.to_value("s ** 2 * d")

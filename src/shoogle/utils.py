from scipy.special import gamma, logsumexp
import numpy as np
from sklearn.gaussian_process.kernels import Matern
from pint.observatory.satellite_obs import SatelliteObs
from pint.fermi_toas import load_Fermi_TOAs
from emcee.autocorr import function_1d
from astropy import units as u


def GWB_psd(f_d, log10Agwb, gamma=-13 / 3):

    Agwb = 10**log10Agwb

    f_yr = np.array([f.to_value("yr ** -1") for f in f_d])

    P = (Agwb**2 / (12 * np.pi**2) * (f_yr**gamma)) * (u.yr**3)

    return P


def model_psd(f, H, LAM, NU=0):

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


def log10Hf_to_H0(f, logHf, LAM, NU):

    Hf = 10**logHf

    ratio = np.sqrt(model_psd(f, 1, LAM, NU)) / np.sqrt(model_psd(0, 1, LAM, NU))

    return Hf / ratio


def H0_to_log10Hf(f, H, LAM, NU):

    ratio = np.sqrt(model_psd(f, 1, LAM, NU)) / np.sqrt(model_psd(0, 1, LAM, NU))

    return np.log10(H * ratio)


def cov(x, y, H, LAM, NU, N=0.0):
    """Simple squared-exponential covariance function"""

    x = np.asarray(np.atleast_1d(x)[:, None], dtype="float")
    y = np.asarray(np.atleast_1d(y)[:, None], dtype="float")

    d = x - y.T

    if NU == 0.0:
        K = H**2 * np.exp(-0.5 * ((x - y.T) / LAM) ** 2)
    else:
        C = Matern(length_scale=LAM, nu=NU)
        K = H**2 * C(x, y)

    # White noise component (for stability with strong timing noise)
    K += np.where((x - y.T) == 0.0, N**2, 0.0)

    return K


def write_template(proffile, A, mu, sigma):

    with open(proffile, "w") as prof:
        prof.write("# gauss\n")
        prof.write("-------------------------\n")
        for k in range(len(A)):
            fwhm = np.sqrt(2 * np.log(2)) * sigma[k] * 2
            prof.write(f"phas{k+1} = {np.mod(mu[k],1.0):.8f} +/- 0.00000000\n")
            prof.write(f"fwhm{k+1} = {fwhm:.8f} +/- 0.00000000\n")
            prof.write(f"ampl{k+1} = {A[k]:.8f} +/- 0.00000000\n")
        prof.write(f"const = {1 - np.sum(A):.8f} +/- 0.00000000\n")
        prof.write("-------------------------")

    return


def _auto_window(taus, c):
    m = np.arange(len(taus)) < c * taus
    if np.any(m):
        return np.argmin(m)
    return len(taus) - 1


def acor(chain):

    ndim = np.shape(chain)[1]
    length = np.shape(chain)[0]

    maxtau = 0
    slowest_param = 0

    acor = np.zeros_like(chain[:, 0])
    for j in range(ndim):

        acor = function_1d(chain[:, j])

        taus = 2.0 * np.cumsum(acor) - 1.0
        window = _auto_window(taus, 10.0)
        tau = taus[window]
        if tau > maxtau:
            maxtau = int(np.ceil(tau))
            slowest_param = j

    return maxtau, slowest_param

import numpy as np
import jax
from jax.scipy.linalg import cholesky, cho_solve, solve
import jax.numpy as jnp
import blackjax
from scipy.special import logit, expit
from pint.templates.lctemplate import LCTemplate, prim_io
from tqdm.auto import tqdm
from astropy import units as u

ONE_OVER_SQRT2PI = 1.0 / (jnp.sqrt(2 * jnp.pi))
YR3_TO_S2D = (1.0 * u.yr**3).to_value("s ** 2 * d")
INVYR_TO_INVDAY = (1.0 / u.yr).to_value("1/d")


class TemplateSampler(object):
    """A class providing the MCMC fitter used to draw samples of the template
    pulse profile parameters, given the current timing model, within the Gibbs
    sampling procedure.

    It is implemented in JAX, for speed and to enable the use of the blackjax
    NUTS sampler, which provides extremely efficient MCMC sampling.
    """

    def __init__(self, proffile, weights, minsigma=0.005, maxsigma=0.25, maxwraps=2):
        """
        Initialises the object, including reading a starting estimate for
        the profile parameters from a file.

        Parameters
        ----------
        proffile   : str
                     Text file containing the initial pulse profile shape.
                     This can be obtained using itemplate.py
        maxwraps   : int, default = 2
                     Maximum number of phase wraps to sum over.
                     More wraps = more accurate, but slower.
        minsigma   : float, default = 0.005
                     Minimum width of a Gaussian peak in the pulse profile,
                     in fractions of a rotation. (default = 0.5% of a rotation)
        maxsigma   : float, default = 0.25
                     Minimum width of a Gaussian peak in the pulse profile,
                     in fractions of a rotation. (default = 25% of a rotation)
        """
        prims, norms = prim_io(proffile)

        self.A_0 = (
            np.array([n for n in norms]) * 0.999
        )  # This prevents nasty numerical issues...
        self.mu_0 = np.array([p.get_location() for p in prims])
        self.sigma_0 = np.array([p.get_width() for p in prims])

        self.tau_0 = jnp.array(np.concatenate((self.A_0, self.mu_0, self.sigma_0)))
        self.npeaks = len(prims)

        self.maxwraps = maxwraps
        self.wraps = jnp.arange(-2, 3)

        self.minlogsigma = jnp.log(minsigma)
        self.maxlogsigma = jnp.log(maxsigma)
        self.w = jnp.array(weights)

    def template(self, tau, phases):

        assert (
            np.sum(tau[: self.npeaks]) <= 1.0
        ), "Error: sum of template amplitudes > 1"

        tau = jnp.array(tau)
        phases = jnp.array(phases)

        return self._jax_template(tau, phases)

    def _jax_template(self, tau, phases):
        """
        Evaluates the pulse profile template at the provided phases.

        Parameters
        ----------

        tau        : jax.numpy.ndarray of dtype float32, shape (3 * npeaks,)
                     Template pulse profile parameters, in order:
                     (amplitudes, positions, widths)

                     Amplitudes should sum to < 1.0
                     Positions should be between (0,1)
                     Widths should be between the min and max values.

                     For speed, these conditions are *not* checked here.
                     The prior transforms used for sampling guarantee them.

        phases     : jax.numpy.ndarray of dtype float32
                     Array of photon phases. Should all be between 0 and 1.

        Returns
        ---------

        profile    : jax.numpy.ndarray
                     The pulse profile, evaluated at the input phases.
        """

        A = tau[: self.npeaks]
        mu = tau[self.npeaks : self.npeaks * 2]
        sigma = tau[self.npeaks * 2 :]

        # Axes are (photons, peaks, wraps)
        wrapped_resids = (
            phases[:, None, None] - mu[None, :, None] - self.wraps[None, None, :]
        )

        # Unpulsed component
        U = 1.0 - jnp.sum(A)

        profile = U + jnp.sum(
            (A * ONE_OVER_SQRT2PI / sigma)[None, :]
            * jnp.sum(
                jnp.exp(-0.5 * (wrapped_resids / sigma[None, :, None]) ** 2), axis=2
            ),
            axis=1,
        )

        return profile

    def _samples_to_phys(self, x):
        """
        Converts from parameters convenient for the sampler,
        which are logit-transformed to ranges of (-inf,inf),
        to physical parameters describing the pulse profile peaks.

        Parameters
        ----------

        x    :   jax.numpy.ndarray of type float32
                 Location of the current point in the sample parameter space

        Returns
        ---------

        tau          : jax.numpy.ndarrays of type float32
                       Pulse profile parameters to pass to self._jax_template

        logprior     : jnp.float32
                       the log-prior for these parameters, accounting for both
                       the priors (Dirichlet prior on amplitudes),
                       and the Jacobian of the logit transforms.
        """

        logprior = 0

        Bk = jax.scipy.special.expit(x[: self.npeaks])
        logprior += jnp.sum(jnp.log(Bk * (1 - Bk)))
        logprior += jax.scipy.stats.beta.logpdf(Bk[0], 1.0, self.npeaks)

        A = Bk.copy()
        U = 1 - A[0]
        for p in range(1, self.npeaks):
            logprior += jax.scipy.stats.beta.logpdf(Bk[p], 1.0, self.npeaks - p)
            A = A.at[p].set(Bk[p] * U)
            U -= A[p]

        mu = jax.scipy.special.expit(x[self.npeaks : 2 * self.npeaks])
        logprior += jnp.sum(jnp.log(mu * (1 - mu)))

        logsigma01 = jax.scipy.special.expit(x[2 * self.npeaks : 3 * self.npeaks])
        logprior += jnp.sum(jnp.log(logsigma01 * (1 - logsigma01)))
        sigma = jnp.exp(
            logsigma01 * (self.maxlogsigma - self.minlogsigma) + self.minlogsigma
        )

        return jnp.concatenate((A, mu, sigma)), logprior

    def _log_like(self, tau, phases):
        """
        Computes the log-likelihood of the observed photon phases given the
        pulse profile template parameters and photon weights.

        Parameters
        ----------
        tau        : jax.numpy.ndarray of dtype float32
                     Array of pulse profile parameters
        phases     : jax.numpy.ndarray of dtype float32
                     Array of observed photon phases (in [0, 1]).

        Returns
        -------
        logL       : jnp.float32
                     The log-likelihood of the data given the template parameters.
                     This is the sum over photons of log(weights * profile + (1 - weights)),
                     where the profile is the evaluated template.
        """

        profile = self._jax_template(tau, phases)

        logL = jnp.sum(jnp.log(self.w * profile + (1 - self.w)))

        return logL

    def _log_post(self, x, phases):
        """
        Computes the log-posterior probability of the template parameters
        given the observed photon phases and weights.

        This combines the log-prior (from `self.samples_to_phys`) and the
        log-likelihood (from `self.log_like`) evaluated at the current parameter
        values.

        Parameters
        ----------
        x          : jax.numpy.ndarray of dtype float32
                     Array of unconstrained (logit-transformed) parameters
                     representing the current point in the parameter space.
                     Shape: (3 * npeaks,)
        phases     : jax.numpy.ndarray of dtype float32
                     Array of observed photon phases (in [0, 1]).
        weights    : jax.numpy.ndarray of dtype float32
                     Array of photon probability weights (in [0, 1]).

        Returns
        -------
        log_post   : jnp.float32
                     The log-posterior probability: log_prior + log_likelihood.
        """
        tau, logprior = self._samples_to_phys(x)
        loglike = self._log_like(tau, phases)

        return logprior + loglike

    def _phys_to_samples(self, tau):

        # Add a tiny bit of unpulsed component to prevent numerical issues
        # with the logit transform -> inf when U is close to 1.0

        A = tau[: self.npeaks]
        mu = tau[self.npeaks : self.npeaks * 2]
        sigma = tau[self.npeaks * 2 :]

        Bk = A.copy()

        # Transform to beta-distributed variables with (0,1)
        # which ensures a Dirichlet prior on amplitudes + unpulsed comp.
        U = 1 - Bk[0]
        for p in range(1, self.npeaks):
            Bk = Bk.at[p].set(A[p] / U)
            U -= A[p]

        # Uniform prior on log(sigma), transformed to (0,1)
        logsigma01 = (jnp.log(sigma) - self.minlogsigma) / (
            self.maxlogsigma - self.minlogsigma
        )

        # then logit transformed -> (-inf,inf)
        x = jax.scipy.special.logit(jnp.concatenate((Bk, mu, logsigma01)))

        return x

    def setup_sampler(self, phases):
        """
        Performs a warm-up run to tune parameters (step size, covariance matrix)
        of the blackjax NUTS sampler for efficient sampling later.

        This also generates the

        Parameters
        ----------
        phases     : jax.numpy.ndarray of dtype float32
                     Array of observed photon phases (in [0, 1]).
        """

        phases = jnp.array(phases)

        logprob_fn = lambda x: self._log_post(x, phases)

        warmup = blackjax.window_adaptation(
            blackjax.nuts, logprob_fn, progress_bar=True
        )

        x0 = self._phys_to_samples(self.tau_0)

        rng_key = jax.random.key(0)
        rng_key, warmup_key, sample_key = jax.random.split(rng_key, 3)
        (state, parameters), _ = warmup.run(warmup_key, x0, num_steps=1000)

        self.nuts_params = parameters
        self.logprob_fn = jax.jit(self._log_post)
        self.sample_key = sample_key

        logprob_fn = lambda x: self._log_post(x, phases)

        step_fn = blackjax.nuts.build_kernel()

        @jax.jit
        def kernel(rng_key, state, phases):

            logprob_fn = lambda x: self._log_post(x, phases)
            state, _ = step_fn(
                rng_key=rng_key,
                state=state,
                logdensity_fn=logprob_fn,
                **self.nuts_params,
            )

            return state, (self._samples_to_phys(state.position)[0], state.logdensity)

        self.kernel = kernel
        tau, logprior = self._samples_to_phys(state.position)

        return np.array(tau), sample_key

    def sample(self, tau, phases, key, num_samples=1000):
        """
        Run the blackjax NUTS sampler to estimate parameters of the template

        Parameters
        ----------

        tau       :   np.ndarray of dtype float
                      Template pulse profile parameters

        phases    :   np.ndarray of dtype float
                      Photon phases

        num_samples : Number of samples to draw, default = 1000

        Returns
        ----------
        samples   :   jnp.ndarray of dtype float32
                      NUTS samples, in the sampling parameter space
                      These can be converted to physical parameters
                      using self._samples_to_phys
        """

        x0 = self._phys_to_samples(tau)

        phases = jnp.array(phases)

        logprob_fn = lambda x: self.logprob_fn(x, phases)
        state = blackjax.nuts.init(x0, logprob_fn)

        one_step = lambda state, key: self.kernel(key, state, phases)
        tau_keys = jax.random.split(key, num_samples + 1)

        _, samples = jax.lax.scan(one_step, state, tau_keys[:-1])

        return samples, tau_keys[-1]

    def sample_tau_given_theta(self, tau, phases, key, num_steps=10):

        samples, key = self.sample(tau, phases, key, num_steps)
        return samples[0][-1], key


class EdepTemplateSampler(TemplateSampler):

    def __init__(
        self, proffile, weights, log10E, minsigma=0.005, maxsigma=0.25, maxwraps=2
    ):

        super().__init__(proffile, weights, minsigma, maxsigma, maxwraps)

        self.tau_0 = np.concatenate((self.tau_0, self.tau_0))
        self.log10E = jnp.array(log10E)
        self.log10E_frac = (self.log10E - self.log10E.min()) / (
            self.log10E.max() - self.log10E.min()
        )

    def template(self, tau, phases, log10E):

        assert (
            np.sum(tau[: self.npeaks]) <= 1.0
        ), "Error: sum of template amplitudes > 1"

        tau = jnp.array(tau)
        phases = jnp.array(phases)

        return self._jax_template(tau, phases, log10E)

    def _jax_template(self, tau, phases, log10E=None):

        if log10E is None:
            log10E_frac = self.log10E_frac
        else:
            log10E_frac = (log10E - self.log10E.min()) / (
                self.log10E.max() - self.log10E.min()
            )

        tau_lo = tau[: 3 * self.npeaks]
        tau_hi = tau[3 * self.npeaks :]

        tau_E = tau_lo[None, :] + log10E_frac[:, None] * (tau_hi - tau_lo)[None, :]

        A_E = tau_E[:, : self.npeaks]
        mu_E = tau_E[:, self.npeaks : 2 * self.npeaks]
        sigma_E = tau_E[:, 2 * self.npeaks : 3 * self.npeaks]

        U_E = 1.0 - jnp.sum(A_E, axis=1)

        # Axes are (photons, peaks, wraps)
        wrapped_resids = (
            phases[:, None, None] - mu_E[:, :, None] - self.wraps[None, None, :]
        )

        profile = U_E + jnp.sum(
            (A_E * ONE_OVER_SQRT2PI / sigma_E)
            * jnp.sum(
                jnp.exp(-0.5 * (wrapped_resids / sigma_E[:, :, None]) ** 2), axis=2
            ),
            axis=1,
        )

        return profile

    def _samples_to_phys(self, x):

        tau_lo, logprior_lo = super()._samples_to_phys(x[: 3 * self.npeaks])
        tau_hi, logprior_hi = super()._samples_to_phys(x[3 * self.npeaks :])

        tau = jnp.concatenate((tau_lo, tau_hi))

        return tau, logprior_lo + logprior_hi

    def _phys_to_samples(self, tau):

        x_lo = super()._phys_to_samples(tau[: 3 * self.npeaks])
        x_hi = super()._phys_to_samples(tau[3 * self.npeaks :])

        x = jnp.concatenate((x_lo, x_hi))

        return x


class LatentVariableSampler(object):

    def __init__(self, weights, npeaks, log10E=None, maxwraps=2):

        self.w = weights
        self.maxwraps = maxwraps
        self.npeaks = npeaks
        self.wraps = jnp.arange(-self.maxwraps, self.maxwraps + 1, 1)

        self.key = jax.random.key(0)

        if log10E is not None:
            self.log10E = log10E
            self.log10E_frac = (self.log10E - self.log10E.min()) / (
                self.log10E.max() - self.log10E.min()
            )

        return

    def Einterp_tau(self, tau):

        tau_lo = tau[: 3 * self.npeaks]
        tau_hi = tau[3 * self.npeaks :]

        tau_E = tau_lo[None, :] + self.log10E_frac[:, None] * (tau_hi - tau_lo)[None, :]

        return tau_E

    def prior_z_given_tau(self, tau):

        if len(tau) // 3 == 2 * self.npeaks:
            tau_E = self.Einterp_tau(tau)
            A = tau_E[:, : self.npeaks]
        else:
            A = tau[: self.npeaks][None, :]

        # Prior probability that each photon came from each peak
        A_ik = self.w[:, None] * A

        # Prior probability that each photon came from the pulsar's unpulsed component
        U_i = self.w * (1 - jnp.sum(A, axis=1))

        # Prior probability that the photon came from the background
        B_i = 1 - self.w

        norm = jnp.sum(A_ik, axis=1) + U_i + B_i  # should be unity

        return A_ik, U_i, B_i

    def likelihood_z_given_theta_tau(self, tau, phases):

        if len(tau) // 3 == 2 * self.npeaks:
            tau_E = self.Einterp_tau(tau)
            A = tau_E[:, : self.npeaks]
            mu = tau_E[:, self.npeaks : 2 * self.npeaks]
            sigma = tau_E[:, 2 * self.npeaks : 3 * self.npeaks]
        else:
            A = tau[: self.npeaks][None, :]
            mu = tau[self.npeaks : self.npeaks * 2][None, :]
            sigma = tau[self.npeaks * 2 :][None, :]

        # Use wrapped Gaussian function to work out photon--component probabilities
        # Wraps for individual photons will be randomly assigned in a later stage

        log_like = (
            -jnp.log(sigma)[:, :, None]
            - 0.5 * jnp.log(2 * np.pi)
            - 0.5
            * (
                (phases[:, None, None] - mu[:, :, None] - self.wraps[None, None, :])
                / sigma[:, :, None]
            )
            ** 2
        )

        return log_like, mu, sigma

    def post_z_given_theta_tau(self, tau, phases):

        A_ik, U_i, B_i = self.prior_z_given_tau(tau)
        log_like, mu, sigma = self.likelihood_z_given_theta_tau(tau, phases)

        # Mixture weights
        # Axes are (photons, peaks, wraps)
        log_rho_ik = jnp.log(A_ik)[:, :, None] + log_like

        # This sums over phase wraps for each component
        rho_ik = jnp.sum(jnp.exp(log_rho_ik), axis=-1)

        sum_rho = jnp.sum(rho_ik, axis=1) + U_i + B_i
        rho_ik = rho_ik / sum_rho[:, None]
        rho_iU = U_i / sum_rho
        rho_iB = B_i / sum_rho

        rho = jnp.concatenate((rho_ik, rho_iU[:, None], rho_iB[:, None]), axis=1)

        return A_ik, U_i, B_i, log_like, rho, mu, sigma

    def sample_z_given_theta_tau(self, tau, phases, key):

        A_ik, U_i, B_i, log_like, rho, mu, sigma = self.post_z_given_theta_tau(
            tau, phases
        )

        C = jnp.cumsum(rho, axis=1)

        z_key, m_key, next_key = jax.random.split(key, 3)

        zr = jax.random.uniform(z_key, (len(phases),))
        mr = jax.random.uniform(m_key, (len(phases),))

        z = jnp.sum(C < zr[:, None], axis=1)

        mu_z = jnp.zeros(len(phases))
        sigma_z = jnp.zeros(len(phases))
        for k in range(self.npeaks):
            mask = z == k

            log_rho_wraps = log_like[:, k]

            rho_wraps = jnp.exp(log_rho_wraps)
            rho_wraps /= jnp.sum(rho_wraps, axis=1)[:, None]

            C_wraps = jnp.cumsum(rho_wraps, axis=1)

            m = jnp.sum(C_wraps < mr[:, None], axis=1)

            if len(tau) // 3 == 2 * self.npeaks:
                mu_z = jnp.where(mask, mu[:, k] + self.wraps[m], mu_z)
                sigma_z = jnp.where(mask, sigma[:, k], sigma_z)
            else:
                mu_z = jnp.where(mask, mu[0, k] + self.wraps[m], mu_z)
                sigma_z = jnp.where(mask, sigma[0, k], sigma_z)

        return mu_z, sigma_z, next_key


class TimingModelSampler(object):

    def __init__(self, phi, M, theta_prior):

        self.phi = phi
        self.npar = np.shape(M)[1]
        self.M = jnp.array(M)
        self.theta_prior = jnp.array(theta_prior)

    def setup_leastsq_given_z_tau(self, mu_z, sigma_z):

        resids = self.phi - mu_z
        precisions = jnp.where(sigma_z > 0, 1.0 / sigma_z**2, 0.0)

        MT_Sigma_inv_M = jnp.einsum("ji,j,jk->ik", self.M, precisions, self.M)
        MT_Sigma_inv_R = jnp.einsum("ji,j,j->i", self.M, precisions, resids)

        return MT_Sigma_inv_M, MT_Sigma_inv_R

    def solve_leastsq(self, MT_Sigma_inv_M, MT_Sigma_inv_R, inv_prior_cov):

        post_cov_inv = jnp.asarray(MT_Sigma_inv_M + inv_prior_cov, dtype=jnp.float64)
        post_cov_inv_U = cholesky(post_cov_inv, lower=False)

        theta_opt = cho_solve(
            (post_cov_inv_U, False), MT_Sigma_inv_R + inv_prior_cov @ self.theta_prior
        )

        return theta_opt, post_cov_inv_U

    def sample_theta_given_lambda_z_m_tau(self, mu_z, sigma_z, inv_prior_cov, key):

        MT_Sigma_inv_M, MT_Sigma_inv_R = self.setup_leastsq_given_z_tau(mu_z, sigma_z)
        theta_opt, post_cov_inv_U = self.solve_leastsq(
            MT_Sigma_inv_M, MT_Sigma_inv_R, inv_prior_cov
        )

        theta_key, key = jax.random.split(key)
        p = jax.random.normal(theta_key, (len(MT_Sigma_inv_R),))
        theta = theta_opt + solve(post_cov_inv_U, p)
        phase_shifts = self.M @ theta
        phase_shifts -= jnp.mean(phase_shifts)

        return theta, phase_shifts, key


def bpl_powspec(pars, freqs):

    logA = pars[0]
    logfc = pars[1]
    gamma = pars[2]

    A = 10**logA
    fc = 10**logfc

    # In units of yr ** 3
    norm = A**2 / (12 * jnp.pi**2) * (fc ** (-gamma))

    psd = norm * (1 + (freqs / fc) ** 2) ** (-gamma / 2)

    return psd * YR3_TO_S2D


def bpl_flattail_powspec(pars, freqs):

    bpl = self.bpl_powspec(pars, freqs)
    logkappa = pars[3]

    flat = 10 ** (2 * logkappa) * YR3_TO_S2D
    psd = jnp.maximum(bpl, flat)

    return psd


class NoiseAndTimingModelSampler(TimingModelSampler):

    def __init__(
        self,
        phi,
        M,
        theta_prior,
        timing_prior_uncertainties,
        noise_models,
        parameter_scales,
        PB0=None,
    ):

        self.phi = phi
        self.npar = np.shape(M)[1]
        self.M = jnp.array(M)
        self.theta_prior = jnp.array(theta_prior)
        self.parameter_scales = parameter_scales

        self.K0 = jnp.array(timing_prior_uncertainties**2)
        self.noise_models = noise_models

        freqs = np.zeros((len(noise_models), len(parameter_scales)))
        free = np.zeros((len(noise_models), 4), dtype=bool)
        bounds = np.zeros((len(noise_models), 4, 2))
        hyp0 = np.zeros((len(noise_models), 4))
        min_freqs = np.zeros((len(noise_models)))

        scales = np.ones(len(noise_models))

        for c in range(len(self.noise_models)):

            freqs[c, self.noise_models[c].Cinds] = self.noise_models[c].freqs.to_value(
                "1/yr"
            )
            freqs[c, self.noise_models[c].Sinds] = self.noise_models[c].freqs.to_value(
                "1/yr"
            )

            min_freqs[c] = self.noise_models[c].freqs.to_value("1/yr").min()

            if "OPV" in self.noise_models[c].prefix:
                scales[c] = 1.0 / PB0.to_value("s") ** 2
                print(scales)

            num_pars = len(self.noise_models[c].free)
            hyp0[c, :num_pars] = self.noise_models[c].x0

            free[c, :num_pars] = self.noise_models[c].free
            bounds[c, :num_pars, :] = self.noise_models[c].bounds

        self.noise_freqs = jnp.array(freqs)

        nfree = np.sum(free)
        hyp_bounds = np.zeros((nfree, 2))
        linear_priors = np.zeros(nfree)
        p = 0
        for c in range(len(self.noise_models)):
            hyp_bounds[p : p + np.sum(free[c]), :] = bounds[c, free[c], :]

            if free[c][0] and self.noise_models[c].linear_amp_prior:
                linear_priors[p] = True

            p += np.sum(free[c])

        self.hyp_bounds = jnp.array(hyp_bounds)
        self.hyp_free = jnp.array(free)
        self.scales = jnp.array(scales)
        self.x0 = jnp.array(hyp0)
        self.min_freqs = jnp.array(min_freqs)
        self.linear_priors = jnp.array(linear_priors)

    def _phys_to_samples(self, hyp):

        cube_pars = (hyp - self.hyp_bounds[:, 0]) / (
            self.hyp_bounds[:, 1] - self.hyp_bounds[:, 0]
        )
        return jax.scipy.special.logit(cube_pars)

    def _samples_to_phys(self, x):

        cube_pars = jax.scipy.special.expit(x)
        logprior = jnp.sum(jnp.log(cube_pars * (1 - cube_pars)))

        hyp = self.hyp_bounds[:, 0] + cube_pars * (
            self.hyp_bounds[:, 1] - self.hyp_bounds[:, 0]
        )

        amp_prior = 0.0
        for c in range(len(x)):
            amp_prior += jax.lax.cond(
                self.linear_priors[c], lambda: hyp[c], lambda: 0.0
            )

        return hyp, logprior + amp_prior

    def _fill_noise_pars(self, hyp):
        p = 0
        all_hyp = []

        for c in range(len(self.noise_models)):
            pars = []
            for i in range(len(self.hyp_free[c])):
                use_hyp = self.hyp_free[c][i]
                param = jax.lax.cond(use_hyp, lambda: hyp[p], lambda: self.x0[c][i])
                pars.append(param)
                p += jax.lax.cond(use_hyp, lambda: 1, lambda: 0)

            all_hyp.append(pars)

        return all_hyp

    def _make_psd_cov(self, all_hyp):

        K = self.K0.copy()

        for c in range(len(self.noise_models)):

            powspec = (
                jnp.where(
                    self.noise_freqs[c] > 0,
                    (
                        bpl_powspec(all_hyp[c], self.noise_freqs[c])
                        * self.min_freqs[c]
                        * INVYR_TO_INVDAY
                    ),
                    0.0,
                )
                * self.scales[c]
            )

            K += powspec

        Kinv = jnp.where(K > 0, 1.0 / K, 0.0)
        Kinv /= self.parameter_scales**2

        logdet_prior = -jnp.sum(jnp.where(Kinv > 0, jnp.log(Kinv), 0.0))

        return jnp.diag(Kinv), logdet_prior

    def log_post(self, x, MT_Sigma_inv_M, MT_Sigma_inv_R):

        hyp, logprior = self._samples_to_phys(x)
        all_hyp = self._fill_noise_pars(hyp)

        inv_prior_cov, logdet_prior = self._make_psd_cov(all_hyp)

        theta_opt, post_cov_inv_U = self.solve_leastsq(
            MT_Sigma_inv_M, MT_Sigma_inv_R, inv_prior_cov
        )

        prior_chi2 = self.theta_prior @ (inv_prior_cov @ self.theta_prior)
        post_chi2 = jnp.sum((post_cov_inv_U @ theta_opt) ** 2)
        logdet_post = -2 * jnp.sum(jnp.log(jnp.diag(post_cov_inv_U)))
        logL = logprior - 0.5 * (logdet_prior + prior_chi2 - logdet_post - post_chi2)

        return logL

    def setup_sampler(self, mu_z, sigma_z):
        """
        Performs a warm-up run to tune parameters (step size, covariance matrix)
        of the blackjax NUTS sampler for efficient sampling later.

        This also generates the

        Parameters
        ----------
        phases     : jax.numpy.ndarray of dtype float32
                     Array of observed photon phases (in [0, 1]).
        """

        x0 = self._phys_to_samples(self.hyp_0)

        MT_Sigma_inv_M, MT_Sigma_inv_R = self.setup_leastsq_given_z_tau(mu_z, sigma_z)

        logprob_fn = lambda x: self.log_post(x, MT_Sigma_inv_M, MT_Sigma_inv_R)

        warmup = blackjax.window_adaptation(
            blackjax.nuts, logprob_fn, progress_bar=True
        )

        rng_key = jax.random.key(0)
        rng_key, warmup_key, sample_key = jax.random.split(rng_key, 3)
        (state, parameters), _ = warmup.run(warmup_key, x0, num_steps=1000)

        self.nuts_params = parameters
        self.sample_key = sample_key

        step_fn = blackjax.nuts.build_kernel()

        @jax.jit
        def kernel(rng_key, state, MT_Sigma_inv_M, MT_Sigma_inv_R):

            logprob_fn = lambda x: self.log_post(x, MT_Sigma_inv_M, MT_Sigma_inv_R)
            state, _ = step_fn(
                rng_key=rng_key,
                state=state,
                logdensity_fn=logprob_fn,
                **self.nuts_params,
            )

            return state, (self._samples_to_phys(state.position)[0], state.logdensity)

        self.logprob_fn = jax.jit(self.log_post)

        self.kernel = kernel
        # hyp, logprior = state.position

        return self._samples_to_phys(state.position)[0], sample_key

    def sample(self, hyp, mu_z, sigma_z, key, num_samples=1000):
        """
        Run the blackjax NUTS sampler to estimate parameters of the template

        Parameters
        ----------

        tau       :   np.ndarray of dtype float
                      Template pulse profile parameters

        phases    :   np.ndarray of dtype float
                      Photon phases

        num_samples : Number of samples to draw, default = 1000

        Returns
        ----------
        samples   :   jnp.ndarray of dtype float32
                      NUTS samples, in the sampling parameter space
                      These can be converted to physical parameters
                      using self._samples_to_phys
        """

        MT_Sigma_inv_M, MT_Sigma_inv_R = self.setup_leastsq_given_z_tau(mu_z, sigma_z)
        logprob_fn = lambda x: self.logprob_fn(x, MT_Sigma_inv_M, MT_Sigma_inv_R)

        x = self._phys_to_samples(hyp)
        state = blackjax.nuts.init(x, logprob_fn)

        one_step = lambda state, key: self.kernel(
            key, state, MT_Sigma_inv_M, MT_Sigma_inv_R
        )
        hyp_keys = jax.random.split(key, num_samples + 1)

        _, samples = jax.lax.scan(one_step, state, hyp_keys[:-1])

        return samples, hyp_keys[-1]

    def sample_lambda_theta_given_tau_zm(self, hyp, mu_z, sigma_z, key, num_steps=10):

        hyp_samples, key = self.sample(hyp, mu_z, sigma_z, key, num_steps)

        new_hyp = hyp_samples[0][-1]
        all_hyp = self._fill_noise_pars(new_hyp)
        inv_prior_cov, logdet_prior = self._make_psd_cov(all_hyp)

        theta, phase_shifts, key = self.sample_theta_given_lambda_z_m_tau(
            mu_z, sigma_z, inv_prior_cov, key
        )

        return new_hyp, theta, phase_shifts, key

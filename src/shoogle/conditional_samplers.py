import numpy as np
import jax
import jax.numpy as jnp
import blackjax
from scipy.special import logit, expit
from pint.templates.lctemplate import LCTemplate, prim_io
from tqdm.auto import tqdm

ONE_OVER_SQRT2PI = 1.0 / (jnp.sqrt(2 * jnp.pi))


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

        self.tau_0 = np.concatenate((self.A_0, self.mu_0, self.sigma_0))
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

        Bk = np.copy(A)

        # Transform to beta-distributed variables with (0,1)
        # which ensures a Dirichlet prior on amplitudes + unpulsed comp.
        U = 1 - Bk[0]
        for p in range(1, self.npeaks):
            Bk[p] = A[p] / U
            U -= A[p]

        # Uniform prior on log(sigma), transformed to (0,1)
        logsigma01 = (np.log(sigma) - self.minlogsigma) / (
            self.maxlogsigma - self.minlogsigma
        )

        # then logit transformed -> (-inf,inf)
        x = jnp.array(np.concatenate((logit(Bk), logit(mu), logit(logsigma01))))

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

            return state

        self.kernel = kernel
        tau, logprior = self._samples_to_phys(state.position)

        return np.array(tau)

    def sample(self, tau, phases, num_samples=1000):
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

        samples = np.zeros((num_samples, len(self.tau_0)))
        logL = np.zeros(num_samples)

        for step in tqdm(range(num_samples)):
            self.sample_key, key = jax.random.split(self.sample_key)
            state = self.kernel(key, state, phases)
            samples[step], _ = self._samples_to_phys(state.position)
            logL[step] = state.logdensity

        return samples, logL

    def sample_tau_given_theta(self, tau, phases, num_steps=10):
        """
        Draw a single set of template parameters, given the current photon phases.
        This is intended to be used to draw single samples from the conditional
        distribution for Gibbs sampling


        Parameters
        ----------

        tau       :   np.ndarray of dtype float
                      Template pulse profile parameters

        phases    :   np.ndarray of dtype float
                      Photon phases

        num_samples : Number of steps to run the chain for, default = 10

        Returns
        ----------
        tau   :       np.ndarray of dtype float32
                      A single set of template pulse profile parameters
        """

        phases = jnp.array(phases)
        logprob_fn = lambda x: self.logprob_fn(x, phases)

        x = self._phys_to_samples(tau)
        state = blackjax.nuts.init(x, logprob_fn)

        samples = np.zeros((num_steps, len(self.tau_0)))
        logL = np.zeros(num_steps)

        for step in range(num_steps):
            self.sample_key, key = jax.random.split(self.sample_key)
            state = self.kernel(key, state, phases)

        tau, logprior = self._samples_to_phys(state.position)

        return np.array(tau)


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
        U_i = self.w * (1 - np.sum(A, axis=1))

        # Prior probability that the photon came from the background
        B_i = 1 - self.w

        norm = np.sum(A_ik, axis=1) + U_i + B_i  # should be unity
        assert np.allclose(
            norm, 1.0
        ), f"Norms don't all add up to one. (Max = {np.max(norm)},min = {np.min(norm)})."

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
        wraps = np.arange(-self.maxwraps, self.maxwraps + 1, 1)

        log_like = (
            -np.log(sigma)[:, :, None]
            - 0.5 * np.log(2 * np.pi)
            - 0.5
            * (
                (phases[:, None, None] - mu[:, :, None] - wraps[None, None, :])
                / sigma[:, :, None]
            )
            ** 2
        )

        return log_like, mu, sigma

    def post_z_given_theta_tau(self, tau, phases):

        A_ik, U_i, B_i = self.prior_z_given_tau(tau)
        log_like, mu, sigma = self.likelihood_z_given_theta_tau(tau, phases)

        # Mixture weights
        log_rho_ik = np.log(A_ik)[:, :, None] + log_like

        # This sums over phase wraps for each component
        rho_ik = np.sum(np.exp(log_rho_ik), axis=-1)

        sum_rho = np.sum(rho_ik, axis=1) + U_i + B_i
        rho_ik = rho_ik / sum_rho[:, None]
        rho_iU = U_i / sum_rho
        rho_iB = B_i / sum_rho

        rho = np.concatenate((rho_ik, rho_iU[:, None], rho_iB[:, None]), axis=1)

        return A_ik, U_i, B_i, log_like, rho, mu, sigma

    def sample_z_given_theta_tau(self, tau, phases):

        A_ik, U_i, B_i, log_like, rho, mu, sigma = self.post_z_given_theta_tau(
            tau, phases
        )

        C = np.cumsum(rho, axis=1)
        u = np.random.rand(len(phases))
        z = np.sum(C < u[:, None], axis=1)
        m = np.zeros(len(phases), dtype=int)
        wraps = np.arange(-1, 2, 1)

        mu_z = np.zeros(len(phases))
        sigma_z = np.zeros(len(phases))
        for k in range(self.npeaks):
            mask = np.where(z == k)[0]

            log_rho_wraps = log_like[mask, k]

            rho_wraps = np.exp(log_rho_wraps)
            rho_wraps /= np.sum(rho_wraps, axis=1)[:, None]

            C_wraps = np.cumsum(rho_wraps, axis=1)

            u = np.random.rand(len(mask))
            m[mask] = np.sum(C_wraps < u[:, None], axis=1)

            # Check for energy dependence
            if np.shape(mu)[0] == 1:
                mu_z[mask] = mu[0, k] + wraps[m[mask]]
                sigma_z[mask] = sigma[0, k]
            else:
                mu_z[mask] = mu[mask, k] + wraps[m[mask]]
                sigma_z[mask] = sigma[mask, k]

        return mu_z, sigma_z

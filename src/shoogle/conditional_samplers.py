import numpy as np
import jax
from jax.scipy.linalg import cholesky, cho_solve, solve_triangular
import jax.numpy as jnp
from jaxopt import LBFGS
import blackjax
from optax.assignment import hungarian_algorithm
from scipy.special import logit, expit
from tqdm.auto import tqdm
from astropy import units as u
import pickle
from pint.templates.lctemplate import LCTemplate, prim_io
from pint.templates.lceprimitives import LCEGaussian
from pint.templates.lcenorm import ENormAngles
import json

ONE_OVER_SQRT2PI = 1.0 / (jnp.sqrt(2 * jnp.pi))
LOG10YR3_TO_S2D = jnp.log10((1.0 * u.yr**3).to_value("s ** 2 * d"))
INVYR_TO_INVDAY = (1.0 / u.yr).to_value("1/d")

from pint.templates.lcprimitives import LCPrimitive


def new_default_bounds(self):
    bounds = [[] for _ in range(len(self.p))]
    bounds[0] = [0.001, 0.5]  # width
    bounds[-1] = [-1, 1]  # position
    if len(bounds) > 2:
        bounds[1] = [0.001, 0.5]

    return bounds


LCPrimitive._default_bounds = new_default_bounds


def read_template(proffile, extra_phase=None):

    if ".pickle" in proffile:
        with open(proffile, "rb") as input_file:
            template = pickle.load(input_file)
            input_file.close()

        amps = np.array([A for A in template.norms()])
        mus = np.mod(np.array([p.get_location() for p in template.primitives]), 1.0)
        sigmas = np.array([p.get_width() for p in template.primitives])

    elif ".json" in proffile:

        with open(proffile, "r") as input_file:
            d = json.loads(input_file.read())
            input_file.close()
        pdicts = d["primitives"]
        primitives = []
        for pd in pdicts:
            const = eval(pd["name"])
            kwargs = {}
            for key in ["p", "free", "slope", "slope_free"]:
                if key in pd:
                    kwargs[key] = pd[key]
            primitives.append(const(**kwargs))
        ndict = d["norms"]
        const = eval(ndict["name"])
        kwargs = {}
        for key in ["free", "slope", "slope_free"]:
            if key in ndict:
                kwargs[key] = ndict[key]
        norms = const(d["norms"]["norms"], **kwargs)
        template = LCTemplate(primitives, norms=norms, cache_kwargs=d["cache_kwargs"])

        amps = np.array([A for A in template.norms()])
        mus = np.mod(np.array([p.get_location() for p in template.primitives]), 1.0)
        sigmas = np.array([p.get_width() for p in template.primitives])

    else:
        amps = []
        mus = []
        sigmas = []

        with open(proffile, "r") as input_file:
            lines = input_file.readlines()

        for line in lines:
            if line.split()[0][:4] == "ampl":
                amps.append(float(line.split()[2]))
            elif line.split()[0][:4] == "phas":
                mus.append(float(line.split()[2]))
            elif line.split()[0][:4] == "fwhm":
                sigmas.append(float(line.split()[2]) / (2 * np.sqrt(2 * np.log(2))))

        amps = np.array(amps)
        mus = np.array(mus)
        sigmas = np.array(sigmas)

    if extra_phase:
        mus = np.mod(mus + extra_phase, 1.0)

    return amps, mus, sigmas


class TemplateSampler(object):
    """A class providing the MCMC fitter used to draw samples of the template
    pulse profile parameters, given the current timing model, within the Gibbs
    sampling procedure.

    It is implemented in JAX, for speed and to enable the use of the blackjax
    NUTS sampler, which provides efficient MCMC sampling.
    """

    def __init__(
        self,
        proffile,
        weights,
        minsigma=0.0,
        maxsigma=0.5,
        maxwraps=2,
        extra_phase=None,
    ):
        """
        Initialises the object, including reading a starting estimate for
        the profile parameters from a file.

        Parameters
        ----------
        proffile   : str
                     Text file containing the initial pulse profile shape.
                     This can be obtained using itemplate.py
        weights    : np.ndarray, dtype = float
                     Photon probability weights
        minsigma   : float, default = 0.005
                     Minimum width of a Gaussian peak in the pulse profile,
                     in fractions of a rotation. (default = 0.5% of a rotation)
        maxsigma   : float, default = 0.25
                     Minimum width of a Gaussian peak in the pulse profile,
                     in fractions of a rotation. (default = 25% of a rotation)
        maxwraps   : int, default = 2
                     Maximum number of phase wraps to sum over.
                     More wraps = more accurate, but slower.
        """
        self.A_0, self.mu_0, self.sigma_0 = read_template(
            proffile, extra_phase=extra_phase
        )
        self.A_0 *= 0.999  # This prevents nasty numerical issues...

        self.tau_0 = jnp.array(np.concatenate((self.A_0, self.mu_0, self.sigma_0)))
        self.npeaks = len(self.A_0)

        self.maxwraps = maxwraps
        self.wraps = jnp.arange(-2, 3)

        self.minsigma_sq = minsigma**2
        self.maxsigma_sq = maxsigma**2
        self.w = jnp.array(weights)

    def template(self, tau, phases):
        """
        Evaluates the pulse profile template, checking the amplitude constraints

        Parameters
        ----------

        tau       : array
                    Template pulse profile parameters, in order:
                    (amplitudes, positions, widths)

                    Amplitudes should sum to < 1.0
                    Positions should be between (0,1)
                    Widths should be between the min and max values.

        phases    : array
                    Array of photon phases
                    Should be between (0,1), though this is not enforced.
                    Values outside the range spanned by maxwraps will be incorrect

        Returns
        -------

        profile   : array
                    The template pulse profile evaluated at the specified phases

        """

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

        x    :   jax.numpy.ndarray
                 Location of the current point in the sample parameter space

        Returns
        ---------

        tau          : jax.numpy.ndarray
                       Pulse profile parameters to pass to self._jax_template

        logprior     : float
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

        # Uniform between 0, 1
        ux = jax.scipy.special.expit(x[self.npeaks : 2 * self.npeaks])
        uy = jax.scipy.special.expit(x[2 * self.npeaks : 3 * self.npeaks])
        logprior += jnp.sum(jnp.log(ux * (1 - ux)))
        logprior += jnp.sum(jnp.log(uy * (1 - uy)))

        # 2D normal distribution
        nx = jax.scipy.stats.norm.isf(ux)
        ny = jax.scipy.stats.norm.isf(uy)

        # Force it to be near the unit circle
        # (avoids coordinate singularity at nx, ny = 0)
        donut_sigma = 0.05
        logprior += jnp.sum(-0.5 * (jnp.sqrt(nx**2 + ny**2) - 1) ** 2 / donut_sigma**2)

        mu = jnp.arctan2(nx, ny) / (2 * jnp.pi)

        sigmasq01 = jax.scipy.special.expit(x[3 * self.npeaks : 4 * self.npeaks])
        logprior += jnp.sum(jnp.log(sigmasq01 * (1 - sigmasq01)))
        sigma = jnp.sqrt(
            sigmasq01 * (self.maxsigma_sq - self.minsigma_sq) + self.minsigma_sq
        )

        W2 = jnp.sum(self.w**2)
        min_SN = 3.0

        # Penalise peaks narrower than expected fluctuations
        invgamma_beta = (min_SN**2 / W2) ** 2
        invgamma_alpha = 0.5
        logprior += jnp.sum(
            invgamma_alpha * jnp.log(invgamma_beta)
            - jax.scipy.special.gammaln(invgamma_alpha)
            - (invgamma_alpha + 1) * jnp.log(sigma**2)
            - invgamma_beta / sigma**2
        )

        return jnp.concatenate((A, mu, sigma)), logprior

    def _log_like(self, tau, phases):
        """
        Computes the log-likelihood of the observed photon phases given the
        pulse profile template parameters and photon weights.

        Parameters
        ----------
        tau        : jax.numpy.ndarray
                     Array of pulse profile parameters
        phases     : jax.numpy.ndarray
                     Array of observed photon phases (in [0, 1]).

        Returns
        -------
        logL       : float
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
        x          : jax.numpy.ndarray
                     Array of unconstrained (logit-transformed) parameters
                     representing the current point in the parameter space.
                     Shape: (3 * npeaks,)
        phases     : jax.numpy.ndarray
                     Array of observed photon phases (in [0, 1]).
        weights    : jax.numpy.ndarray
                     Array of photon probability weights (in [0, 1]).

        Returns
        -------
        log_post   : float
                     The log-posterior probability: log_prior + log_likelihood.
        """
        tau, logprior = self._samples_to_phys(x)
        loglike = self._log_like(tau, phases)

        return logprior + loglike

    def _phys_to_samples(self, tau):
        """
        Transforms from physically meaningful parameters (As, mus, sigmas)
        to the unconstrained sampling space (logit-transformed).

        Parameters
        ----------
        tau        : jax.numpy.ndarray
                     Array of template pulse profile parameters

        Returns
        -------
        x          : Internal co-ordinates used for sampling with blackjax
        """

        A = tau[: self.npeaks]
        mu = tau[self.npeaks : self.npeaks * 2]
        sigma = tau[self.npeaks * 2 :]

        Bk = A.copy()

        # Transform to beta-distributed variables within (0,1)
        # which ensures a Dirichlet prior on amplitudes + unpulsed comp.
        U = 1 - Bk[0]
        for p in range(1, self.npeaks):
            Bk = Bk.at[p].set(A[p] / U)
            U -= A[p]

        # Uniform prior on log(sigma), transformed to (0,1)
        sigmasq01 = (sigma**2 - self.minsigma_sq) / (
            self.maxsigma_sq - self.minsigma_sq
        )

        nx = jnp.sin(2 * jnp.pi * mu)
        ny = jnp.cos(2 * jnp.pi * mu)

        ux = jax.scipy.stats.norm.sf(nx)
        uy = jax.scipy.stats.norm.sf(ny)

        # then logit transformed -> (-inf,inf)
        x = jax.scipy.special.logit(jnp.concatenate((Bk, ux, uy, sigmasq01)))

        return x

    def setup_sampler(self, phases, rng_key):
        """
        Performs a warm-up run to tune parameters (step size, covariance matrix)
        of the blackjax NUTS sampler for efficient sampling later.

        This also generates the

        Parameters
        ----------
        phases     : jax.numpy.ndarray
                     Array of observed photon phases (in [0, 1]).
        """

        logprob_fn = lambda x: self._log_post(x, phases)
        neg_logprob_fn = lambda x: -self._log_post(x, phases)

        x0 = self._phys_to_samples(self.tau_0)
        logpost0 = -neg_logprob_fn(x0)
        loglike0 = self._log_like(self.tau_0, phases)

        optimiser = LBFGS(neg_logprob_fn)
        res = optimiser.run(x0)

        x0 = res.params
        self.tau_0 = self._samples_to_phys(x0)[0]
        print("Optimising template:", logpost0, "->", -neg_logprob_fn(x0))
        print("log-likelihoods:", loglike0, "->", self._log_like(self.tau_0, phases))

        warmup_key, sample_key = jax.random.split(rng_key, 2)

        step_fn = blackjax.nuts.build_kernel(
            integrator=blackjax.mcmc.integrators.yoshida
        )

        @jax.jit
        def inv_hess_diag(x, phases):

            grad = lambda x: jax.grad(self._log_post, argnums=0)(x, phases)

            def d2f_dxi2(i):
                v = jnp.zeros_like(x).at[i].set(1.0)
                _, hi = jax.jvp(grad, (x,), (v,))
                return hi[i]

            Hdiag = jax.vmap(d2f_dxi2, in_axes=0, out_axes=0)(jnp.arange(len(x)))

            return jnp.clip(1.0 / jnp.abs(Hdiag), 0, 1e-2)

        def kernel_generator(step_size):

            def kernel(rng_key, state):

                IMM = inv_hess_diag(state.position, phases)

                logprob_fn = lambda x: self._log_post(x, phases)

                return step_fn(
                    rng_key=rng_key,
                    state=state,
                    logdensity_fn=logprob_fn,
                    step_size=step_size,
                    max_num_doublings=10,
                    inverse_mass_matrix=IMM,
                )

            return kernel

        state = blackjax.mcmc.hmc.HMCState(
            position=x0,
            logdensity=self._log_post(x0, phases),
            logdensity_grad=jax.grad(self._log_post, argnums=0)(x0, phases),
        )

        step_size_opt = blackjax.adaptation.step_size.find_reasonable_step_size(
            warmup_key,
            kernel_generator,
            state,
            initial_step_size=0.1,
            target_accept=0.95,
        )

        @jax.jit
        def kernel(rng_key, state, phases):

            IMM = inv_hess_diag(state.position, phases)

            logprob_fn = lambda x: self._log_post(x, phases)
            state, _ = step_fn(
                rng_key=rng_key,
                state=state,
                logdensity_fn=logprob_fn,
                step_size=0.5,
                inverse_mass_matrix=IMM,
                max_num_doublings=15,
            )

            return state, (self._samples_to_phys(state.position)[0], state.logdensity)

        self.kernel = kernel

        return self.tau_0, sample_key

    def sample(self, tau, phases, key, num_samples=1000):
        """
        Run the blackjax NUTS sampler to estimate parameters of the template

        Parameters
        ----------

        tau       :   jax.numpy.ndarray
                      Template pulse profile parameters

        phases    :   jax.numpy.ndarray
                      Photon phases

        key       :   jax PRNGKeyArray
                      Random number key

        num_samples : Number of samples to draw, default = 1000

        Returns
        ----------
        samples   :   jnp.ndarray of dtype float32
                      NUTS samples, in the sampling parameter space
                      These can be converted to physical parameters
                      using self._samples_to_phys

        key       :   A new PRNG key
        """

        x0 = self._phys_to_samples(tau)

        # Make sure starting point isn't too close to a boundary
        x0 = jnp.clip(x0, -10, 10)

        logprob_fn = lambda x: self._log_post(x, phases)
        state = blackjax.nuts.init(x0, logprob_fn)

        one_step = lambda state, key: self.kernel(key, state, phases)
        tau_keys = jax.random.split(key, num_samples + 1)

        _, samples = jax.lax.scan(one_step, state, tau_keys[:-1])

        return samples, tau_keys[-1]

    def unswap_peaks(self, tau):

        K = self.npeaks

        # Indices of all pairs of peaks (including matching pairs)
        peak1, peak2 = jnp.triu_indices(K, -K)

        A = tau[:K]
        mu = tau[K : 2 * K]
        sigma = tau[2 * K : 3 * K]

        A0 = self.tau_0[:K]
        mu0 = self.tau_0[K : 2 * K]
        sigma0 = self.tau_0[2 * K : 3 * K]

        logL_matrix = self.peak_similarity(
            A0[:, None],
            mu0[:, None],
            sigma0[:, None],
            A[None, :],
            mu[None, :],
            sigma[None, :],
        )

        i, j = hungarian_algorithm(-logL_matrix)

        A_swapped = A.at[i].set(A[j])
        mu_swapped = mu.at[i].set(mu[j])
        sigma_swapped = sigma.at[i].set(sigma[j])

        return jnp.concatenate((A_swapped, mu_swapped, sigma_swapped))

    def peak_similarity(self, A1, mu1, sigma1, A2, mu2, sigma2):

        diff = jnp.abs(mu1 - mu2)
        diff = jnp.minimum(diff, 1 - diff)

        logL = A1 * (
            jnp.log(A2)
            - jnp.log(sigma2)
            - 0.5 * (diff**2 / sigma2**2 + sigma1**2 / sigma2**2 + jnp.log(2 * jnp.pi))
        ) + A2 * (
            jnp.log(A1)
            - jnp.log(sigma1)
            - 0.5 * (diff**2 / sigma1**2 + sigma2**2 / sigma1**2 + jnp.log(2 * jnp.pi))
        )

        return logL

    def sample_tau_given_theta(self, tau, phases, key, num_steps=10):
        """
        Obtain one sample from the conditional distribution of tau

        Parameters
        ----------

        tau        :  jax.numpy.ndarray
                      Template pulse profile parameters,
                      used as the starting point for the sampler

        phases     :  jax.numpy.ndarray
                      Photon phases

        key        :  jax PRNGKeyArray
                      Random number key

        num_steps  : Number of steps to run before drawing a sample
                      to reduce dependence on the starting point,
                      default = 10

        Returns
        -------

        sample     :  jax.numpy.ndarray
                      A new sample of the template pulse profile parameters

        key        :  jax PRNGKeyArray
                      New random number key
        """

        samples, key = self.sample(tau, phases, key, num_steps)
        return self.unswap_peaks(samples[0][-1]), key


class EdepTemplateSampler(TemplateSampler):
    """A class extending the TemplateSampler to energy-dependent pulse profiles.
    These are parameterised as low- and high-energy pulse profiles, with
    profiles at intermediate energies obtained by linearly extrapolating
    over log-energy between these two profiles.
    """

    def __init__(self, proffile, weights, log10E, **kwargs):
        """
        Initialises the object, including reading a starting estimate for
        the profile parameters from a file.

        Parameters
        ----------
        proffile   : str
                     Text file containing the initial pulse profile shape.
                     This can be obtained using itemplate.py
        weights    : np.ndarray, dtype = float
                     Photon probability weights
        log10E     : np.ndarray, dtype = float
                     Log-energies (in MeV units)
        minsigma   : float, default = 0.005
                     Minimum width of a Gaussian peak in the pulse profile,
                     in fractions of a rotation. (default = 0.5% of a rotation)
        maxsigma   : float, default = 0.25
                     Minimum width of a Gaussian peak in the pulse profile,
                     in fractions of a rotation. (default = 25% of a rotation)
        maxwraps   : int, default = 2
                     Maximum number of phase wraps to sum over.
                     More wraps = more accurate, but slower.
        """

        super().__init__(proffile, weights, **kwargs)

        self.tau_0 = jnp.concatenate((self.tau_0, self.tau_0))
        self.log10E = jnp.array(log10E)
        self.log10E_frac = (self.log10E - self.log10E.min()) / (
            self.log10E.max() - self.log10E.min()
        )

    def template(self, tau, phases, log10E):
        """
        Evaluates the pulse profile template, checking the amplitude constraints

        Parameters
        ----------

        tau       : array
                    Template pulse profile parameters, in order:
                    (amplitudes, positions, widths)

                    Amplitudes should sum to < 1.0
                    Positions should be between (0,1)
                    Widths should be between the min and max values.

        phases    : array
                    Array of photon phases
                    Should be between (0,1), though this is not enforced.
                    Values outside the range spanned by maxwraps will be incorrect

        log10E    : array
                    Array of photon log-energies

        Returns
        -------

        profile   : array
                    The template pulse profile evaluated at the specified phases

        """

        assert (
            np.sum(tau[: self.npeaks]) <= 1.0
        ), "Error: sum of low-energy template amplitudes > 1"

        assert (
            np.sum(tau[3 * self.npeaks : 4 * self.npeaks]) <= 1.0
        ), "Error: sum of high-energy template amplitudes > 1"

        tau = jnp.array(tau)
        phases = jnp.array(phases)
        log10E = jnp.array(log10E)

        return self._jax_template(tau, phases, log10E)

    def _jax_template(self, tau, phases, log10E=None):
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

        log10E     : jax.numpy.ndarray or None
                     If None, uses the original log-energies
                     Otherwise, evaluates at the specified log-energies

        Returns
        ---------

        profile    : jax.numpy.ndarray
                     The pulse profile, evaluated at the input phases.
        """

        if log10E is None:
            log10E_frac = self.log10E_frac
        else:
            log10E_frac = (log10E - self.log10E.min()) / (
                self.log10E.max() - self.log10E.min()
            )

        tau_lo = tau[: 3 * self.npeaks]
        tau_hi = tau[3 * self.npeaks :]

        A_lo = tau_lo[: self.npeaks]
        A_hi = tau_hi[: self.npeaks]

        # Ensure peak centres always shift the shortest way around the circle
        mu_lo = tau_lo[self.npeaks : 2 * self.npeaks]
        mu_hi = tau_hi[self.npeaks : 2 * self.npeaks]
        mu_lo_wrapped = jnp.mod(mu_lo - mu_hi + 0.5, 1.0) - 0.5 + mu_hi

        sigma_lo = tau_lo[2 * self.npeaks :]
        sigma_hi = tau_hi[2 * self.npeaks :]

        A_E = A_lo[None, :] + log10E_frac[:, None] * (A_hi - A_lo)[None, :]
        mu_E = (
            mu_lo_wrapped[None, :]
            + log10E_frac[:, None] * (mu_hi - mu_lo_wrapped)[None, :]
        )
        sigma_E = (
            sigma_lo[None, :] + log10E_frac[:, None] * (sigma_hi - sigma_lo)[None, :]
        )

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
        """
        Converts from parameters convenient for the sampler,
        which are logit-transformed to ranges of (-inf,inf),
        to physical parameters describing the pulse profile peaks.

        Parameters
        ----------

        x    :   jax.numpy.ndarray
                 Location of the current point in the sample parameter space

        Returns
        ---------

        tau          : jax.numpy.ndarray
                       Pulse profile parameters to pass to self._jax_template

        logprior     : float
                       the log-prior for these parameters, accounting for both
                       the priors (Dirichlet prior on amplitudes),
                       and the Jacobian of the logit transforms.
        """

        tau_lo, logprior_lo = super()._samples_to_phys(x[: 4 * self.npeaks])
        tau_hi, logprior_hi = super()._samples_to_phys(x[4 * self.npeaks :])

        tau = jnp.concatenate((tau_lo, tau_hi))

        return tau, logprior_lo + logprior_hi

    def _phys_to_samples(self, tau):
        """
        Transforms from physically meaningful parameters (As, mus, sigmas)
        to the unconstrained sampling space (logit-transformed).

        Parameters
        ----------
        tau        : jax.numpy.ndarray
                     Array of template pulse profile parameters

        Returns
        -------
        x          : Internal co-ordinates used for sampling with blackjax
        """

        x_lo = super()._phys_to_samples(tau[: 3 * self.npeaks])
        x_hi = super()._phys_to_samples(tau[3 * self.npeaks :])

        x = jnp.concatenate((x_lo, x_hi))

        return x

    def unswap_peaks(self, tau):

        K = self.npeaks

        # Indices of all pairs of peaks (including matching pairs)
        A_lo = tau[:K]
        mu_lo = tau[K : 2 * K]
        sigma_lo = tau[2 * K : 3 * K]

        A0_lo = self.tau_0[:K]
        mu0_lo = self.tau_0[K : 2 * K]
        sigma0_lo = self.tau_0[2 * K : 3 * K]

        A_hi = tau[3 * K : 4 * K]
        mu_hi = tau[4 * K : 5 * K]
        sigma_hi = tau[5 * K : 6 * K]

        A0_hi = self.tau_0[3 * K : 4 * K]
        mu0_hi = self.tau_0[4 * K : 5 * K]
        sigma0_hi = self.tau_0[5 * K : 6 * K]

        logL_matrix = self.peak_similarity(
            A0_lo[:, None],
            mu0_lo[:, None],
            sigma0_lo[:, None],
            A_lo[None, :],
            mu_lo[None, :],
            sigma_lo[None, :],
        ) + self.peak_similarity(
            A0_hi[:, None],
            mu0_hi[:, None],
            sigma0_hi[:, None],
            A_hi[None, :],
            mu_hi[None, :],
            sigma_hi[None, :],
        )

        i, j = hungarian_algorithm(-logL_matrix)

        A_lo_swapped = A_lo.at[i].set(A_lo[j])
        mu_lo_swapped = mu_lo.at[i].set(mu_lo[j])
        sigma_lo_swapped = sigma_lo.at[i].set(sigma_lo[j])

        A_hi_swapped = A_hi.at[i].set(A_hi[j])
        mu_hi_swapped = mu_hi.at[i].set(mu_hi[j])
        sigma_hi_swapped = sigma_hi.at[i].set(sigma_hi[j])

        return jnp.concatenate(
            (
                A_lo_swapped,
                mu_lo_swapped,
                sigma_lo_swapped,
                A_hi_swapped,
                mu_hi_swapped,
                sigma_hi_swapped,
            )
        )


class LatentVariableSampler(object):
    """A class for obtaining random draws of the latent variables,
    z & m, which assign photons to profile peaks and phase wraps.
    """

    def __init__(self, weights, npeaks, log10E=None, maxwraps=2):
        """
        Parameters
        ----------

        weights    : np.ndarray
                     Photon probability weights

        npeaks     : int
                     Number of peaks in the pulse profile

        log10E     : np.ndarray or None
                     Photon log-energies, if the profile is energy-dependent

        maxwraps   : int, default=2
                     Max number of phase wraps
        """

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
        """
        Obtain pulse profile parameters for each photon, interpolating over energies

        Parameters
        ----------

        tau       : np.ndarray, size (6 * npeaks,)
                    Pulse profile parameters

        Returns
        -------

        tau_E     : np.ndarray, size (nphotons, 3 * npeaks)
                    Pulse profile parameters specific to each photon
        """

        tau_lo = tau[: 3 * self.npeaks]
        tau_hi = tau[3 * self.npeaks :]

        A_lo = tau_lo[: self.npeaks]
        A_hi = tau_hi[: self.npeaks]

        # Ensure peak centres always shift the shortest way around the circle
        mu_lo = tau_lo[self.npeaks : 2 * self.npeaks]
        mu_hi = tau_hi[self.npeaks : 2 * self.npeaks]
        mu_lo_wrapped = jnp.mod(mu_lo - mu_hi + 0.5, 1.0) - 0.5 + mu_hi

        sigma_lo = tau_lo[2 * self.npeaks :]
        sigma_hi = tau_hi[2 * self.npeaks :]

        A_E = A_lo[None, :] + self.log10E_frac[:, None] * (A_hi - A_lo)[None, :]
        mu_E = (
            mu_lo_wrapped[None, :]
            + self.log10E_frac[:, None] * (mu_hi - mu_lo_wrapped)[None, :]
        )
        sigma_E = (
            sigma_lo[None, :]
            + self.log10E_frac[:, None] * (sigma_hi - sigma_lo)[None, :]
        )

        tau_E = jnp.concatenate((A_E, mu_E, sigma_E), axis=1)

        return tau_E

    def prior_z_given_tau(self, tau):
        """
        Calculate the prior probabilities for photon--peak assignments,
        given the peak amplitudes and photon weights

        Parameters
        ----------

        tau       : array, size = (3 * npeaks,)
                                    or (6 * npeaks,) if energy-dependent
                    Pulse profile parameters

        Returns
        -------

        A_ik      : array, size = (nphotons, npeaks)
                    Prior probabilities for each photon/peak pair

        U_i       : array, size = (nphotons,)
                    Prior probability for unpulsed component of pulsar flux

        B_i       : array, size = (nphotons,)
                    Prior probability of photon coming from the background
        """

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
        """
        Calculates the likelihoods for photon--peak assignments

        Parameters
        ----------

        tau       : Array, size = (3 * npeaks,)
                                    or (6 * npeaks,) if energy-dependent
                    Pulse profile parameters

        phases    : Array, size = (nphotons,)
                    Photon phases

        Returns
        -------

        log_like  : Array, size = (nphotons, npeaks)
                    Log-likelihood for photon--peak assignments

        mu        : Array, size = (1, npeaks) or (nphotons, npeaks)
                    Peak positions (photon-specific if energy dependence)

        sigma     : Array, size = (1, npeaks) or (nphotons, npeaks)
                    Peak widths (photon-specific if energy dependence)
        """

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
        """
        Calculates the posterior probabilities for photon--peak assignments

        Parameters
        ----------

        tau       : Array, size = (3 * npeaks,)
                                    or (6 * npeaks,) if energy-dependent
                    Pulse profile parameters

        phases    : Array, size = (nphotons,)
                    Photon phases

        Returns
        -------

        A_ik      : array, size = (nphotons, npeaks)
                    Prior probabilities for each photon/peak pair

        U_i       : array, size = (nphotons,)
                    Prior probability for unpulsed component of pulsar flux

        B_i       : array, size = (nphotons,)
                    Prior probability of photon coming from the background

        log_like  : Array, size = (nphotons, npeaks)
                    Log-likelihood for photon--peak assignments

        rho       : Array, size = (nphotons, npeaks + 2)
                    Posterior probabilities for peaks/unpulsed/background

        mu        : Array, size = (1, npeaks) or (nphotons, npeaks)
                    Peak positions (photon-specific if energy dependence)

        sigma     : Array, size = (1, npeaks) or (nphotons, npeaks)
                    Peak widths (photon-specific if energy dependence)

        """

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
        """
        Draws random z, m vectors, and returns the resulting per-photon
        peak position and width


        Parameters
        ----------

        tau       : Array, size = (3 * npeaks,)
                                    or (6 * npeaks,) if energy-dependent
                    Pulse profile parameters

        phases    : Array, size = (nphotons,)
                    Photon phases

        key       : JAX PRNG key

        Returns
        -------

        mu_z        : Array, size = (nphotons,)
                    Photon-specific peak positions (0 if unpulsed/background)

        sigma_z     : Array, size = (nphotons,)
                    Photon-specific peak widths (0 if unpulsed/background)

        next_key    : JAX PRNG key
        """

        A_ik, U_i, B_i, log_like, rho, mu, sigma = self.post_z_given_theta_tau(
            tau, phases
        )

        C = jnp.cumsum(rho, axis=1)

        z_key, m_key, next_key = jax.random.split(key, 3)

        # Photon->peak/wrap assignments are done by inverse CDF sampling.
        # We compute the CDF, generate a random number, and check how many bins are below that number

        # We need to clip the limits to be not exactly 0 or 1.
        # If r = 1, and the wrap likelihood looks like [0,0,1,0,0],
        # then the CDF looks like [0,0,1,1,1], so sum(C <= r) = 5
        # and the photon would be assigned to index 5, which doesn't exist!
        eps = jnp.finfo(jnp.float32).eps
        zr = jax.random.uniform(z_key, (len(phases),), minval=eps, maxval=1 - eps)
        mr = jax.random.uniform(m_key, (len(phases),), minval=eps, maxval=1 - eps)

        z = jnp.sum(C <= zr[:, None], axis=1)

        mu_z = jnp.zeros(len(phases))
        sigma_z = jnp.zeros(len(phases))
        for k in range(self.npeaks):
            mask = z == k

            log_rho_wraps = log_like[:, k]

            rho_wraps = jnp.exp(log_rho_wraps - jnp.max(log_rho_wraps, axis=1)[:, None])
            rho_wraps /= jnp.sum(rho_wraps, axis=1)[:, None]

            C_wraps = jnp.cumsum(rho_wraps, axis=1)

            m = jnp.sum(C_wraps <= mr[:, None], axis=1)

            if len(tau) // 3 == 2 * self.npeaks:
                mu_z = jnp.where(mask, mu[:, k] + self.wraps[m], mu_z)
                sigma_z = jnp.where(mask, sigma[:, k], sigma_z)
            else:
                mu_z = jnp.where(mask, mu[0, k] + self.wraps[m], mu_z)
                sigma_z = jnp.where(mask, sigma[0, k], sigma_z)

        return mu_z, sigma_z, next_key


class TimingModelSampler(object):
    """A class for obtaining random samples of the timing model, given the
    latent variables and template pulse profile"""

    def __init__(self, phi, M, theta_prior):
        """
        Parameters
        ----------

        phi        : Array, size (nphotons,)
                     Photon phases

        M          : Array, size (nphotons, npars)
                     Design matrix

        theta_prior : Array, size (npars,)
                      Prior means for timing model parameters
        """

        self.phi = phi
        self.npar = np.shape(M)[1]
        self.M = jnp.array(M)
        self.theta_prior = jnp.array(theta_prior)

    def setup_leastsq_given_z_tau(self, mu_z, sigma_z):
        """
        Computes M^T Sigma^{-1} M and M^T Sigma^{-1} R required for the
        weighted least squares fit to the timing model

        Parameters
        ----------

        mu_z        : Array, size (nphotons,)
                      Peak positions for each photon (0 if unpulsed/background)

        sigma_z        : Array, size (nphotons,)
                      Peak widths for each photon (0 if unpulsed/background)

        Returns
        -------

        MT_Sigma_inv_M : Array, size (npars, npars)
                         Inverse covariance matrix for timing parameter likelihood

        MT_Sigma_inv_R : Array, size (npars,)
                         RHS for WLS fit
        """

        resids = self.phi - mu_z
        precisions = jnp.where(sigma_z > 0, 1.0 / sigma_z**2, 0.0)

        MT_Sigma_inv_M = self.M.T @ (precisions[:, None] * self.M)
        MT_Sigma_inv_R = self.M.T @ (precisions * resids)

        # Enforce symmetry
        MT_Sigma_inv_M = 0.5 * (MT_Sigma_inv_M + MT_Sigma_inv_M.T)

        return MT_Sigma_inv_M, MT_Sigma_inv_R

    def solve_leastsq(self, MT_Sigma_inv_M, MT_Sigma_inv_R, inv_prior_cov):
        """
        Solves the weighted least squares fit to the timing model to find the
        conditional distribution on the timing model given the latent variables
        /pulse profile.

        Parameters
        ----------

        MT_Sigma_inv_M : Array, size (npars, npars)
                         Inverse covariance matrix for timing parameter likelihood

        MT_Sigma_inv_R : Array, size (npars,)
                         RHS for WLS fit

        inv_prior_cov  : Array, size (npars, npars)
                         Inverse covariance matrix for timing parameter priors

        Returns
        -------

        theta_opt      : Array, size (npars,)
                         Posterior mean for timing model parameters

        V              : Array, size (npars,npars)
                         Upper-triangular Cholesky decomposition, V where G = V^T V
                         where D G D = Cinv, the inverse of the posterior covariance matrix C
                         U = VD is the Cholesky decomposition of Cinv

        Dinv           : Array, size (npars)
                         The inverse of the square-root of the diagonal of the posterior covariance matrix
                         Used to scale/rescale the matrix for numerical stability.

        logdet_prior   : float
                         Log of the determinant of the prior covariance matrix

        prior_chi2     : float
                         chi-squared term for the prior timing model mean
        """

        Kinv = jnp.diag(inv_prior_cov)
        logdet_prior = -jnp.sum(jnp.where(Kinv > 0, jnp.log(Kinv), 0.0))
        prior_chi2 = jnp.sum(Kinv * self.theta_prior**2)

        RHS = MT_Sigma_inv_R + Kinv * self.theta_prior

        post_cov_inv = MT_Sigma_inv_M + inv_prior_cov

        # Divide out the diagonal to make Cholesky decomposition more stable
        Dinv = 1.0 / jnp.sqrt(jnp.diag(post_cov_inv))
        G = Dinv[:, None] * post_cov_inv * Dinv[None, :] + jnp.eye(len(Dinv)) * 1e-6

        V = cholesky(G, lower=False)

        theta_opt = Dinv * cho_solve((V, False), Dinv * RHS)

        return theta_opt, V, Dinv, logdet_prior, prior_chi2

    def sample_theta_given_lambda_tau_zm(self, mu_z, sigma_z, inv_prior_cov, key):
        """
        Draws a random sample of the timing model parameters given z,m,tau,lambda

        Parameters
        ----------

        mu_z           : Array, size (nphotons,)
                         Peak positions for each photon (0 if unpulsed/background)

        sigma_z        : Array, size (nphotons,)
                         Peak widths for each photon (0 if unpulsed/background)

        inv_prior_cov  : Array, size (npars, npars)
                         Inverse covariance matrix for timing parameter priors

        key            : JAX PRNG key

        Returns
        -------

        theta          : Array, size (npars,)
                         Random draw of timing model parameters

        phase_shifts   : Array, size (nphotons,)
                         Phase shifts according to theta, relative to theta=0

        key            : JAX PRNG key
        """
        MT_Sigma_inv_M, MT_Sigma_inv_R = self.setup_leastsq_given_z_tau(mu_z, sigma_z)
        theta_opt, V, Dinv, _, _ = self.solve_leastsq(
            MT_Sigma_inv_M, MT_Sigma_inv_R, inv_prior_cov
        )

        theta_key, key = jax.random.split(key)
        p = jax.random.normal(theta_key, (len(MT_Sigma_inv_R),))
        theta = theta_opt + Dinv * solve_triangular(V, p, lower=False)
        phase_shifts = self.M @ theta
        phase_shifts -= jnp.mean(phase_shifts)

        return theta, phase_shifts, key


def bpl_log10powspec(pars, freqs):
    """
    Evaluates a smoothly broken power-law PSD

    PSD = (A^2 yr^3 / 12 pi^2) * (fc / fyr)^(-gamma) * (1 + (f / fc)^2)^(-gamma/2)

    Parameters
    ----------

    pars      : Array, size = (3,) (or longer)
                PSD parameters: (log10A (log-amplitude at reference freq of 1/yr),
                                 log10fc (log10 of corner frequency, in units 1/yr)
                                 gamma (spectral index))

    freqs     : Array, size = (nfreq,)
                Frequencies to evaluate PSD at, in units of 1/yr

    Returns
    -------

    PSD       : Array, size = (nfreq,)
                Power-spectral density, in units of s^2 d
    """

    logA = pars[0]
    logfc = pars[1]
    gamma = pars[2]

    log10Asquared = 2 * logA + LOG10YR3_TO_S2D
    fc = 10**logfc

    log10norm = log10Asquared - jnp.log10(12 * jnp.pi**2) + jnp.log10(fc ** (-gamma))

    log10psd = log10norm + jnp.log10((1 + (freqs / fc) ** 2) ** (-gamma / 2))

    return log10psd


def bpl_flattail_log10powspec(pars, freqs):
    """
    Evaluates a smoothly broken power-law PSD with a flat-tail

    PSD = (A^2 yr^3 / 12 pi^2) * (fc / fyr)^(-gamma) * (1 + (f / fc)^2)^(-gamma/2)+
              kappa^2

    Parameters
    ----------

    pars      : Array, size = (3,) (or longer)
                PSD parameters: (log10A (log-amplitude at reference freq of 1/yr),
                                 log10fc (log10 of corner frequency, in units 1/yr)
                                 gamma (spectral index),
                                 log10kappa (log-amplitude of flat tail))

    freqs     : Array, size = (nfreq,)
                Frequencies to evaluate PSD at, in units of 1/yr

    Returns
    -------

    PSD       : Array, size = (nfreq,)
                Power-spectral density, in units of s^2 d
    """

    log10bpl = bpl_log10powspec(pars, freqs)
    log10kappa = pars[3]
    ln10 = jnp.log(10.0)

    log10flat = 2 * log10kappa + LOG10YR3_TO_S2D
    log10psd = (
        jax.scipy.special.logsumexp(
            jnp.concatenate(
                (
                    log10bpl[:, None] * ln10,
                    jnp.ones_like(log10bpl)[:, None] * log10flat * ln10,
                ),
                axis=1,
            ),
            axis=1,
        )
        / ln10
    )

    return log10psd


class NoiseAndTimingModelSampler(TimingModelSampler):
    """A class for obtaining random samples of the noise parameters and timing model,
    given the latent variables and template pulse profile"""

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
        """
        Parameters
        ----------

        phi        : Array, size (nphotons,)
                     Photon phases

        M          : Array, size (nphotons, npars)
                     Design matrix

        theta_prior : Array, size (npars,)
                      Prior means for timing model parameters

        timing_prior_uncertainties : Array, size (npars,)
                      Prior widths for timing model parameters

        noise_models : list
                       shoogle noise models

        parameter_scales : Array, size (npars,)
                      Parameter scales used to ensure numerical stability

        PB0          : float, or None
                       Necessary if fitting OPVs to scale the orbital phase PSD
        """

        self.phi = phi
        self.npar = np.shape(M)[1]
        self.M = jnp.array(M)
        self.theta_prior = jnp.array(theta_prior)
        self.parameter_scales = parameter_scales

        self.K0 = jnp.array((timing_prior_uncertainties * parameter_scales) ** 2)
        self.noise_models = noise_models

        freqs = np.zeros((len(noise_models), len(parameter_scales)))
        free = np.zeros((len(noise_models), 4), dtype=bool)
        bounds = np.zeros((len(noise_models), 4, 2))
        hyp0 = np.zeros((len(noise_models), 4))
        min_freqs = np.zeros((len(noise_models)))

        scales = np.ones(len(noise_models))

        free_par_indices = np.array([], dtype="int")

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

            num_pars = len(self.noise_models[c].free)
            hyp0[c, :num_pars] = self.noise_models[c].x0
            free[c, :num_pars] = self.noise_models[c].free
            bounds[c, :num_pars, :] = self.noise_models[c].bounds

        self.free_par_indices = np.where(free)
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
        self.fixed_hyp = jnp.array(hyp0)
        self.min_freqs = jnp.array(min_freqs)
        self.linear_priors = jnp.array(linear_priors)

    def _phys_to_samples(self, hyp):
        """
        Transforms from physically meaningful hyperparmeters
        to the unconstrained sampling space (logit-transformed).

        Parameters
        ----------
        hyp        : jax.numpy.ndarray
                     Array of hyperparameters

        Returns
        -------
        x          : Internal co-ordinates used for sampling with blackjax
        """

        cube_pars = (hyp - self.hyp_bounds[:, 0]) / (
            self.hyp_bounds[:, 1] - self.hyp_bounds[:, 0]
        )
        return jax.scipy.special.logit(cube_pars)

    def _samples_to_phys(self, x):
        """
        Converts from parameters convenient for the sampler,
        which are logit-transformed to ranges of (-inf,inf),
        to physically-meaningful hyperparameters

        Parameters
        ----------

        x    :   jax.numpy.ndarray
                 Location of the current point in the sample parameter space

        Returns
        ---------

        hyp          : jax.numpy.ndarray
                       Hyper-parameters

        logprior     : float
                       the log-prior for these parameters, accounting for both
                       the priors (possible linear-uniform prior on noise amplitudes)
                       and the Jacobian of the logit transforms.
        """

        cube_pars = jax.scipy.special.expit(x)
        logprior = jnp.sum(jnp.log(cube_pars * (1 - cube_pars)))

        hyp = self.hyp_bounds[:, 0] + cube_pars * (
            self.hyp_bounds[:, 1] - self.hyp_bounds[:, 0]
        )

        amp_prior = 0.0
        for c in range(len(x)):
            amp_prior += jax.lax.cond(
                self.linear_priors[c], lambda: hyp[c] * np.log(10), lambda: 0.0
            )

        return hyp, logprior + amp_prior

    def _fill_noise_pars(self, hyp):
        """
        Obtains the full array hyperparameters (including fixed values)
        given values of the free parameters

        Parameters
        ----------

        hyp          : Array
                       Free hyperparameter values

        Returns
        -------

        all_hyp      : Array
                       All PSD parameters, including both free and fixed values
        """

        all_hyp = self.fixed_hyp.at[self.free_par_indices].set(hyp)

        return all_hyp

    def _make_psd_cov(self, all_hyp):
        """
        Computes the inverse prior covariance matrix

        Parameters
        ----------

        all_hyp     : Array
                      All necessary PSD parameters

        Returns
        -------

        inv_prior_cov : Array, size (npars, npars)
                        Inverse prior covariance matrix

        """

        K = self.K0.copy()

        for c in range(len(self.noise_models)):

            log10powspec = jnp.where(
                self.noise_freqs[c] > 0,
                (
                    bpl_flattail_log10powspec(all_hyp[c], self.noise_freqs[c])
                    + jnp.log10(self.min_freqs[c])
                    + jnp.log10(INVYR_TO_INVDAY)
                    + jnp.log10(self.scales[c])
                    + 2 * jnp.log10(self.parameter_scales)
                ),
                -1000,
            )

            K += 10 ** (log10powspec)

        Kinv = jnp.where(K > 0, 1.0 / K, 0.0)

        return jnp.diag(Kinv)

    def log_post(self, x, MT_Sigma_inv_M, MT_Sigma_inv_R):
        """
        Computes the log-posterior for the hyperparameters

        Parameters
        ----------

        x     : Array, size (nhyp_free,)
                Sampled values (in the logit-transformed space)

        MT_Sigma_inv_M : Array, size (npars, npars)
                         Inverse covariance matrix for timing parameter likelihood

        MT_Sigma_inv_R : Array, size (npars,)
                         RHS for WLS fit

        Returns
        -------

        logpost      : Log-posterior probability
        """

        hyp, logprior = self._samples_to_phys(x)
        all_hyp = self._fill_noise_pars(hyp)

        inv_prior_cov = self._make_psd_cov(all_hyp)

        theta_opt, V, Dinv, logdet_prior, prior_chi2 = self.solve_leastsq(
            MT_Sigma_inv_M, MT_Sigma_inv_R, inv_prior_cov
        )

        post_chi2 = jnp.sum((V @ (theta_opt / Dinv)) ** 2)
        logdet_post = -2 * jnp.sum(jnp.log(jnp.diag(V))) + 2 * jnp.sum(jnp.log(Dinv))
        logpost = logprior - 0.5 * (logdet_prior + prior_chi2 - logdet_post - post_chi2)

        return logpost

    def setup_sampler(self, mu_z, sigma_z, rng_key):
        """
        Performs a warm-up run to tune parameters (step size, covariance matrix)
        of the blackjax NUTS sampler for efficient sampling later.

        Parameters
        ----------
        mu_z           : Array, size (nphotons,)
                         Peak positions for each photon (0 if unpulsed/background)

        sigma_z        : Array, size (nphotons,)
                         Peak widths for each photon (0 if unpulsed/background)

        Returns
        -------

        hyp            : Array, size (nhyp_free,)
                         Starting values for free hyperparameters after burn-in

        key            : JAX PRNG key
        """

        x0 = self._phys_to_samples(self.hyp_0)

        MT_Sigma_inv_M, MT_Sigma_inv_R = self.setup_leastsq_given_z_tau(mu_z, sigma_z)

        logprob_fn = lambda x: self.log_post(x, MT_Sigma_inv_M, MT_Sigma_inv_R)

        warmup = blackjax.window_adaptation(
            blackjax.nuts,
            logprob_fn,
            progress_bar=True,
            target_acceptance_rate=0.9,
            is_mass_matrix_diagonal=True,
            max_num_doublings=15,
            integrator=blackjax.mcmc.integrators.yoshida,
        )

        warmup_key, sample_key = jax.random.split(rng_key, 2)
        (state, parameters), info = warmup.run(warmup_key, x0, num_steps=2000)

        self.nuts_params = parameters
        print(info)

        step_fn = blackjax.nuts.build_kernel(blackjax.mcmc.integrators.yoshida)

        @jax.jit
        def kernel(rng_key, state, MT_Sigma_inv_M, MT_Sigma_inv_R, step_size):

            logprob_fn = lambda x: self.log_post(x, MT_Sigma_inv_M, MT_Sigma_inv_R)
            state, info = step_fn(
                rng_key=rng_key,
                state=state,
                logdensity_fn=logprob_fn,
                inverse_mass_matrix=self.nuts_params["inverse_mass_matrix"],
                max_num_doublings=self.nuts_params["max_num_doublings"],
                step_size=step_size,
            )

            return state, info

        self.logprob_fn = jax.jit(self.log_post)
        self.kernel = kernel

        return self._samples_to_phys(state.position)[0], sample_key

    def sample(self, hyp, mu_z, sigma_z, key, step_size, num_samples=1000):
        """
        Run the blackjax NUTS sampler to estimate hyperparameters

        Parameters
        ----------

        hyp       :   Array
                      Initial hyperparameters as a starting point

        mu_z           : Array, size (nphotons,)
                         Peak positions for each photon (0 if unpulsed/background)

        sigma_z        : Array, size (nphotons,)
                         Peak widths for each photon (0 if unpulsed/background)

        key         : JAX PRNG key

        num_samples : Number of samples to draw, default = 1000

        Returns
        ----------
        samples   :   Array
                      Hyperparameter samples

        key         : JAX PRNG key
        """

        MT_Sigma_inv_M, MT_Sigma_inv_R = self.setup_leastsq_given_z_tau(mu_z, sigma_z)
        logprob_fn = lambda x: self.logprob_fn(x, MT_Sigma_inv_M, MT_Sigma_inv_R)

        x = self._phys_to_samples(hyp)
        state = blackjax.nuts.init(x, logprob_fn)

        one_step = lambda state, key: self.kernel(
            key, state, MT_Sigma_inv_M, MT_Sigma_inv_R, step_size
        )
        hyp_keys = jax.random.split(key, num_samples + 1)

        state, infos = jax.lax.scan(one_step, state, hyp_keys[:-1])
        loop_state = (state, hyp_keys[-1], step_size, infos)
        diverged0 = infos.is_divergent[-1]
        accept0 = infos.acceptance_rate[-1]

        def diverged(loop_state):
            infos = loop_state[-1]
            step_size = loop_state[2]

            return (infos.is_divergent[-1] | (infos.acceptance_rate[-1] < 0.2)) & (
                step_size > self.nuts_params["step_size"] / 16
            )

        def body_fun(loop_state):
            state = loop_state[0]
            key = loop_state[1]
            keys = jax.random.split(key, num_samples + 1)
            step_size = loop_state[2] / 2.0

            one_step = lambda state, key: self.kernel(
                key, state, MT_Sigma_inv_M, MT_Sigma_inv_R, step_size
            )
            state, infos = jax.lax.scan(one_step, state, keys[:-1])

            return (state, keys[-1], step_size, infos)

        state, key, step_size, infos = jax.lax.while_loop(
            diverged, body_fun, loop_state
        )

        sample = self._samples_to_phys(state.position)[0]

        return sample, hyp_keys[-1], accept0, diverged0, step_size

    def sample_lambda_theta_given_tau_zm(self, hyp, mu_z, sigma_z, key, num_steps=10):
        """
        Draw random samples of hyperparameters (via NUTS) and timing model
        parameters (from Gaussian posterior), given template pulse profile and
        photon--peak assignments

        Parameters
        ----------

        hyp        :   Array
                      Initial hyperparameters as a starting point

        mu_z       : Array, size (nphotons,)
                         Peak positions for each photon (0 if unpulsed/background)

        sigma_z    : Array, size (nphotons,)
                         Peak widths for each photon (0 if unpulsed/background)

        key        : JAX PRNG key

        num_steps  : Number of steps to run before drawing a sample
                      to reduce dependence on the starting point,
                      default = 10

        Returns
        ----------
        new_hyp   :   Array
                      New draw of hyperparameters

        theta     :   Array
                      New draw of timing model parameters

        phase_shifts   : Array, size (nphotons,)
                         Phase shifts according to theta, relative to theta=0

        key         : JAX PRNG key

        """

        step_size = self.nuts_params["step_size"]
        new_hyp, key, divergent, diverged0, step_size = self.sample(
            hyp, mu_z, sigma_z, key, step_size, num_steps
        )

        all_hyp = self._fill_noise_pars(new_hyp)
        inv_prior_cov = self._make_psd_cov(all_hyp)

        theta, phase_shifts, key = self.sample_theta_given_lambda_tau_zm(
            mu_z, sigma_z, inv_prior_cov, key
        )

        return new_hyp, theta, phase_shifts, key, divergent, diverged0, step_size

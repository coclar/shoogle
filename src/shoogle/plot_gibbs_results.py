#!/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from scipy.special import gamma, logsumexp
from scipy.stats import norm, multivariate_normal
from scipy.linalg import cholesky, cho_solve, solve
from sklearn.gaussian_process.kernels import Matern
from sklearn.cluster import mean_shift
from matplotlib.gridspec import GridSpec
import copy
import warnings
import corner
from emcee.autocorr import function_1d
import sys
from shoogle.utils import *
from pint.templates.lctemplate import LCTemplate
from pint.templates.lceprimitives import LCGaussian


def fermi_lc(
    phases,
    weights,
    ax=None,
    xbins=30,
    bgstyle="--",
    bgcolor="red",
    centre=None,
    exposure=None,
    exp_phases=None,
):
    """Plots a histogram from weighted Fermi photons (e.g. a pulse profile, or
    orbital light curve). Plots the background level (computed from the
    distribution of weights, and optionally the folded exposure."""

    if centre:
        phases = np.mod(phases - centre + 0.5, 1.0) + centre - 0.5
    else:
        centre = 0.5

    wcounts, bins = np.histogram(phases, xbins, weights=weights)
    sqerrors, bins = np.histogram(phases, xbins, weights=(weights**2.0))
    width = bins[1] - bins[0]
    bg = (weights.sum() - ((weights**2.0).sum())) / xbins
    src = (weights**2.0).sum() / xbins
    errors = np.sqrt(sqerrors + 1)

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(7.5, 5))

    ax.bar(
        np.append(bins[:-1], bins[:-1] + 1.0),
        np.append(wcounts, wcounts),
        width=width,
        color="lightgray",
        edgecolor="lightgray",
        capsize=0,
        linewidth=1.0,
        align="edge",
    )
    ax.errorbar(
        np.append(bins[:-1], bins[:-1] + 1.0) + width / 2.0,
        np.append(wcounts, wcounts),
        yerr=np.append(errors, errors),
        ecolor="black",
        elinewidth=1.0,
        color="none",
    )
    ax.step(
        np.append(bins[:-1], bins + 1.0),
        np.append(np.append(wcounts, wcounts), wcounts[0]),
        color="black",
        where="post",
        linewidth=1.0,
    )
    ax.axhline(bg, linestyle=bgstyle, color=bgcolor, linewidth=1.5)
    ax.set_xlim(left=centre - 0.5, right=centre + 1.5)
    ax.set_ylim(bottom=0)
    ax.tick_params(axis="both", which="major")
    ax.set_ylabel("Weighted Counts")

    ax.xaxis.set_visible(False)
    if exposure is not None:
        binned_exp, exp_bins = np.histogram(exp_phases, nbins, weights=exposure)
        ax.step(
            exp_bins,
            np.append(binned_exp, binned_exp[-1]),
            where="post",
            lw=3,
            color="black",
        )

    return bins, wcounts, errors


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


class GibbsResults(object):
    def __init__(self, psr):
        self.psr = psr

    def load_results(self, load_from=None, decimated=False):
        decimate_factor = 2

        loglike_chain = np.array([]).reshape(0)
        timing_chain = np.array([]).reshape(0, self.psr.n_timing_pars)

        nhyp = 0
        for component in self.psr.noise_models:
            nhyp += component.nfree

        if self.psr.has_OPV:
            opv_amps_chain = np.array([]).reshape(0, 2 * self.psr.nOPVfreqs)

        hyp_chain = np.array([]).reshape(0, nhyp)

        template_chain = np.array([]).reshape(0, len(self.psr.tau_sampler.tau_0))

        for f in load_from:
            dat = np.load(f)
            try:
                temp_loglike_chain = dat["loglike_chain"]
                temp_timing_chain = dat["timing_chain"]
                if self.psr.fit_TN or self.psr.fit_OPV:
                    temp_hyp_chain = dat["hyp_chain"]
                if self.psr.has_OPV:
                    temp_opv_amps_chain = dat["opv_amps_chain"]
                    opv_freqs = dat["opv_freqs"]
                temp_template_chain = dat["template_chain"]

            except:
                continue

            parameter_scales = dat["parameter_scales"]
            parameter_names = dat["parameter_names"]
            timing_parameter_values = dat["timing_parameter_values"]
            timing_parameter_uncertainties = dat["timing_parameter_uncertainties"]

            # PINT sometimes loads parameters from the same .par file in a different order?!
            reordered_timing_chain = np.copy(temp_timing_chain)
            reordered_param_names = np.copy(parameter_names)
            reordered_param_values = np.copy(timing_parameter_values)
            reordered_param_uncertainties = np.copy(timing_parameter_uncertainties)
            reordered_param_scales = np.copy(parameter_scales)

            for i, par in enumerate(parameter_names):
                new_idx = np.where(self.psr.timing_parameter_names == par)[0][0]
                reordered_timing_chain[:, new_idx] = temp_timing_chain[:, i]
                reordered_param_names[new_idx] = parameter_names[i]
                reordered_param_scales[new_idx] = parameter_scales[i]
                reordered_param_values[new_idx] = timing_parameter_values[i]
                reordered_param_uncertainties[new_idx] = timing_parameter_uncertainties[
                    i
                ]

            temp_timing_chain = reordered_timing_chain
            parameter_names = reordered_param_names
            parameter_scales = reordered_param_scales
            timing_parameter_values = reordered_param_values
            timing_parameter_uncertainties = reordered_param_uncertainties

            for k in range(self.psr.npeaks):
                mu = temp_template_chain[:, self.psr.npeaks + k]

                # Peaks close to the wrap can jump either side, let's undo that
                if np.std(mu) > 0.25:
                    mu = np.mod(mu, 1.0)
                    temp_template_chain[:, self.psr.npeaks + k] = mu

                # If that didn't help, undo the undoing!
                if np.std(mu) > 0.25:
                    mu = np.mod(mu + 0.5, 1.0) - 0.5
                    temp_template_chain[:, self.psr.npeaks + k] = mu

            if self.psr.Edep:
                p0 = self.psr.npeaks * 3
                for k in range(self.psr.npeaks):
                    mu = temp_template_chain[:, p0 + self.psr.npeaks + k]

                    # Peaks close to the wrap can jump either side, let's undo that
                    if np.std(mu) > 0.25:
                        mu = np.mod(mu, 1.0)
                        temp_template_chain[:, p0 + self.psr.npeaks + k] = mu

                    # If that didn't help, undo the undoing!
                    if np.std(mu) > 0.25:
                        mu = np.mod(mu + 0.5, 1.0) - 0.5
                        temp_template_chain[:, p0 + self.psr.npeaks + k] = mu

            if decimated:
                loglike_chain = np.append(loglike_chain, temp_loglike_chain, axis=0)
                timing_chain = np.append(timing_chain, temp_timing_chain, axis=0)
                if self.psr.has_OPV:
                    opv_amps_chain = np.append(
                        opv_amps_chain, temp_opv_amps_chain, axis=0
                    )
                if self.psr.fit_TN or self.psr.fit_OPV:
                    hyp_chain = np.append(hyp_chain, temp_hyp_chain, axis=0)
                template_chain = np.append(template_chain, temp_template_chain, axis=0)
            else:
                # Ignoring template for now, since label-switching causes trouble
                chains = np.copy(
                    temp_timing_chain
                )  # np.append(temp_timing_chain, temp_template_chain,axis=1)

                if self.psr.fit_TN or self.psr.fit_OPV:
                    chains = np.append(chains, temp_hyp_chain, axis=1)

                if self.psr.has_OPV:
                    chains = np.append(chains, temp_opv_amps_chain, axis=1)

                tau, j = acor(chains)
                steps = np.maximum(tau // decimate_factor, 10)
                burnin = tau * 5

                chains = chains[burnin:, :]
                tau, j = acor(chains)
                steps = np.maximum(tau // decimate_factor, 10)

                print(
                    f"Loading from {f}, tau = {tau}, samples={len(temp_loglike_chain)}, decimated samples={(len(temp_loglike_chain) - burnin)// (tau//decimate_factor)}"
                )

                loglike_chain = np.append(
                    loglike_chain, temp_loglike_chain[-1:burnin:-steps][::-1]
                )
                timing_chain = np.append(
                    timing_chain, temp_timing_chain[-1:burnin:-steps][::-1], axis=0
                )

                if self.psr.has_OPV:
                    opv_amps_chain = np.append(
                        opv_amps_chain,
                        temp_opv_amps_chain[-1:burnin:-steps][::-1],
                        axis=0,
                    )
                if self.psr.fit_TN or self.psr.has_OPV:
                    hyp_chain = np.append(
                        hyp_chain, temp_hyp_chain[-1:burnin:-steps][::-1], axis=0
                    )
                template_chain = np.append(
                    template_chain,
                    temp_template_chain[-1:burnin:-steps][::-1],
                    axis=0,
                )

                output_dict = {
                    "loglike_chain": loglike_chain,
                    "timing_chain": timing_chain,
                    "template_chain": template_chain,
                    "parameter_names": parameter_names,
                    "parameter_scales": parameter_scales,
                    "timing_parameter_values": timing_parameter_values,
                    "timing_parameter_uncertainties": timing_parameter_uncertainties,
                }

                if self.psr.fit_TN or self.psr.fit_OPV:
                    output_dict["hyp_chain"] = hyp_chain

                if self.psr.has_OPV:
                    output_dict["opv_freqs"] = opv_freqs
                    output_dict["opv_amps_chain"] = opv_amps_chain

                # np.savez('decimated_chains.npz',output_dict)

        self.loglike_chain = loglike_chain
        self.timing_chain = timing_chain
        if self.psr.has_OPV:
            self.opv_amps_chain = opv_amps_chain
        if self.psr.fit_TN or self.psr.fit_OPV:
            self.hyp_chain = hyp_chain
        self.template_chain = template_chain
        self.parameter_names = parameter_names
        self.parameter_scales = parameter_scales
        self.timing_parameter_values = timing_parameter_values
        self.timing_parameter_uncertainties = timing_parameter_uncertainties

        self.phys_timing_chain = (
            timing_parameter_values
            + timing_chain / parameter_scales[None, : self.psr.n_timing_pars]
        )

        self._find_MAP()

    def _find_MAP(self):

        all_chains = np.append(self.timing_chain, self.template_chain, axis=1)

        if self.psr.fit_TN or self.psr.fit_OPV:
            all_chains = np.append(all_chains, self.hyp_chain, axis=1)
        if self.psr.has_OPV:
            all_chains = np.append(all_chains, self.opv_amps_chain, axis=1)

        MAP, _ = mean_shift(all_chains, seeds=np.mean(all_chains, axis=0)[None, :])
        MAP = MAP[0, :]

        e = self.psr.n_timing_pars
        self.timing_MAP = MAP[:e]
        s = self.psr.n_timing_pars
        n = len(self.psr.tau_sampler.tau_0)
        self.template_MAP = MAP[s : s + n]
        s = s + n

        if self.psr.fit_TN or self.psr.fit_OPV:
            n = self.psr.nhyp
            self.hyp_MAP = MAP[s : s + n]
            s = s + n

        if self.psr.has_OPV:
            self.opv_amps_MAP = MAP[s:]

        self.phys_timing_MAP = (
            self.timing_parameter_values
            + self.timing_MAP / self.parameter_scales[: self.psr.n_timing_pars]
        )

        self.MAP_timing_model = copy.deepcopy(self.psr.timing_model)

        if self.psr.has_OPV:
            self.ORBWAVES_MAP = (
                self.opv_amps_MAP - self.psr.theta_prior[self.psr.n_timing_pars :]
            ) / self.parameter_scales[self.psr.n_timing_pars :]

        for i in range(self.psr.n_timing_pars):

            try:
                getattr(self.MAP_timing_model, self.parameter_names[i]).value = (
                    self.phys_timing_MAP[i]
                )
            except:
                continue

            if self.parameter_names[i] == "RAJ":
                unit = u.hourangle
            elif self.parameter_names[i] == "DECJ":
                unit = u.deg
            else:
                unit = 1

            getattr(self.MAP_timing_model, self.parameter_names[i]).uncertainty = (
                np.std(self.phys_timing_chain[:, i]) * unit
            )

        if self.psr.has_OPV:
            for nf in range(self.psr.nOPVfreqs):
                getattr(self.MAP_timing_model, f"ORBWAVES{nf}").value = (
                    self.ORBWAVES_MAP[2 * nf]
                )
                getattr(self.MAP_timing_model, f"ORBWAVEC{nf}").value = (
                    self.ORBWAVES_MAP[2 * nf + 1]
                )

    def write_new_parfile(self, output_parfile):

        tm = self.MAP_timing_model
        pf_str = tm.as_parfile()

        c = 0
        i = 0

        for component in self.psr.noise_models:
            for i in range(component.npar):
                if component.free[i]:
                    hyp_mean = self.hyp_MAP[c]
                    hyp_free = 1
                    hyp_uncert = np.std(self.hyp_chain[:, c])
                    c += 1
                else:
                    hyp_mean = component.x0[i]
                    hyp_free = 0
                    hyp_uncert = 0.0

                par_name = f"{component.prefix}_{component.param_names[i]}"
                pf_str += (
                    f"{par_name:15s} {hyp_mean:8.2f} {hyp_free:d} {hyp_uncert:8.2f}\n"
                )

        with open(output_parfile, "w") as par:
            par.write(pf_str)

    def write_orbifunc_parfile(self, output_parfile):

        tm = self.MAP_timing_model
        orbwaves = self.ORBWAVES_MAP

        with open(output_parfile, "w") as outpar:
            pf_str = tm.as_parfile().split("\n")

            for line in pf_str:
                if "ORBWAVE" in line:
                    continue
                else:
                    outpar.write(line + "\n")

            outpar.write(f"SORBIFUNC           2         0\n")
            orbifunc_t = np.linspace(self.psr.t.min(), self.psr.t.max(), 1000)

            WOM = tm.ORBWAVE_OM.quantity.to_value("rad/d")
            WEPOCH = tm.ORBWAVE_EPOCH.value

            Morbifunc = np.zeros((len(orbifunc_t), 2 * len(self.psr.OPVfreqs)))
            for nf in range(self.psr.nOPVfreqs):
                ii = 2 * nf
                Morbifunc[:, ii] = np.sin(WOM * (nf + 1) * (orbifunc_t - WEPOCH))
                Morbifunc[:, ii + 1] = np.cos(WOM * (nf + 1) * (orbifunc_t - WEPOCH))

            orbifunc = Morbifunc @ orbwaves
            for i in range(len(orbifunc_t)):
                outpar.write(
                    f"ORBIFUNC{i+1}        {orbifunc_t[i]:.10f} {orbifunc[i]:.10f}  0.0\n"
                )

    def write_new_template(self, output_templatefile):

        A = self.template_MAP[: self.psr.npeaks]
        mu = self.template_MAP[self.psr.npeaks : 2 * self.psr.npeaks]
        sigma = self.template_MAP[2 * self.psr.npeaks :]

        if self.psr.Edep:
            write_template(output_templatefile + ".Elo", A, mu, sigma)

            p0 = self.psr.npeaks * 3
            A = self.template_MAP[p0 : p0 + self.psr.npeaks]
            mu = self.template_MAP[p0 + self.psr.npeaks : p0 + 2 * self.psr.npeaks]
            sigma = self.template_MAP[p0 + 2 * self.psr.npeaks :]
            write_template(output_templatefile + ".Ehi", A, mu, sigma)
        else:
            write_template(output_templatefile, A, mu, sigma)

    def MAP_phases(self):

        if self.psr.has_OPV:
            theta_MAP = np.concatenate((self.timing_MAP, self.opv_amps_MAP))
        else:
            theta_MAP = self.timing_MAP
        phase_shifts = self.psr.M @ theta_MAP
        self.phi_MAP = np.mod(self.psr.phi - phase_shifts, 1.0)

        if hasattr(self.psr, "radio_toas"):
            radio_phase_shifts = self.psr.Mradio @ theta_MAP
            self.radio_resids_MAP = self.psr.radio_resids - radio_phase_shifts

    def write_tvsph(self, outputfile, phi):
        np.savetxt(
            outputfile,
            np.array(
                [
                    self.MAP_timing_model.get_barycentric_toas(
                        self.psr.photon_toas
                    ).value,
                    phi,
                    self.psr.w,
                ]
            ).T,
        )

    def hyp_corner(self):

        names = []
        for component in self.psr.noise_models:
            for i in range(component.npar):
                if component.free[i]:
                    names.append(f"{component.prefix}{component.param_names[i]}")

        exp_lev = np.array([1, 2, 3])
        mp_array = 1 - np.exp(-0.5 * exp_lev**2)

        fig = corner.corner(
            self.hyp_chain,
            bins=50,
            truths=self.hyp_MAP,
            labels=names,
            range=[0.999] * len(self.hyp_MAP),
            quantiles=[0.05, 0.5, 0.95],
            levels=mp_array,
            show_titles=True,
        )

        return fig

    def timing_corner(self, plot_wxcomp=False, nWX=10):
        if plot_wxcomp == False:
            range_plot = self.psr.n_timing_pars - (nWX * 2)
        else:
            range_plot = self.psr.n_timing_pars

        exp_lev = np.array([1, 2, 3])
        mp_array = 1 - np.exp(-0.5 * exp_lev**2)

        fig = corner.corner(
            self.phys_timing_chain[:, :range_plot],
            labels=self.parameter_names[:range_plot],
            truths=self.phys_timing_MAP[:range_plot],
            hist_kwargs={"density": True},
            levels=mp_array,
            bins=30,
        )

        ax = np.array(fig.get_axes()).reshape(range_plot, range_plot)

        for i in range(range_plot):
            if self.timing_parameter_uncertainties[i] > 0.0:
                vec = np.linspace(
                    self.phys_timing_chain[:, i].min(),
                    self.phys_timing_chain[:, i].max(),
                    1000,
                )
                ax[i, i].plot(
                    vec,
                    norm.pdf(
                        vec,
                        loc=self.timing_parameter_values[i],
                        scale=self.timing_parameter_uncertainties[i],
                    ),
                    color="red",
                )

        return fig

    def template_corner(self):

        exp_lev = np.array([1, 2, 3])
        mp_array = 1 - np.exp(-0.5 * exp_lev**2)

        fig = corner.corner(
            self.template_chain[:, : 3 * self.psr.npeaks],
            labels=[f"$A_{i}$" for i in range(self.psr.npeaks)]
            + [f"$\\mu_{i}$" for i in range(self.psr.npeaks)]
            + [f"$\\sigma_{i}$" for i in range(self.psr.npeaks)],
            truths=self.template_MAP[: 3 * self.psr.npeaks],
            truths_color="C0",
            levels=mp_array,
            bins=30,
            color="C0",
        )
        if self.psr.Edep:
            corner.corner(
                self.template_chain[:, 3 * self.psr.npeaks :],
                fig=fig,
                labels=[f"$A_{i}$" for i in range(self.psr.npeaks)]
                + [f"$\\mu_{i}$" for i in range(self.psr.npeaks)]
                + [f"$\\sigma_{i}$" for i in range(self.psr.npeaks)],
                truths=self.template_MAP[3 * self.psr.npeaks :],
                truths_color="C1",
                levels=mp_array,
                bins=30,
                color="C1",
            )

        return fig

    def opv_corner(self):

        fig = corner.corner(
            self.opv_amps_chain[:, :12],
            truths=self.opv_amps_MAP[:12],
            bins=30,
        )
        return fig

    def plot_radio_resids(self, ax=None):

        if ax is None:
            fig, ax = plt.subplots(2, 2, sharex="col", figsize=(15, 10))

        ophi = np.mod(
            (self.psr.radio_t - self.psr.timing_model.TASC.quantity.tdb.mjd)
            / self.psr.PB0.to_value("d"),
            1.0,
        )
        ax[0, 0].errorbar(
            self.psr.radio_t,
            self.psr.radio_resids,
            yerr=self.psr.radio_toa_uncerts,
            color="red",
            ls="none",
            marker="x",
        )
        ax[1, 0].errorbar(
            self.psr.radio_t,
            self.radio_resids_MAP,
            yerr=self.psr.radio_toa_uncerts,
            color="green",
            ls="none",
            marker="x",
        )
        ax[0, 1].errorbar(
            ophi,
            self.psr.radio_resids,
            yerr=self.psr.radio_toa_uncerts,
            color="red",
            ls="none",
            marker="x",
        )
        ax[1, 1].errorbar(
            ophi,
            self.radio_resids_MAP,
            yerr=self.psr.radio_toa_uncerts,
            color="green",
            ls="none",
            marker="x",
        )

        ax[1, 0].set_xlabel("Time (MJD)")
        ax[1, 1].set_xlabel("Orbital phase")
        ax[0, 0].set_ylabel("Pre-fit residual")
        ax[1, 0].set_ylabel("Post-fit residual")
        ax[0, 1].set_ylabel("Pre-fit residual")
        ax[1, 1].set_ylabel("Post-fit residual")

        return ax

    def scatter_phases(self, phi, ax):
        w = np.append(self.psr.w, self.psr.w)
        idx = np.argsort(w)
        w = w[idx]
        t = np.append(self.psr.t, self.psr.t)[idx]
        phi = np.append(phi, phi + 1)[idx]

        ax.scatter(phi, t, 5, c=w, cmap="binary", vmin=0, vmax=1, rasterized=True)
        ax.set_xlim(0, 2)
        ax.set_ylim(t.min(), t.max())

        ax.set_xlabel("Rotational Phase")
        ax.set_ylabel("Time (MJD)")
        ax.xaxis.set_tick_params(bottom=True, labelbottom=True)

    def plot_templates(self, ax, xbins=50):

        tau = self.template_MAP

        phi_vec = np.arange(0, 1.0, 0.001)

        S = np.sum(self.psr.w**2) / xbins
        B = np.sum(self.psr.w * (1 - self.psr.w)) / xbins

        if self.psr.Edep:
            Eavg = np.average(self.psr.log10E, weights=self.psr.w)
            ax.plot(
                phi_vec,
                self.psr.tau_sampler.template(
                    tau, phi_vec, np.ones_like(phi_vec) * Eavg
                )
                * S
                + B,
                color="orange",
                lw=1,
                zorder=10,
            )
        else:
            ax.plot(
                phi_vec,
                self.psr.tau_sampler.template(tau, phi_vec) * S + B,
                color="orange",
                lw=1,
                zorder=10,
            )

        for i in np.random.choice(
            np.arange(np.shape(self.template_chain)[0]), replace=True, size=100
        ):
            tau = self.template_chain[i, :]
            if self.psr.Edep:
                ax.plot(
                    phi_vec,
                    self.psr.tau_sampler.template(
                        tau, phi_vec, np.ones_like(phi_vec) * Eavg
                    )
                    * S
                    + B,
                    color="black",
                    alpha=0.2,
                    lw=1,
                )
            else:
                ax.plot(
                    phi_vec,
                    self.psr.tau_sampler.template(tau, phi_vec) * S + B,
                    color="black",
                    alpha=0.2,
                    lw=1,
                )

    def plot_tasc_shifts(
        self,
        ax,
        color="black",
        offset=0,
        tasc_slwin=None,
        dtasc_r=None,
        dtasc_l=None,
        tasc_offset=None,
    ):

        if tasc_slwin is not None:
            dtasc_step = (dtasc_r - dtasc_l) / np.shape(tasc_slwin)[0]
            extent = [
                dtasc_l - dtasc_step / 2,
                dtasc_r + dtasc_step / 2,
                self.psr.t.min(),
                self.psr.t.max(),
            ]

            c = ax.imshow(
                tasc_slwin.T,
                vmin=0,
                vmax=30,
                extent=extent,
                origin="lower",
                zorder=-1,
                cmap="binary",
                aspect="auto",
            )
            plt.colorbar(
                c,
                orientation="vertical",
                label="$\\log \\mathcal{L}$",
                pad=0.01,
                fraction=0.075,
                aspect=40,
            )

        theta_base = np.concatenate((self.timing_MAP, np.zeros_like(self.opv_amps_MAP)))

        theta_prior = np.copy(self.psr.theta_prior)
        if "FB1" in self.parameter_names:
            fb1_idx = np.where(self.parameter_names == "FB1")[0][0]
            theta_prior[fb1_idx] = (
                -self.psr.timing_parameter_values[fb1_idx]
                * self.psr.parameter_scales[fb1_idx]
            )

        mu_orb_base = self.psr.Mo @ theta_base

        mu_orbs = np.zeros((len(self.psr.t), 1000))

        for c, i in enumerate(
            np.random.choice(
                np.arange(np.shape(self.timing_chain)[0]), replace=True, size=1000
            )
        ):
            theta = np.concatenate((self.timing_chain[i], self.opv_amps_chain[i]))
            mu_orbs[:, c] = self.psr.Mo @ (theta - theta_prior) - mu_orb_base

        mu_orbs = np.sort(mu_orbs, axis=1)
        mu_orbs -= np.mean(mu_orbs)

        mu_sample_2sig_lo = np.quantile(mu_orbs, norm.cdf(-2.0), axis=1)
        mu_sample_2sig_hi = np.quantile(mu_orbs, norm.cdf(2.0), axis=1)
        mu_sample_1sig_lo = np.quantile(mu_orbs, norm.cdf(-1.0), axis=1)
        mu_sample_1sig_hi = np.quantile(mu_orbs, norm.cdf(1.0), axis=1)

        ax.fill_betweenx(
            self.psr.t,
            mu_sample_2sig_lo,
            mu_sample_2sig_hi,
            alpha=0.25,
            facecolor=color,
        )
        ax.fill_betweenx(
            self.psr.t,
            mu_sample_1sig_lo,
            mu_sample_1sig_hi,
            alpha=0.25,
            facecolor=color,
        )
        ax.set_xlabel("$\\Delta T_{\\rm asc}$ (s)")

    def plot_phase_shifts(self, ax, nsamps=1000, color="black"):

        theta_MAP = np.copy(self.timing_MAP)

        if self.psr.has_TN:
            theta_MAP[self.psr.WXSinds] -= self.psr.theta_prior[self.psr.WXSinds]
            theta_MAP[self.psr.WXCinds] -= self.psr.theta_prior[self.psr.WXCinds]

        if self.psr.has_OPV:
            theta_MAP = np.append(theta_MAP, self.opv_amps_MAP)

        mu_MAP = self.psr.M @ theta_MAP

        mu_samples = np.zeros((len(self.psr.t), nsamps))

        for c, i in enumerate(
            np.random.choice(
                np.arange(np.shape(self.timing_chain)[0]), replace=True, size=nsamps
            )
        ):
            theta = self.timing_chain[i]
            if self.psr.has_OPV:
                theta = np.append(theta, self.opv_amps_chain[i])
            mu_samples[:, c] = self.psr.M @ theta - mu_MAP
            if c % 10 == 0:
                print(c, "/", nsamps, end="\r")

        mu_samples = np.sort(mu_samples, axis=1)

        mu_sample_2sig_lo = np.quantile(mu_samples, norm.cdf(-2.0), axis=1)
        mu_sample_2sig_hi = np.quantile(mu_samples, norm.cdf(2.0), axis=1)
        mu_sample_1sig_lo = np.quantile(mu_samples, norm.cdf(-1.0), axis=1)
        mu_sample_1sig_hi = np.quantile(mu_samples, norm.cdf(1.0), axis=1)

        ax.fill_betweenx(
            self.psr.t,
            mu_sample_2sig_lo,
            mu_sample_2sig_hi,
            alpha=0.25,
            facecolor=color,
        )
        ax.fill_betweenx(
            self.psr.t,
            mu_sample_1sig_lo,
            mu_sample_1sig_hi,
            alpha=0.25,
            facecolor=color,
        )
        ax.set_xlim(-0.5, 0.5)
        ax.set_xlabel("$\\Delta \\phi(t)$")

    def plot_PB_shifts(self, ax, color="black", offset=0):

        mu_orb_base_MAP = self.psr.Mo @ np.concatenate(
            (self.timing_MAP, np.zeros_like(self.opv_amps_MAP))
        )

        mu_orbs = np.zeros((len(self.psr.t), 1000))

        for c, i in enumerate(
            np.random.choice(
                np.arange(np.shape(self.timing_chain)[0]), replace=True, size=1000
            )
        ):

            theta = np.concatenate((self.timing_chain[i], self.opv_amps_chain[i]))
            mu_orbs[:, c] = (
                self.psr.Mo @ (theta - self.psr.theta_prior) - mu_orb_base_MAP
            )

        d_orbphi = -(mu_orbs / 86400) / self.psr.PB0
        dFB0 = np.gradient(d_orbphi, self.psr.t, axis=0)  # In orbits/day

        dPB = -dFB0 * (self.psr.PB0**2)

        dPB_over_PB = dPB / self.psr.PB0

        dPB_over_PB_sorted = np.sort(dPB_over_PB, axis=1)
        dPB_over_PB_min = dPB_over_PB_sorted[:, int((2.5 / 100) * 1000)]
        dPB_over_PB_max = dPB_over_PB_sorted[:, int(1000 - (2.5 / 100) * 1000)]

        ax.fill_betweenx(
            self.psr.t, dPB_over_PB_min, dPB_over_PB_max, alpha=0.5, facecolor=color
        )
        ax.set_xlabel("$\\Delta P_{\\rm orb}/P_{\\rm orb}$")

    def noise_psd(self, f, t, resids, precision, hyp, kind="TN"):

        if kind != "TN" and kind != "OPV":
            raise ValueError(
                'Must specify either kind="TN" or kind="OPV" for noise_psd function'
            )

        K = np.diag(1.0 / precision)

        s = 0
        for component in self.psr.noise_models:
            n = component.nfree
            p = component.all_parameters(hyp[s : s + n])
            print(p)
            if kind == "TN" and component.prefix[:2] == "TN":
                K += component.cov(t, p)
            elif kind == "OPV" and component.prefix[:3] == "OPV":
                K += component.cov(t, p)

            s += n

        Kinv = np.linalg.inv(K)

        Kinv_y = Kinv @ resids

        X = np.zeros((len(t), 3))
        X[:, 2] = 1.0
        # X[:, 3] = (t - np.mean(t)) / self.psr.Tobs
        # X[:, 4] = ((t - np.mean(t)) / self.psr.Tobs) ** 2
        mu_PSD_cho = np.zeros(len(f))

        path1 = np.einsum_path("ij,jk", Kinv, X, optimize="optimal")[0]
        path2 = np.einsum_path("ij,jk", X.T, X, optimize="optimal")[0]
        path3 = np.einsum_path("ij,j", X.T, resids, optimize="optimal")[0]

        PSD = np.zeros(len(f))
        for i in range(len(f)):
            X[:, 0] = np.cos(2.0 * np.pi * f[i].to_value("1/d") * t)
            X[:, 1] = np.sin(2.0 * np.pi * f[i].to_value("1/d") * t)

            # if NU > 2.0:
            Kinv_X = np.einsum("ij,jk", Kinv, X, optimize=path1)
            XT_Kinv_X = np.einsum("ij,jk", X.T, Kinv_X, optimize=path2)
            XT_Kinv_r = np.einsum("ij,j", X.T, Kinv_y, optimize=path3)

            chocov = np.linalg.inv(XT_Kinv_X)
            b = chocov @ XT_Kinv_r

            b_sample = multivariate_normal.rvs(size=1, mean=b, cov=chocov)

            PSD[i] = (b_sample[0] ** 2 + b_sample[1] ** 2) * self.psr.Tobs
            # mu_PSD_cho[i] = (b[0] ** 2 + b[1] ** 2) * self.psr.Tobs

            # The expected value of the Fourier power
            # P = S^2 + C^2
            # S and C have mean = 0, variance = diag((M^T K^-1 M)^-1)
            # mu_PSD_cho[i] = (chocov[0, 0] + chocov[1, 1]) * self.psr.Tobs

        return PSD  # , mu_PSD_cho

    def plot_opv_psds(self, ax, psd_units="s ** 2 * yr", freq_units="yr ** -1"):
        # WN_psd = np.sort(self.WN_spin_psd, axis=1)

        n_model_psd_samples = np.shape(self.fit_opv_psd)[1]
        idx68_up = int(n_model_psd_samples - (n_model_psd_samples * 16) // 100 - 1)
        idx68_lo = int((n_model_psd_samples * 16) // 100)
        idx95_up = int(n_model_psd_samples - (n_model_psd_samples * 2.5) // 100 - 1)
        idx95_lo = int((n_model_psd_samples * 2.5) // 100)

        psd_scale = (1.0 * u.s**2 * u.d).to_value(psd_units)

        obs_psd = np.sort(self.obs_opv_psd, axis=1) * psd_scale
        full_psd = np.sort(self.full_opv_psd, axis=1) * psd_scale
        fit_psd = np.sort(self.fit_opv_psd, axis=1) * psd_scale

        n_psd_samples = np.shape(obs_psd)[1]

        idx68_up = int(n_psd_samples - (n_psd_samples * 16) // 100 - 1)
        idx68_lo = int((n_psd_samples * 16) // 100)
        idx95_up = int(n_psd_samples - (n_psd_samples * 2.5) // 100 - 1)
        idx95_lo = int((n_psd_samples * 2.5) // 100)

        obs_psd_68up = obs_psd[:, idx68_up]
        obs_psd_68lo = obs_psd[:, idx68_lo]
        obs_psd_95up = obs_psd[:, idx95_up]
        obs_psd_95lo = obs_psd[:, idx95_lo]
        obs_psd_mean = np.median(obs_psd, axis=1)

        if hasattr(self.psr, "radio_toas"):
            obs_gamma_psd = np.mean(self.gamma_psd, axis=1)
            obs_radio_psd = np.mean(self.radio_psd, axis=1)

        full_psd_68up = full_psd[:, idx68_up]
        full_psd_68lo = full_psd[:, idx68_lo]
        full_psd_95up = full_psd[:, idx95_up]
        full_psd_95lo = full_psd[:, idx95_lo]
        full_psd_mean = np.median(full_psd, axis=1)

        if hasattr(self.psr, "radio_toas"):
            ax.plot(
                self.opv_psd_f.to_value("yr ** -1"),
                obs_gamma_psd,
                color="magenta",
                zorder=0,
            )

        fit_psd_mean = np.mean(fit_psd, axis=1)
        ax.plot(
            self.opv_psd_f.to_value("yr ** -1"), obs_psd_mean, color="black", zorder=0
        )
        ax.plot(
            self.opv_psd_f.to_value("yr ** -1"), full_psd_mean, color="red", zorder=0
        )

        fit_psd = np.zeros((len(self.psr.noise_models), len(self.opv_psd_f), 1000))
        for c, i in enumerate(
            np.random.choice(
                np.arange(np.shape(self.hyp_chain)[0]),
                replace=True,
                size=1000,
            )
        ):
            s = 0
            for m, component in enumerate(self.psr.noise_models):

                n = component.nfree
                p = component.all_parameters(self.hyp_chain[i, s : s + n])

                if component.prefix[:3] == "OPV":
                    fit_psd[m, :, c] = (
                        component.powspec(p, self.opv_psd_f)
                        * self.psr.PB0.to_value("s") ** 2
                    )

                s += n

        idx68_up = int(1000 - (1000 * 16) // 100 - 1)
        idx68_lo = int((1000 * 16) // 100)
        idx95_up = int(1000 - (1000 * 2.5) // 100 - 1)
        idx95_lo = int((1000 * 2.5) // 100)

        for m, component in enumerate(self.psr.noise_models):
            fit_psd_s = np.sort(fit_psd[m], axis=1) * psd_scale
            fit_psd_68up = fit_psd_s[:, idx68_up]
            fit_psd_68lo = fit_psd_s[:, idx68_lo]
            fit_psd_95up = fit_psd_s[:, idx95_up]
            fit_psd_95lo = fit_psd_s[:, idx95_lo]
            fit_psd_mean = np.median(fit_psd_s, axis=1)

            ax.plot(
                self.opv_psd_f.to_value("yr ** -1"),
                fit_psd_mean,
                color="blue",
                zorder=0,
                ls="--",
            )
            ax.plot(
                self.opv_psd_f.to_value("yr ** -1"),
                fit_psd_mean,
                color="blue",
                zorder=0,
                ls="--",
            )
            # ax.fill_between(
            #     self.psd_f.to_value("yr ** -1"),
            #     fit_psd_68lo,
            #     fit_psd_68up,
            #     facecolor="blue",
            #     zorder=2,
            #     alpha=0.3,
            # )
            # ax.fill_between(
            #     self.psd_f.to_value("yr ** -1"),
            #     fit_psd_95lo,
            #     fit_psd_95up,
            #     facecolor="blue",
            #     zorder=2,
            #     alpha=0.3,
            # )

        # ax.plot(self.psd_f.to_value("yr ** -1"), fit_psd_mean, color="blue", zorder=0)

        ax.fill_between(
            self.opv_psd_f.to_value("yr ** -1"),
            obs_psd_68lo,
            obs_psd_68up,
            facecolor="black",
            zorder=2,
            alpha=0.3,
        )

        ax.fill_between(
            self.opv_psd_f.to_value("yr ** -1"),
            obs_psd_95lo,
            obs_psd_95up,
            facecolor="black",
            zorder=2,
            alpha=0.2,
        )

        ax.fill_between(
            self.opv_psd_f.to_value("yr ** -1"),
            full_psd_68lo,
            full_psd_68up,
            facecolor="red",
            zorder=2,
            alpha=0.3,
        )

        ax.fill_between(
            self.opv_psd_f.to_value("yr ** -1"),
            full_psd_95lo,
            full_psd_95up,
            facecolor="red",
            zorder=2,
            alpha=0.2,
        )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.axvline(
            (1.0 / (self.psr.Tobs * u.d)).to_value("yr ** -1"), ls="--", color="black"
        )
        ax.axvline(1, ls=":", color="black")
        ax.set_xlabel("Frequency (1/yr)")
        ax.set_ylabel(
            f"Power spectral density (${psd_units.replace(' ** ','^').replace(' * ',' ')}$)"
        )
        ax.set_ylim(bottom=full_psd_mean[-1] / 5)
        ax.set_xlim(
            left=(1.0 / (1.5 * self.psr.Tobs * u.d)).to_value("yr ** -1"),
            right=self.opv_psd_f.to_value("yr ** -1")[-1],
        )

    def plot_psds(self, ax, psd_units="s ** 2 * yr", freq_units="yr ** -1"):
        fit_psd = (np.sort(self.fit_psd, axis=1) * u.s**2 * u.d).to_value(psd_units)
        n_model_psd_samples = np.shape(fit_psd)[1]

        idx68_up = int(n_model_psd_samples - (n_model_psd_samples * 16) // 100 - 1)
        idx68_lo = int((n_model_psd_samples * 16) // 100)
        idx95_up = int(n_model_psd_samples - (n_model_psd_samples * 2.5) // 100 - 1)
        idx95_lo = int((n_model_psd_samples * 2.5) // 100)

        fit_psd_68up = fit_psd[:, idx68_up]
        fit_psd_68lo = fit_psd[:, idx68_lo]
        fit_psd_95up = fit_psd[:, idx95_up]
        fit_psd_95lo = fit_psd[:, idx95_lo]
        fit_psd_mean = np.mean(fit_psd, axis=1)

        obs_psd = (np.sort(self.obs_psd, axis=1) * u.s**2 * u.d).to_value(psd_units)
        full_psd = (np.sort(self.full_psd, axis=1) * u.s**2 * u.d).to_value(psd_units)

        n_psd_samples = np.shape(obs_psd)[1]

        idx68_up = int(n_psd_samples - (n_psd_samples * 16) // 100 - 1)
        idx68_lo = int((n_psd_samples * 16) // 100)
        idx95_up = int(n_psd_samples - (n_psd_samples * 2.5) // 100 - 1)
        idx95_lo = int((n_psd_samples * 2.5) // 100)

        obs_psd_68up = obs_psd[:, idx68_up]
        obs_psd_68lo = obs_psd[:, idx68_lo]
        obs_psd_95up = obs_psd[:, idx95_up]
        obs_psd_95lo = obs_psd[:, idx95_lo]
        obs_psd_mean = np.mean(obs_psd, axis=1)

        if hasattr(self.psr, "radio_toas"):
            obs_gamma_psd = (np.mean(self.gamma_psd, axis=1) * u.s**2 * u.d).to_value(
                psd_units
            )
            obs_radio_psd = (np.mean(self.radio_psd, axis=1) * u.s**2 * u.d).to_value(
                psd_units
            )

        full_psd_68up = full_psd[:, idx68_up]
        full_psd_68lo = full_psd[:, idx68_lo]
        full_psd_95up = full_psd[:, idx95_up]
        full_psd_95lo = full_psd[:, idx95_lo]
        full_psd_mean = np.mean(full_psd, axis=1)

        f = self.psd_f.to_value(freq_units)
        if hasattr(self.psr, "radio_toas"):
            ax.plot(f, obs_gamma_psd, color="magenta", zorder=0)

        ax.plot(f, obs_psd_mean, color="black", zorder=0)
        # ax.fill_between(
        #     f,
        #     obs_psd_68lo,
        #     obs_psd_68up,
        #     facecolor="black",
        #     zorder=2,
        #     alpha=0.25,
        # )
        ax.fill_between(
            f,
            obs_psd_95lo,
            obs_psd_95up,
            facecolor="black",
            zorder=2,
            alpha=0.25,
        )
        # ax[0,1].fill_between(f,obs_psd_68lo,obs_psd_68up,facecolor='black',zorder=2,alpha=0.25)

        ax.plot(f, fit_psd_mean, color="red", zorder=1, alpha=0.5)
        # for i in range(np.shape(self.fit_psd)[1]):
        #     ax.plot(f,self.fit_psd[:,i],color='green',zorder=1,alpha=0.25)

        ax.plot(f, fit_psd_mean, color="red", zorder=1, alpha=0.5)
        ax.fill_between(
            f,
            fit_psd_95lo,
            fit_psd_95up,
            facecolor="red",
            zorder=1,
            alpha=0.25,
        )
        # ax.fill_between(
        #     f,
        #     fit_psd_68lo,
        #     fit_psd_68up,
        #     facecolor="red",
        #     zorder=1,
        #     alpha=0.25,
        # )

        # ax.plot(f, full_psd_mean, color="blue", zorder=2, alpha=0.5)
        # ax.fill_between(
        #     f,
        #     full_psd_95lo,
        #     full_psd_95up,
        #     facecolor="blue",
        #     zorder=1,
        #     alpha=0.25,
        # )
        # ax.fill_between(
        #     f,
        #     full_psd_68lo,
        #     full_psd_68up,
        #     facecolor="blue",
        #     zorder=1,
        #     alpha=0.25,
        # )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(bottom=np.min(obs_psd_mean) / 10, top=np.max(fit_psd_95up) * 10)
        ax.set_xlim(f.min(), f.max())
        ax.axvline(
            (self.psr.freqs.min() * u.d**-1).to_value(freq_units),
            ls="--",
            color="black",
        )
        ax.axvline((1.0 / u.yr).to_value(freq_units), ls=":", color="black")
        ax.axvline(
            (self.psr.freqs.max() * u.d**-1).to_value(freq_units),
            ls="-.",
            color="black",
        )
        ax.set_xlabel("Frequency (1/yr)")
        ax.set_ylabel("Power spectral density ($s^2 yr$)")

    def plot_psd_posterior(
        self, ax=None, kind="TN", nit=1000, xunits="yr ** -1", yunits="s ** 2 * yr"
    ):

        if kind == "TN":
            wxf = self.psr.WXfreqs.to_value("1/d")
            f = np.arange(0.25 * wxf.min(), wxf.max() * 3.0, 0.25 * wxf.min()) * u.d**-1

            powers = (
                self.phys_timing_chain[:, self.psr.WXSinds] ** 2
                + self.phys_timing_chain[:, self.psr.WXCinds] ** 2
            ) * u.s**2

            psd_f = self.psr.WXfreqs
            psd = (0.5 * powers / (psd_f[1] - psd_f[0])).to_value(yunits)

            # Pick the last point in the chain as a reproducible,
            # but "random" sample from which to estimate the WN level
            s = -1
            theta = self.timing_chain[s]
            if self.psr.has_OPV:
                theta = np.append(theta, self.opv_amps_chain[s])
            phase_shifts = self.psr.M @ theta
            phases = self.psr.phi + phase_shifts - np.mean(phase_shifts)

            phi = np.arange(-0.005, 0.005, 0.00001)

            logL = np.zeros_like(phi)

            for idx in range(len(phi)):
                shifted_phases = np.mod(phases + phi[idx], 1.0)

                logL[idx] = np.sum(
                    np.log(
                        self.psr.w * self.psr.tau_sampler.template(self.template_chain[s,:],shifted_phases,self.psr.log10E)
                        + (1 - self.psr.w)
                    )
                )
                if idx % 10 == 0:
                    print(idx, "/", len(phi), end = "\r")
            pdf = np.exp(logL - logL.max())
            pdf /= np.trapezoid(pdf, phi)
            dphi_sq = (
                np.trapezoid(phi**2 * pdf, phi) - np.trapezoid(phi * pdf, phi) ** 2
            )

            W = 2 * dphi_sq * self.psr.Tobs * u.d / self.psr.timing_model.F0.quantity**2

        elif kind == "OPV":
            opvf = self.psr.OPVfreqs.to_value("1/d")
            f = (
                np.arange(0.5 * opvf.min(), opvf.max() * 1.5, 0.5 * opvf.min())
                * u.d**-1
            )

            OPV_amps = (
                self.opv_amps_chain - self.psr.theta_prior[self.psr.n_timing_pars :]
            ) / self.parameter_scales[self.psr.n_timing_pars :]

            powers = (OPV_amps[:, ::2] ** 2 + OPV_amps[:, 1::2] ** 2) * self.psr.PB0**2

            psd_f = self.psr.OPVfreqs
            psd = (0.5 * powers / (psd_f[1] - psd_f[0])).to_value(yunits)

            # Pick the last point in the chain as a reproducible,
            # but "random" sample from which to estimate the WN level
            s = -1
            phase_shifts = self.psr.M @ (
                np.append(self.timing_chain[s], self.opv_amps_chain[s])
            )

            phases = self.psr.phi + phase_shifts - np.mean(phase_shifts)

            gaussians = []
            for p in range(self.psr.npeaks):
                gaussians.append(
                    LCGaussian(
                        p=[
                            self.template_chain[s, 2 * self.psr.npeaks + p],
                            np.mod(self.template_chain[s, self.psr.npeaks + p], 1.0),
                        ]
                    )
                )

            template = LCTemplate(
                gaussians, norms=self.template_chain[s, : self.psr.npeaks]
            )

            TASC_ind = np.where(self.parameter_names == "TASC")[0][0]
            dTASC = np.arange(-1.0, 1.0, 0.01) / 86400.0

            shifted_phases = np.mod(
                phases[:, None]
                + (
                    self.psr.M[:, TASC_ind, None]
                    * dTASC[None, :]
                    * self.parameter_scales[TASC_ind]
                ),
                1.0,
            )

            logL = np.sum(
                np.log(
                    self.psr.w[:, None] * template(shifted_phases)
                    + (1 - self.psr.w[:, None])
                ),
                axis=0,
            )

            pdf = np.exp(logL - logL.max())
            pdf /= np.trapezoid(pdf, dTASC)
            dTASC_sq = (
                np.trapezoid(dTASC**2 * pdf, dTASC)
                - np.trapezoid(dTASC * pdf, dTASC) ** 2
            )

            W = 2 * (dTASC_sq * u.d**2) * self.psr.Tobs * u.d

        if nit > len(self.hyp_chain):
            samples = np.arange(len(self.hyp_chain))
            nit = len(self.hyp_chain)
        else:
            samples = np.random.choice(
                np.arange(np.shape(self.hyp_chain)[0]),
                replace=True,
                size=nit,
            )

        fit_psd = np.zeros((len(f), nit))

        nc = 0
        for component in self.psr.noise_models:
            if component.prefix[: len(kind)] == kind:
                nc += 1

        component_psds = np.zeros((len(f), nit, nc))

        for c, i in enumerate(samples):

            print(c, end="\r")
            s = 0
            model_powspec = np.zeros(len(f))
            t = 0
            for j, component in enumerate(self.psr.noise_models):
                n = component.nfree
                p = component.all_parameters(self.hyp_chain[i, s : s + n])

                if component.prefix[: len(kind)] == kind:
                    component_psds[:, c, t] = component.powspec(p, f) * (
                        1.0 * u.s**2 * u.d
                    ).to_value(yunits)
                    model_powspec += component_psds[:, c, t]
                    t += 1

                s += n

            # amps = norm.rvs(size=2 * len(f)).reshape(len(f),2) * np.sqrt(model_powspec)[:,None]
            fit_psd[:, c] = model_powspec  # 0.5 * np.sum(amps ** 2,axis=1)

        if ax is None:
            fig, ax = plt.subplots(1, 1)

        ax.fill_between(
            f.to_value(xunits),
            np.quantile(fit_psd, norm.cdf(-2.0), axis=1),
            np.quantile(fit_psd, norm.cdf(2.0), axis=1),
            facecolor="red",
            alpha=0.33,
            zorder=0,
        )

        ax.fill_between(
            f.to_value(xunits),
            np.quantile(fit_psd, norm.cdf(-1.0), axis=1),
            np.quantile(fit_psd, norm.cdf(1.0), axis=1),
            facecolor="red",
            alpha=0.33,
            zorder=0,
        )

        if nc > 1:
            for j in range(nc):
                ax.plot(
                    f.to_value(xunits),
                    np.mean(component_psds[:, :, j], axis=1),
                    color="C0",
                    ls="-.",
                    zorder=1,
                )

        ax.plot(f.to_value(xunits), np.median(fit_psd, axis=1), color="red", zorder=0)

        med_psd = np.median(psd, axis=0)
        psd_err_lo = med_psd - np.quantile(psd, norm.cdf(-1.0), axis=0)
        psd_err_hi = np.quantile(psd, norm.cdf(1.0), axis=0) - med_psd

        ax.errorbar(
            psd_f.to_value(xunits),
            med_psd,
            yerr=(psd_err_lo, psd_err_hi),
            color="black",
            linestyle="none",
            zorder=2,
        )
        # psd_err_lo = med_psd - np.quantile(psd, norm.cdf(-1.0), axis=0)
        # psd_err_hi = np.quantile(psd, norm.cdf(1.0), axis=0) - med_psd

        # ax.errorbar(psd_f.to_value(xunits),
        #             med_psd,
        #             yerr = (psd_err_lo, psd_err_hi),
        #             color='black',linestyle='none',
        #             marker='x',
        #              )

        ax.axhline(W.to_value(yunits), color="black", ls="--")

        if kind == "TN":
            obs_psd_str = "obs_spin_psd"

        elif kind == "OPV":
            obs_psd_str = "obs_opv_psd"

        if hasattr(self, obs_psd_str):
            psd_units = 1.0 * u.s**2 * u.d
            obs_psd = (getattr(self, obs_psd_str) * psd_units).to_value(yunits)

            ax.fill_between(
                f.to_value(xunits),
                np.quantile(obs_psd, norm.cdf(-2.0), axis=1),
                np.quantile(obs_psd, norm.cdf(2.0), axis=1),
                facecolor="black",
                alpha=0.2,
            )
            ax.fill_between(
                f.to_value(xunits),
                np.quantile(obs_psd, norm.cdf(-1.0), axis=1),
                np.quantile(obs_psd, norm.cdf(1.0), axis=1),
                facecolor="black",
                alpha=0.2,
            )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(
            (psd_f.min() / 2.0).to_value(xunits), (psd_f.max() * 2.0).to_value(xunits)
        )
        ax.set_ylim(bottom=W.to_value(yunits) / 100.0)

        xunit_str = xunits.replace(" ** ", "^").replace("**", "^").replace("*", " ")
        if "-" in xunit_str:
            xunit_str = xunit_str.replace("-", "{-") + "}"

        ax.set_xlabel(f"Frequency (${xunit_str}$)")
        ax.set_ylabel(
            f"Power spectral density (${yunits.replace(' ** ','^').replace(' * ',' ')}$)"
        )

        return ax

    def summary_plot(
        self,
        tasc_colour="black",
        tasc_offset=0,
        tasc_slwin=None,
        dtasc_l=0,
        dtasc_r=0,
        xbins=50,
        tn_xunits="1/yr",
        opv_xunits="1/yr",
        tn_yunits="s ** 2 * yr",
        opv_yunits="s ** 2 * yr",
    ):

        ncol = 1
        if self.psr.fit_TN:
            ncol += 1
        if self.psr.fit_OPV:
            ncol += 1

        fig, ax = plt.subplots(
            2,
            ncol,
            height_ratios=[1, 2],
            figsize=(8 * ncol, 12),
            gridspec_kw={"wspace": 0.25, "hspace": 0.05},
        )

        ax = np.reshape(ax, (2, -1))

        ax[0, 0].sharex(ax[1, 0])
        if self.psr.fit_TN or self.psr.fit_OPV:
            ax[1, 0].sharey(ax[1, 1])
            ax[1, 1].set_ylabel("Time (MJD)")
        if self.psr.fit_TN and self.psr.fit_OPV:
            ax[1, 1].sharey(ax[1, 2])
            ax[1, 2].set_ylabel("Time (MJD)")

        if not hasattr(self, "phi_MAP"):
            self.MAP_phases()

        fermi_lc(self.phi_MAP, self.psr.w, ax=ax[0, 0], xbins=xbins)
        self.plot_templates(ax[0, 0], xbins=xbins)
        self.scatter_phases(self.phi_MAP, ax[1, 0])

        c = 1
        if self.psr.fit_TN:
            self.plot_psd_posterior(
                ax=ax[0, 1], xunits=tn_xunits, yunits=tn_yunits, kind="TN"
            )
            self.plot_phase_shifts(ax[1, 1])
            ax[0, 1].xaxis.set_label_position("top")
            ax[0, 1].xaxis.tick_top()
            c += 1
        if self.psr.fit_OPV:
            self.plot_psd_posterior(
                ax=ax[0, c], xunits=opv_xunits, yunits=opv_yunits, kind="OPV"
            )
            self.plot_tasc_shifts(ax[1, c])
            ax[0, c].xaxis.set_label_position("top")
            ax[0, c].xaxis.tick_top()

        return ax

    def plot_Edep_profiles(self, xbins=100):

        Ebounds = np.log10(np.array([100, 300, 1000, 3000, 10000]))
        Eavg = np.average(self.psr.log10E, weights=self.psr.w)

        phi = np.arange(0.0, 1.0, 0.001)
        fig, ax = plt.subplots(len(Ebounds) - 1, 1, figsize=(8, 12), sharex=True)
        for E in range(len(Ebounds) - 1):
            mask = np.where(
                (self.psr.log10E > Ebounds[E]) & (self.psr.log10E < Ebounds[E + 1])
            )
            fermi_lc(
                self.phi_MAP[mask],
                self.psr.w[mask],
                xbins=xbins,
                ax=ax[len(Ebounds) - 2 - E],
                bgcolor="C0",
            )
            src = np.sum(self.psr.w[mask] ** 2) / xbins
            bkg = np.sum(self.psr.w[mask] - self.psr.w[mask] ** 2) / xbins

            for i in np.random.choice(
                np.arange(np.shape(self.template_chain)[0]), replace=True, size=100
            ):
                prof = bkg + src * self.psr.tau_sampler.template(
                    self.template_chain[i],
                    phi,
                    np.ones_like(phi) * 0.5 * (Ebounds[E] + Ebounds[E + 1]),
                )
                ax[len(Ebounds) - 2 - E].plot(phi, prof, color="black", alpha=0.1)

            prof = bkg + src * self.psr.tau_sampler.template(
                self.template_MAP,
                phi,
                np.ones_like(phi) * 0.5 * (Ebounds[E] + Ebounds[E + 1]),
            )
            ax[len(Ebounds) - 2 - E].plot(
                phi,
                prof,
                color="orange",
                alpha=1.0,
            )
            ax[len(Ebounds) - 2 - E].set_title(
                f"${10 ** (Ebounds[E] - 3):.2g}\\,{{\\rm GeV}} < E < {10 ** (Ebounds[E+1] - 3):.2g}\\,{{\\rm GeV}}$"
            )

            avgprof = bkg + src * self.psr.tau_sampler.template(
                self.template_MAP, phi, np.ones_like(phi) * Eavg
            )
            ax[len(Ebounds) - 2 - E].plot(phi, avgprof, color="red", ls="--", alpha=1.0)

        ax[-1].set_xlim(0, 2)
        ax[-1].xaxis.set_visible(True)
        ax[-1].set_xlabel("Phase")

        return fig, ax

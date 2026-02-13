import sys
from optparse import OptionParser
import glob
import matplotlib.pyplot as plt

from astropy.io import fits

from shoogle.gibbs_sampler import Gibbs
from shoogle.plot_gibbs_results import GibbsResults, fermi_lc


def main(argv=None):

    desc = "Fit a Gaussian process orbital phase model to pulsar data"
    parser = OptionParser(usage=" %prog options", description=desc)
    parser.add_option("--ft1", type="string", default=None, help="Weighted FT1 file")
    parser.add_option(
        "--ft2",
        type="string",
        default=None,
        help="FT2 file for interpolating spacecraft positions",
    )
    parser.add_option(
        "-W",
        "--weightfield",
        type="string",
        default=None,
        help="Column name in FT1 file for photon weights (optional, default is to guess the column)",
    )
    parser.add_option(
        "-p",
        "--parfile",
        type="string",
        default=None,
        help="Pulsar ephemeris .par file",
    )
    parser.add_option(
        "-P",
        "--priorfile",
        type="string",
        default=None,
        help="File defining timing model priors and noise hyperparameters",
    )
    parser.add_option(
        "-t", "--template", type="string", default=None, help="Template pulse profile"
    )
    parser.add_option(
        "-r",
        "--radio_toas",
        type="string",
        default=None,
        help=".tim file with radio TOAs",
    )
    parser.add_option(
        "-c",
        "--weightcut",
        type="float",
        default=0.00,
        help="Minimum photon probability weight",
    )
    parser.add_option(
        "-d",
        "--decimated",
        action="store_true",
        default=False,
        help="Chain is already decimated",
    )
    parser.add_option(
        "-o",
        "--output",
        type="string",
        default=None,
        help="Base filename for saving plots",
    )
    parser.add_option(
        "-E",
        "--Edep",
        action="store_true",
        default=False,
        help="Fit for energy-dependence in the template pulse profile",
    )
    parser.add_option(
        "-q",
        "--quiet",
        action="store_true",
        default=False,
        help="Quiet mode, only save plots, don't show them",
    )

    (options, args) = parser.parse_args(argv)

    if options.weightfield is None:
        weightfield = None
        f = fits.open(options.ft1)
        colnames = f[1].data.names

        for c in colnames:
            w = None
            if c == "WEIGHT":
                w = "WEIGHT"
            elif c == "MODEL_WEIGHT":
                w = "MODEL_WEIGHT"
            elif c[:4] == "4FGL":
                w = c

            if w is not None:
                if weightfield is not None:
                    raise ValueError("Cannot unambiguously determine the weight column")
                weightfield = w
    else:
        weightfield = options.weightfield

    psr = Gibbs(
        parfile=options.parfile,
        priorfile=options.priorfile,
        ft1file=options.ft1,
        ft2file=options.ft2,
        timfile=options.radio_toas,
        weightfield=weightfield,
        templatefile=options.template,
        wmin=options.weightcut,
        Edep=options.Edep,
    )

    res = GibbsResults(psr)
    res.load_results(load_from=[options.output + ".npz"], decimated=True)
    res.MAP_phases()

    fig1, ax = plt.subplots(2, 1, height_ratios=[1, 2], figsize=(7, 12))
    fermi_lc(res.phi_MAP, res.psr.w, ax=ax[0], xbins=50)
    res.plot_templates(ax[0])
    res.scatter_phases(res.phi_MAP, ax[1])
    plt.savefig(options.output + "_phases.pdf", bbox_inches="tight")

    if hasattr(res.psr, "radio_toas"):
        fig2, ax2 = plt.subplots(2, 2, sharex="col")
        res.plot_radio_resids(ax2)
        plt.savefig(options.output + "_radio.pdf", bbox_inches="tight")

    if options.output:
        res.write_tvsph(options.output + "_MAP.tvsph", res.phi_MAP)
        res.write_new_template(options.output + "_prof.dat")
        if res.psr.has_OPV:
            res.write_new_parfile(options.output + "_orbwaves.par")
            res.write_orbifunc_parfile(options.output + "_orbifunc.par")
        else:
            res.write_new_parfile(options.output + ".par")

    if options.Edep:
        fig2, ax = res.plot_Edep_profiles()
        plt.savefig(options.output + "_Edep_profiles.pdf", bbox_inches="tight")

    if res.psr.fit_TN or res.psr.fit_OPV:
        fig3 = res.hyp_corner()
        if options.output:
            plt.savefig(options.output + "_hyperparameters.pdf", bbox_inches="tight")

    fig4 = res.timing_corner(nWX=psr.nWXfreqs)
    if options.output:
        plt.savefig(options.output + "_timingparameters.pdf", bbox_inches="tight")

    fig5 = res.template_corner()
    if options.output:
        plt.savefig(options.output + "_templateparameters.pdf", bbox_inches="tight")

    if res.psr.fit_TN or res.psr.fit_OPV:
        fig6 = res.summary_plot()
        if options.output:
            plt.savefig(options.output + "_summary.pdf", bbox_inches="tight")

    if not options.quiet:
        plt.show()

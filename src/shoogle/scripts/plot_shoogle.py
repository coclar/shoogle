import sys
from optparse import OptionParser
import glob
import matplotlib.pyplot as plt

from shoogle.gibbs_sampler import Gibbs
from shoogle.plot_gibbs_results import GibbsResults, fermi_lc
from shoogle.utils import *


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
        default="WEIGHT",
        help="Column name in FT1 file for photon weights",
    )
    parser.add_option(
        "-p",
        "--parfile",
        type="string",
        default=None,
        help="Pulsar ephemeris .par file",
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

    (options, args) = parser.parse_args(argv)

    psr = Gibbs(
        parfile=options.parfile,
        ft1file=options.ft1,
        ft2file=options.ft2,
        timfile=options.radio_toas,
        weightfield=options.weightfield,
        templatefile=options.template,
        wmin=options.weightcut,
    )

    res = GibbsResults(psr)
    res.load_results(load_from=[options.output + ".npz"], decimated=True)
    res.MAP_phases()

    fig, ax = plt.subplots(2, 1, height_ratios=[1, 2], figsize=(7, 12))
    fermi_lc(res.phi_MAP, res.psr.w, ax=ax[0], xbins=50)
    res.plot_templates(ax[0])
    res.scatter_phases(res.phi_MAP, ax[1])

    if hasattr(res.psr, "radio_toas"):
        fig2, ax2 = plt.subplots(2, 2, sharex="col")
        res.plot_radio_resids(ax2)
    plt.show()

    if options.output:
        res.write_tvsph(options.output + "_MAP.tvsph", res.phi_MAP)
        res.write_new_template(options.output + "_prof.dat")
        if res.psr.has_OPV:
            res.write_new_parfile(options.output + "_orbwaves.par")
            res.write_orbifunc_parfile(options.output + "_orbifunc.par")
        else:
            res.write_new_parfile(options.output + ".par")

    if res.psr.fit_TN and res.psr.has_OPV:
        fig1 = res.hyp_corner()
        if options.output:
            plt.savefig(options.output + "_hyperparameters.pdf", bbox_inches="tight")
        plt.show()

    fig2 = res.timing_corner(nWX=psr.nWXfreqs)
    if options.output:
        plt.savefig(options.output + "_timingparameters.pdf", bbox_inches="tight")
    plt.show()

    fig3 = res.template_corner()
    if options.output:
        plt.savefig(options.output + "_templateparameters.pdf", bbox_inches="tight")
    plt.show()

    res.summary_plot()
    if options.output:
        plt.savefig(options.output + "_summary.pdf", bbox_inches="tight")
    plt.show()

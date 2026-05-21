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
        "--outputfile",
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

    G = Gibbs(
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

    res = GibbsResults(G)
    res.load_results(load_from=[options.outputfile + ".npz"], decimated=True)

    res.write_tvsph(options.outputfile + "_MAP.tvsph", res.phi_MAP)
    res.write_new_template(options.outputfile + "_prof.dat")
    if res.psr.has_OPV:
        res.write_new_parfile(options.outputfile + "_orbwaves.par")
        res.write_orbifunc_parfile(options.outputfile + "_orbifunc.par")
    else:
        res.write_new_parfile(options.outputfile + ".par")

    endfile = ".png"
    if G.fit_TN or G.fit_OPV:
        fig1 = res.hyp_corner()
        plt.savefig(
            options.outputfile + "_hyperparameters" + endfile, bbox_inches="tight"
        )

    if res.psr.has_TN:
        fig2 = res.timing_corner(nWX=G.nWXfreqs)
    else:
        fig2 = res.timing_corner(plot_wxcomp=True)
    plt.savefig(options.outputfile + "_timingparameters" + endfile, bbox_inches="tight")

    fig3 = res.template_corner()
    plt.savefig(
        options.outputfile + "_templateparameters" + endfile, bbox_inches="tight"
    )

    fig4 = res.summary_plot()
    plt.savefig(options.outputfile + "_summary" + endfile, bbox_inches="tight")

    if options.Edep:
        fig5, ax = res.plot_Edep_profiles()
        plt.savefig(options.outputfile + "_Edep_prof" + endfile, bbox_inches="tight")

    if not options.quiet:
        plt.show()

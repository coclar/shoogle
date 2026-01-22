import sys
from optparse import OptionParser
import matplotlib.pyplot as plt

from shoogle.gibbs_sampler import Gibbs
from shoogle.plot_gibbs_results import GibbsResults


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
        "-r",
        "--radio_toas",
        type="string",
        default=None,
        help=".tim file with radio TOAs",
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
        "-c",
        "--weightcut",
        type="float",
        default=0.00,
        help="Minimum photon probability weight",
    )
    parser.add_option(
        "-o",
        "--outputfile",
        type="string",
        default=None,
        help="Base name for file to store the chains in",
    )
    parser.add_option(
        "-q",
        "--quiet",
        action="store_true",
        default=False,
        help="Quiet mode, turns plots and some terminal output off",
    )
    parser.add_option(
        "-n",
        "--nacor",
        type=int,
        default=10,  # 10000
        help="Number of autocorrelation times desired in chain",
    )
    parser.add_option(
        "-u",
        "--update",
        type=int,
        default=100,
        help="Number of steps in between updates of chain files and intermediate plots",
    )
    parser.add_option(
        "-N",
        "--max_iterations",
        type=int,
        default=1000000,
        help="Maximum number of Gibbs sampling steps to take before exiting.",
    )
    parser.add_option(
        "-E",
        "--Edep",
        action="store_true",
        default=False,
        help="Fit for energy-dependence in the template pulse profile",
    )
    (options, args) = parser.parse_args(argv)

    if options.outputfile is None:
        print("Error: -o/--outputfile is a required argument", file=sys.stderr)
        sys.exit(1)

    G = Gibbs(
        parfile=options.parfile,
        ft1file=options.ft1,
        ft2file=options.ft2,
        timfile=options.radio_toas,
        weightfield=options.weightfield,
        templatefile=options.template,
        wmin=options.weightcut,
        Edep=options.Edep,
    )

    G.sample(
        outputfile=options.outputfile,
        plots=(not options.quiet),
        max_iterations=options.max_iterations,
        n_acor_target=options.nacor,
        update=options.update,
    )

    if not options.quiet:
        res = GibbsResults(G)
        res.load_results(load_from=[options.outputfile + ".npz"], decimated=True)
        res.MAP_phases()
        res.maxlogL_phases()

        res.write_tvsph(options.outputfile + "_MAP.tvsph", res.phi_MAP)
        res.write_tvsph(options.outputfile + "_maxlogL.tvsph", res.phi_maxlogL)
        res.write_new_template(options.outputfile + "_prof.dat")
        if res.psr.has_OPV:
            res.write_new_parfile(options.outputfile + "_orbwaves.par")
            res.write_orbifunc_parfile(options.outputfile + "_orbifunc.par")
        else:
            res.write_new_parfile(options.outputfile + ".par")

        endfile = ".png"
        if G.fit_TN:
            fig1 = res.hyp_corner()
            plt.savefig(
                options.outputfile + "_hyperparameters" + endfile, bbox_inches="tight"
            )

        if res.psr.has_TN:
            fig2 = res.timing_corner(nWX=G.nWXfreqs)
        else:
            fig2 = res.timing_corner(plot_wxcomp=True)
        plt.savefig(
            options.outputfile + "_timingparameters" + endfile, bbox_inches="tight"
        )

        fig3 = res.template_corner()
        plt.savefig(
            options.outputfile + "_templateparameters" + endfile, bbox_inches="tight"
        )

        fig4 = res.summary_plot()
        plt.savefig(options.outputfile + "_summary" + endfile, bbox_inches="tight")

        plt.show()

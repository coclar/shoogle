from pint.models import get_model
from pint.fermi_toas import get_Fermi_TOAs
from pint.observatory.satellite_obs import SatelliteObs
from pint.templates.lctemplate import LCTemplate, prim_io
from pint.residuals import Residuals
from pint import toa
from pint import logging
from astropy import units as u
from astropy.time import Time
from matplotlib.gridspec import GridSpec

import sys
import numpy as np
import matplotlib.pyplot as plt

from optparse import OptionParser

from shoogle.gibbs_sampler import read_input_ft1_file

logging.setup(level="DEBUG")

desc = "Fold fermi photons using pint, output an ascii file with time|phase|weight|dphi_dtasc columns"
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
    help="Column name in FT1 file for photon weights",
)
parser.add_option(
    "-p", "--parfile", type="string", default=None, help="Pulsar ephemeris .par file"
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
    "-O",
    "--orbphase",
    action="store_true",
    default=False,
    help="Also calculate and store orbital phase in tvsph file.",
)
parser.add_option(
    "-E",
    "--energy",
    action="store_true",
    default=False,
    help="Also store photon energies in tvsph file.",
)

(options, args) = parser.parse_args()

parfile = options.parfile
ft1file = options.ft1
ft2file = options.ft2
wmin = options.weightcut

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

timing_model = get_model(parfile)
fermi_toas, weights, energies = read_input_ft1_file(
    ft1file, ft2file, weightfield, wmin, timing_model
)

fermi_phases = np.mod(timing_model.phase(fermi_toas).frac, 1.0)
tssb = timing_model.get_barycentric_toas(fermi_toas).value

output = np.array([tssb, fermi_phases, weights]).T

if options.orbphase:
    orbphase = timing_model.orbital_phase(fermi_toas, anom="mean", radians=False)
    output = np.append(output, orbphase[:, None], axis=1)

if options.energy:
    output = np.append(output, energies[:, None], axis=1)

np.savetxt(options.outputfile + ".tvsph", output)

from pint.models import get_model
from pint.fermi_toas import get_Fermi_TOAs
from pint.observatory.satellite_obs import SatelliteObs
from pint.templates.lctemplate import LCTemplate, prim_io
from pint.residuals import Residuals
from pint import toa
from astropy import units as u
from astropy.time import Time
from matplotlib.gridspec import GridSpec

import sys
import numpy as np
import matplotlib.pyplot as plt

from optparse import OptionParser


def read_input_ft1_file(infile, FT2, weightfield, wmin):

    try:
        SatelliteObs(name="Fermi", ft2name=FT2)
    except ValueError:
        pass

    toas = get_Fermi_TOAs(
        infile,
        weightcolumn=weightfield,
        minweight=wmin,
        include_bipm=False,
        planets=True,
    )

    # toas = toa.get_TOAs_list(tl,include_bipm=False,planets=True)
    weights = np.array([float(w["weight"]) for w in toas.table["flags"]])
    energies = np.array([float(w["energy"]) for w in toas.table["flags"]])
    return toas, weights, energies


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
    default="MODEL_WEIGHT",
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
weightfield = options.weightfield
wmin = options.weightcut

timing_model = get_model(parfile)
fermi_toas, weights, energies = read_input_ft1_file(ft1file, ft2file, weightfield, wmin)

fermi_phases = np.mod(timing_model.phase(fermi_toas).frac, 1.0)
tssb = timing_model.get_barycentric_toas(fermi_toas).value

output = np.array([tssb, fermi_phases, weights]).T

if options.orbphase:
    orbphase = timing_model.orbital_phase(fermi_toas, anom="mean", radians=False)
    output = np.append(output, orbphase[:, None], axis=1)

if options.energy:
    output = np.append(output, energies[:, None], axis=1)

np.savetxt(options.outputfile + ".tvsph", output)

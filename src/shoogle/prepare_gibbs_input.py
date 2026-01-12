from astropy.io import fits
import numpy as np
import sys, os, re
from pint.models import get_model
from pint.utils import wavex_setup
from astropy import units as u
from pint.models.parameter import floatParameter


def combine_ft1s(input_dir, out_dir):
    """
    Given a list of weekly photon observation files, this function combines
    all the information in a single ft1 file saved in out_dir

    Input:
    - input_dir : input directory (.txt file containing the list of .fits files to combine)
    - out_dir : output directory (.fits file)

    """

    with open(input_dir, "r") as dat:
        datafiles = [l.strip() for l in dat.readlines()]
        dat.close()

    fit = fits.open(datafiles[0])
    data_dtype = fit[1].data.dtype
    gti_dtype = fit[2].data.dtype
    fit.close()

    nd = 0
    ng = 0

    tstart = None
    tstop = None

    rad = None
    emin = None
    emax = None
    zmax = None

    for f in datafiles:
        print(f, end="\r")

        fit = fits.open(f)
        nd += len(fit[1].data)
        ng += len(fit[2].data)

        tstart_tmp = fit[1].header["TSTART"]
        tstop_tmp = fit[1].header["TSTOP"]
        date_tmp = fit[1].header["DATE"]
        dstart_tmp = fit[1].header["DATE-OBS"]
        dend_tmp = fit[1].header["DATE-END"]

        i = 1
        while f"DSTYP{i}" in fit[1].header.keys():
            if fit[1].header[f"DSTYP{i}"] == "POS(RA,DEC)":
                rad_tmp = int(fit[1].header[f"DSVAL{i}"].split(",")[-1].rstrip(")"))
            elif fit[1].header[f"DSTYP{i}"] == "ENERGY":
                erange = fit[1].header[f"DSVAL{i}"].split(":")
                emin_tmp = float(erange[0])
                emax_tmp = float(erange[1])
            elif fit[1].header[f"DSTYP{i}"] == "ZENITH_ANGLE":
                zrange = fit[1].header[f"DSVAL{i}"].split(":")
                zmax_tmp = float(zrange[1])
            i += 1

        if tstart is None:
            tstart = tstart_tmp
            dstart = dstart_tmp
        if tstop is None:
            tstop = tstop_tmp
            dend = dend_tmp

        if rad is None:
            rad = rad_tmp
        if emin is None:
            emin = emin_tmp
        if emax is None:
            emax = emax_tmp
        if zmax is None:
            zmax = zmax_tmp

        if tstart_tmp < tstart:
            tstart = tstart_tmp
            dstart = dstart_tmp
        if tstop_tmp > tstop:
            tstop = tstop_tmp
            dend = dend_tmp
            date = date_tmp

        if rad_tmp > rad:
            rad = rad_tmp
        if emin_tmp < emin:
            emin = emin_tmp
        if emax_tmp > emax:
            emax = emax_tmp
        if zmax_tmp > zmax:
            zmax = zmax_tmp

        fit.close()

    data = np.zeros(nd, dtype=data_dtype)
    gti = np.zeros(ng, dtype=gti_dtype)

    ds = 0
    gs = 0
    ton = 0
    for i, fil in enumerate(datafiles):
        fit = fits.open(fil)
        d = len(fit[1].data)
        g = len(fit[2].data)

        data[ds : ds + d] = fit[1].data
        gti[gs : gs + g] = fit[2].data

        print(i, len(datafiles), fil, ds + d, end="\r")
        ds += d
        gs += g

        fit.close()

    idx = np.argsort(data["TIME"])
    data = data[idx]

    idx = np.argsort(gti["START"])
    gti = gti[idx]

    _, idx = np.unique(gti["START"], return_index=True)
    gti = gti[idx]

    fit = fits.open(datafiles[0])
    fit[1].data = data
    fit[1].header["TSTART"] = tstart
    fit[1].header["TSTOP"] = tstop
    fit[1].header["DATE-OBS"] = dstart
    fit[1].header["DATE-END"] = dend

    i = 1
    while f"DSTYP{i}" in fit[1].header.keys():
        if fit[1].header[f"DSTYP{i}"] == "POS(RA,DEC)":
            circle_filter_parts = fit[1].header[f"DSVAL{i}"].split(",")
            circle_filter_parts[-1] = str(rad) + ")"
            circle_str_new = ",".join(circle_filter_parts)
            fit[1].header[f"DSVAL{i}"] = circle_str_new
        elif fit[1].header[f"DSTYP{i}"] == "ENERGY":
            fit[1].header[f"DSVAL{i}"] = f"{emin}:{emax}"
        elif fit[1].header[f"DSTYP{i}"] == "ZENITH_ANGLE":
            fit[1].header[f"DSVAL{i}"] = f"0:{zmax}"
        i += 1

    fit[2].data = gti
    fit[2].header["TSTART"] = tstart
    fit[2].header["TSTOP"] = tstop
    fit[2].header["DATE-OBS"] = dstart
    fit[2].header["DATE-END"] = dend
    fit[2].header["TELAPSE"] = tstop - tstart

    fit.writeto(out_dir, overwrite=True)
    fit.close()


def write_input_parfile(
    par_input,
    par_output,
    redlogA=0.0,
    redlogfc=None,
    redgam=0.0,
    redkappa=None,
    redfit=1,
    redfitfc=0,
    nWX=None,
    tobs=None,
):
    """
    Given a pulsar parfile and the parameters for the shoogle analysis, this function
    write a new version of the parfile at outdir adding the required parameters (like WAVEX etc.)

    Input:
    - par_input : directory of the original .par file
    - par_output : directory of the new parfile to write
    optional:
    - redlogA : value for the red noise logA (default 0.)
    - redlogfc : value for the red noise frequency cutoff (default None)
    - redgam : value for the red noise gamma (default 0.)
    - redfit : 0/1 flag for the fit of the red noise hyperparameters (default 1)
    - redfit : 0/1 flag for the fit of the red noise cutoff frequency (default 0)
    - nWX : number of wavex components (default None)
    - tobs : predefined total observation time (for gwb runs), unit: days (default None)
    """

    # reading original timing model
    timing_model = get_model(par_input)

    # adding wavex components
    if nWX:
        if tobs:
            Tobs = tobs
        else:
            t1 = timing_model.START.value
            t2 = timing_model.FINISH.value
            Tobs = t2 - t1
        wx_inds = wavex_setup(timing_model, Tobs * u.d, n_freqs=nWX)

    # writing the new parfile
    with open(par_output, "w") as f:
        f.write(timing_model.as_parfile())

    # check for additional phase shifts to add
    extra_phase = None
    with open(par_input, "r") as f:
        lines = f.readlines()
        f.close()
    for l in lines:
        try:
            if l.split()[0] == "LATPHASE":
                extra_phase = l
        except IndexError:  # it accounts for blank lines in the par file
            continue

    # adding other hyperparams
    with open(par_output, "r") as f:
        lines = f.readlines()
        f.close()
    list_no_write = [
        "TN_REDAMP",
        "TN_REDFC",
        "TN_REDGAM",
        "WAVE_OM",
        "WAVEEPOCH",
        "TN_REDKAPPA",
    ]
    if redlogfc is None:
        redlogfc = -np.log10(Tobs * (u.d / u.yr)) - 0.8
    with open(par_output, "w") as f:
        for l in lines:
            if l.split()[0] in list_no_write:
                continue
            elif re.match(r"WAVE\d+", l.split()[0]):
                continue
            else:
                f.writelines(l)
        f.writelines(
            "TN_REDAMP \t\t " + str(redlogA) + " \t " + str(int(redfit)) + "\n"
        )
        f.writelines(
            "TN_REDFC \t\t " + str(redlogfc) + " \t " + str(int(redfitfc)) + "\n"
        )
        f.writelines("TN_REDGAM \t\t " + str(redgam) + " \t " + str(int(redfit)) + "\n")
        if redkappa:
            f.writelines(
                "TN_REDKAPPA \t\t " + str(redkappa) + " \t " + str(int(redfit)) + "\n"
            )
        if extra_phase:
            f.writelines(extra_phase)
        f.close()


############################################################################################

if __name__ == "__main__":

    from optparse import OptionParser

    desc = "Preparing the ft1s and .par files in the correct input format for Shoogle"
    parser = OptionParser(usage=" %prog options", description=desc)
    parser.add_option(
        "--homedir",
        type="string",
        default="./",
        help="home directory",
    )
    parser.add_option(
        "--outdir",
        type="string",
        default="./",
        help="output directory",
    )
    parser.add_option(
        "--psr",
        type="string",
        default=None,
        help="pulsar name",
    )
    parser.add_option(
        "--ft1",
        type="string",
        default=None,
        help="directory of the .txt file containing the list of ft1s files to merge",
    )
    parser.add_option(
        "--par",
        type="string",
        default=None,
        help="directory of the .par file to modify",
    )
    parser.add_option(
        "--fitRED",
        type="int",
        default=1,
        help="1 if you want to fit for the red noise hyperparameters, 0 if you want to keep them fixed",
    )
    parser.add_option(
        "--TN_REDAMP",
        type="float",
        default=-15.0,
        help="red noise log10_amplitude",
    )
    parser.add_option(
        "--TN_REDGAM",
        type="float",
        default=3.0,
        help="red noise gamma",
    )
    parser.add_option(
        "--TN_REDFC",
        type="float",
        default=None,
        help="red noise log10_frequency_cutoff",
    )
    parser.add_option(
        "--nWXfreqs",
        type="int",
        default=None,
        help="number of frequency components for red noise wavex frequency modeling",
    )
    parser.add_option(
        "--tobs",
        type=int,
        default=None,
        help="Possibility to set a defined Tobs, useful for GWB analysis (unit: days)",
    )
    (options, args) = parser.parse_args()

    # make the output directory (in case it does not exist already)
    os.system("mkdir " + options.homedir + options.outdir)

    # combining ft1 files
    if options.ft1:
        print("Starting to combine the ft1 files...")
        ft1dir = options.homedir + options.ft1
        ft1_outdir = (
            options.homedir + options.outdir + options.psr + "_combined_ft1s.fits"
        )
        combine_ft1s(ft1dir, ft1_outdir)
        print("Ft1s files combined. The final result can be found at " + ft1_outdir)

    # writing a modified .par file
    if options.par:
        print("Writing a new .par file...")
        par_input = options.homedir + options.par
        par_output = options.homedir + options.outdir + options.psr + "_shoogle.par"
        write_input_parfile(
            par_input,
            par_output,
            redlogA=options.TN_REDAMP,
            redlogfc=options.TN_REDFC,
            redgam=options.TN_REDGAM,
            redfit=options.fitRED,
            nWX=options.nWXfreqs,
            tobs=options.tobs,
        )
        print("The new .par file can be found at " + par_output)

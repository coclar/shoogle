import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy.interpolate import interp1d


def local_PB_TASC(t, orbifunc, PB0, TASC0, dt=1):

    # Get corrected orbital phases at epoch, and "dt" days later
    phi = (t - TASC0) / PB0 + orbifunc(t)
    phi_orb2 = (t + dt - TASC0) / PB0 + orbifunc(t + 1)

    # Work out new orbital frequency and period from these phases
    FB = (phi_orb2 - phi) / dt
    PB_new = 1.0 / FB

    # Find the closest TASC to epoch
    norb = np.round(phi)
    TASC_new = (TASC0 + norb * PB0) - orbifunc(t) * PB_new

    return PB_new, TASC_new


parfile = sys.argv[1]

with open(parfile, "r") as pf:
    parlines = pf.readlines()

orbifunc_t = []
orbifunc_val = []
for line in parlines:
    if len(line.split()) == 0:
        continue

    if line.split()[0] == "PB":
        PB0 = float(line.split()[1])
    elif line.split()[0] == "TASC":
        TASC0 = float(line.split()[1])
    elif line.split()[0][:8] == "ORBIFUNC":
        orbifunc_t.append(float(line.split()[1]))
        orbifunc_val.append(float(line.split()[2]))

orbifunc_t = np.array(orbifunc_t)
orbifunc_val = np.array(orbifunc_val)

orbifunc = interp1d(orbifunc_t, orbifunc_val)

if len(sys.argv) < 3:
    PB_new, TASC_new = local_PB_TASC(orbifunc_t[1:-1], orbifunc, PB0, TASC0)
    fig, ax = plt.subplots(1, 2, figsize=(10, 10), sharey=True)
    ax[0].plot((PB_new - PB0) / PB0, orbifunc_t[1:-1])
    ax[1].plot(TASC_new, orbifunc_t[1:-1])
    plt.show()

try:
    epochs = [float(f) for f in sys.argv[2:]]
    for epoch in epochs:
        PB_new, TASC_new = local_PB_TASC(epoch, orbifunc, PB0, TASC0)
        print(epoch, PB_new, TASC_new)
except:

    inffiles = sys.argv[2:]

    for inffile in inffiles:
        with open(inffile, "r") as inf:
            inflines = inf.readlines()

        for line in inflines:
            if "Epoch" in line:
                epoch = float(line.split()[-1])

        PB_new, TASC_new = local_PB_TASC(epoch, orbifunc, PB0, TASC0)

        newparfile = inffile.replace(".inf", ".par")

        # Write a new parfile
        with open(newparfile, "w") as npf:
            for line in parlines:
                if len(line.split()) == 0:
                    continue
                elif line.split()[0] == "PB":
                    print(f"PB           {PB_new:.15f}", file=npf)
                elif line.split()[0] == "TASC":
                    print(f"TASC         {TASC_new:.15f}", file=npf)
                elif "ORBIFUNC" not in line:
                    print(line, file=npf, end="")

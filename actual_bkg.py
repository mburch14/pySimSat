from xspec import *
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
import os
from pathlib import Path
from xspec import Xset
from scipy.stats import chi2

params = {
    "axes.labelsize": 15,
    "font.size": 15,
    "legend.fontsize": 15,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "font.family": "serif",
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "xtick.top": True,
    "ytick.right": True,
    "xtick.direction": "in",
    "ytick.direction": "in",
}
plt.rcParams.update(params)


pha = "/disk/bifrost/bifrost/CALDB/data/swift/bat/cpf/swbbkgspec20041120v001.pha"

with fits.open(pha) as hdul:
    spec = hdul["SPECTRUM"].data  # type: ignore
    ebounds = hdul["EBOUNDS"].data  # type: ignore
    emin = ebounds["E_MIN"]
    emax = ebounds["E_MAX"]
    rate = spec["RATE"]
    mask = (emin >= 15.0) & (emax <= 195.0)
    background_rate = np.sum(rate[mask])

print(f"BAT background rate (15–195 keV): {background_rate:.2f} counts/s")
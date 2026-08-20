from xspec import *
import matplotlib.pyplot as plt
import numpy as np
import subprocess
from astropy.io import fits
from pathlib import Path
from xspec import Xset

Xset.chatter = 5

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

def generate_background_spectrum(background, mission, sourcechars, spec_dir, resp_dir):
    ### Create a background ASCII file, convert it to a pyxspec model, and run fakeit on the model with the rsp file ###

    #remove existing background models if they exist.
    for file in ["bkg.mod", "bkg.mod.gz"]:
        path = Path(file)
        if path.exists():
            path.unlink()

    back_pha_path = spec_dir / "background.pha"
    back_dat_path = spec_dir / "background.dat"

    #remove existing background spectrum if it exists.
    if back_pha_path.exists():
        back_pha_path.unlink()
    if back_dat_path.exists():
        back_dat_path.unlink()

    backgroundname = background.gen_spectrum_table(output = spec_dir / 'background.dat', dcxr = sourcechars["dcxr"], albedo= sourcechars["albedo"], cralbedo= sourcechars["cralbedo"], cosmicrays = sourcechars["cosmicrays"])

    AllData.clear()
    AllModels.clear()

    #turns the ASCII file into an xspec model. 
    subprocess.run(["flx2tab", backgroundname, "bkg", "bkg.mod"], check = True)
    bkg_model = Model("atable{bkg.mod}")

    #run xspec on the model using the response files. 
    fake = FakeitSettings(response = str(resp_dir / f"{mission.name}.rsp"), exposure = str(sourcechars["exposure"]), fileName = str(spec_dir / "background.pha"))
    AllData.fakeit(1, fake)


def plot_background_spectrum(num_det_pixels, exposureTime, output_dir, spec_dir):
    ### Plot the fakeit spectrum generated in generate_background_spectrum ###
    
    AllData.clear()
    AllData(str(spec_dir / "background.pha"))
    # Energy bin edges
    spec = AllData(1)
    energies = np.array(spec.energies) #type: ignore
    elow = energies[:,0]
    ehigh = energies[:,1]
    dE = ehigh - elow
    energy = (elow + ehigh) / 2

    with fits.open(spec_dir / "background.pha") as hdul:
        pha = hdul["SPECTRUM"].data #type: ignore
        counts = np.array(pha["COUNTS"])
    countRate = ((counts / exposureTime) / dE) #puts the y-axis in the correct units.

    plt.figure(figsize=(8,5))
    plt.step(energy, countRate, where="mid", color="black")
    plt.xlabel("Energy (keV)")
    plt.ylabel("Count rate (counts/s/keV)")
    plt.yscale("log")
    plt.xscale('log')
    plt.title("Simulated background Spectrum")
    plt.savefig(output_dir / "sim_background_spectrum.png", dpi=300, bbox_inches="tight")
    plt.close()


def gen_background(background, mission, sourcechars, num_det_pixels, output_dir, spec_dir, resp_dir):
    generate_background_spectrum(background, mission, sourcechars, spec_dir, resp_dir)
    plot_background_spectrum(num_det_pixels, sourcechars["exposure"], output_dir, spec_dir)
from xspec import *
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
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


def generate_observation_spectrum(mission, sourcechars, spec_dir, resp_dir):

    obs_pha_path = spec_dir / "observation.pha"
    obs_bkg_pha_path = spec_dir / "observation_bkg.pha"

    #remove existing observation spectrum if it exists.
    if obs_pha_path.exists():
        obs_pha_path.unlink()
    if obs_bkg_pha_path.exists():
        obs_bkg_pha_path.unlink()

    AllData.clear()
    AllModels.clear()

    #this is the source that we are doing.
    sourcemodel = Model(sourcechars["sourceShape"])
    sourcemodel.powerlaw.PhoIndex = sourcechars["phoIndex"] #type: ignore
    sourcemodel.powerlaw.norm = sourcechars["normalization"] #type: ignore

    #run xspec on the model using the response files. 
    fake = FakeitSettings(response= str(resp_dir / f"{mission.name}.rsp"), exposure= str(sourcechars["exposure"]), fileName=str(spec_dir / "observation.pha"), background = str(spec_dir / "background.pha"))
    AllData.fakeit(1, fake)


def plot_observation_spectrum(sourcename, exposureTime, output_dir, spec_dir):
    AllData.clear()
    AllData(str(spec_dir / "observation.pha"))
    spec = AllData(1)

    # Energy bin edges
    energies = np.array(spec.energies) #type: ignore
    elow = energies[:,0]
    ehigh = energies[:,1]
    dE = ehigh - elow
    energy = (elow + ehigh) / 2

    with fits.open(spec_dir / "observation.pha") as hdul:
        pha = hdul["SPECTRUM"].data #type: ignore
        counts = np.array(pha["COUNTS"])
    countRate = ((counts / exposureTime) / dE) #puts the y-axis in the correct units.

    plt.figure(figsize=(8,5))
    plt.step(energy, countRate, where="mid", color="black")
    plt.xlabel("Energy (keV)")
    plt.ylabel("rate (counts/s/keV)")
    plt.yscale("log")
    plt.xscale('log')
    plt.title(f"Simulated {sourcename} Spectrum with Background")
    plt.savefig(output_dir / f"Sim_{sourcename}_spectrum.png", dpi=300, bbox_inches="tight")
    plt.close()
    

def calculate_snr(spec_dir, sourcechars, chars):
    obs = fits.getdata(spec_dir / "observation.pha", 1)["COUNTS"].sum() #type: ignore
    bkg = fits.getdata(spec_dir / "observation_bkg.pha", 1)["COUNTS"].sum() #type: ignore
    time = sourcechars["exposure"]
    detArea = chars['config']['detl'] * chars['config']['detw']
    f = chars['config']['maskOpen']
    source = (obs-bkg)/detArea
    background = bkg/detArea
    std = np.sqrt(((source + background)/(f*detArea)) + (background/((1-f)*detArea)))
    
    snr = source/std

    print(f"Source count rate: {(obs-bkg)/time:.2f}")
    print(f"Background count rate: {(bkg)/time:.2f}")
    print(f'Simulated SNR: {snr:.2f}')

    try:
        sourcects = sourcechars["sourceCounts"]
        bkgcts = sourcechars["backgroundCounts"]
        actsource = sourcects/detArea
        actbkg = bkgcts/detArea
        actstd = np.sqrt(((actsource + actbkg)/(f*detArea)) + (actbkg/((1-f)*detArea)))
        actsnr = actsource / actstd
        print(f'\nActual source count rate: {sourcects/time:.2f}')
        print(f'Actual Background count rate: {bkgcts/time:.2f}')
        print(f'Actual SNR: {actsnr:.2f}')
    except:
        pass


def gen_observation(mission, sourcechars, output_dir, spec_dir, resp_dir, chars):
    generate_observation_spectrum(mission, sourcechars, spec_dir, resp_dir)
    plot_observation_spectrum(sourcechars["name"], sourcechars["exposure"], output_dir, spec_dir)
    calculate_snr(spec_dir, sourcechars, chars)
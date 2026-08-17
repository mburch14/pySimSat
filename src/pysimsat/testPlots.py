import matplotlib.pyplot as plt
import numpy as np
from xspec import *
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from . import MissionClasses as mc
from pathlib import Path
import commentjson

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


#This is for if you need to get the classes for any of the plots.
def build_satellite(instrument, instrument_json, source, source_json):

    with open(instrument_json) as f:
        jsons = commentjson.load(f)
    chars = jsons[instrument]

    with open(source_json) as f:
        jsons = commentjson.load(f)
    sourcechars = jsons[source]

    #This is for our specific Cubesat
    orb = mc.Orbit(altitude = chars["altitude"], inclination = chars['inclination'])
    geo = mc.geometry( config = chars['config'])
    mission = mc.Mission(instrument, chars['e_min'], chars['e_max'])
    mask = mc.optics(thickness = chars['config']['maskh'], mask_material = chars['mask_material'], mask_density = chars['mask_material_density'], localized = sourcechars["localized"])
    detector = mc.detector(geometry = geo, orbit = orb, mission = mission, optics = mask, res = chars["spec_resolution"], grad = chars["spec_gradient"], low_ecut = chars["low_ecut"], material = chars["detector_material"], mat_density = chars["det_material_density"])
    background = mc.BackgroundModel(detector = detector)

    return orb, geo, mission, detector, background, mask


#Plot the different background components
def plot_background_components(mission, geo, background, output_dir):

    output_dir = Path(output_dir)    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    energy = np.linspace(mission.energymin, mission.energymax, 500)
    cxb = np.array([e*e*background.cxb(e, geo.fov_sr)/geo.fov_sr for e in energy])
    albedo = np.array([e*e*background.albedo(e, geo.fov_sr)/geo.fov_sr for e in energy])
    particles = np.array([e*e*background.particle(e, geo.fov_sr)/geo.fov_sr for e in energy])

    print(cxb[::5])
    print("\n")
    print(albedo[::5])
    print("\n")
    print(particles[::5])

    plt.figure(figsize=(8,5))
    plt.scatter(energy, cxb, color="black")
    plt.scatter(energy, albedo, color="red")
    plt.scatter(energy, particles, color="blue")
    plt.xlabel("Energy (keV)")
    plt.ylabel(r"Energy$^2$ $\times$ $\frac{dN}{dt}$   keV$^2$(photons cm$^{-2}$ s$^{-1}$ sr$^{-1}$ keV$^{-1}$)")
    plt.title("background Components")
    plt.savefig(output_dir / "sampleBKG.png", dpi=300, bbox_inches="tight")
    plt.close()


#See what is in your arf file
def plot_effective_area_arf(mission, spec_dir, output_dir):

    output_dir = Path(output_dir)
    spec_dir = Path(spec_dir)
        
    output_dir.mkdir(parents=True, exist_ok=True)
    spec_dir.mkdir(parents=True, exist_ok=True)

    with fits.open(spec_dir / f"{mission.name}.arf") as hdul:
        arf = hdul["SPECRESP"].data #type: ignore

    energy = (arf["ENERG_LO"] + arf["ENERG_HI"])/2
    area = arf["SPECRESP"]

    plt.plot(energy, area)
    plt.xlabel(r"Energy (keV)")
    plt.ylabel(r"Effective area (cm$^2$)")
    plt.yscale("log")
    plt.xscale("log")
    plt.savefig(output_dir / "ARFeffectiveArea.png", dpi=300, bbox_inches="tight")
    plt.show()


#Plot the effective area on a given range
def plot_effective_area(mission, detector, mask, output_dir, start, end, num):

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    energy_vals = np.linspace(start, end, num)

    effective_area = [detector.effective_area(energy=e) for e in energy_vals]
    mask_trans = [mask.transmission(energy=e) for e in energy_vals]
    det_abs = [detector.det_absorption(energy=e) for e in energy_vals]

    fig, ax1 = plt.subplots(figsize=(8, 5))

    # Left y-axis: Effective area
    ax1.plot(energy_vals, effective_area, color="black", linewidth=2, label="Effective Area")
    ax1.set_xlabel(r"Energy ($keV$)")
    ax1.set_ylabel(r"Effective Area ($cm^2$)", color="black")
    #ax1.set_ylim(0, 1500)
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.set_xscale("log")

    # Right y-axis: Transmission/Absorption
    ax2 = ax1.twinx()
    ax2.plot(energy_vals, det_abs * 100, color="tab:blue", linestyle="--", label="Detector Absorption")
    ax2.plot(energy_vals, mask_trans * 100, color="tab:red", linestyle="-.", label="Optics Transmission")
    ax2.set_ylabel(r"Efficiency (%)")
    ax2.set_ylim(0, 100)
    ax2.tick_params(axis='y')

    # Combine legends from both axes
    lines = ax1.get_lines() + ax2.get_lines()
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="best") #type: ignore

    plt.title(f"{mission.name} Effective Area and Efficiencies")
    plt.tight_layout()
    plt.savefig(output_dir / "TotalEffectiveArea.png", dpi=300, bbox_inches="tight")
    plt.show()

    
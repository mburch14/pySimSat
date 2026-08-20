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


def build_satellite(instrument, instrument_json, source, source_json):
### This is needed to create the objects for your satellite. Inputs for BAT (top) and CZTI (bottom) are given below. 
### Input that into your terminal, and then you will be able to work in testPlots. ###

    """orb, geo, mission, detector, background, mask = pysimsat.testPlots.build_satellite("SWIFTBAT", "example_sat_source/instrumentCharacteristics.json", "CRAB-SWIFTBAT", "example_sat_source/source.json")"""
    """orb, geo, mission, detector, background, mask = pysimsat.testPlots.build_satellite("ASTROSAT", "example_sat_source/instrumentCharacteristics.json", "CRAB-ASTROSAT", "example_sat_source/source.json")"""


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
    detector = mc.detector(geometry = geo, orbit = orb, mission = mission, optics = mask, res = chars["spec_resolution"], grad = chars["spec_gradient"], low_ecut = chars["low_ecut"], material = chars["detector_material"], mat_density = chars["det_material_density"], activedetector=chars["activedetector"])
    background = mc.BackgroundModel(detector = detector, orbit=orb)

    return orb, geo, mission, detector, background, mask


def plot_background_components(mission, geo, background, output_dir):
### This is to plot the different components of the background (dcrb, albedo, cr-albedo, cr) individually ###
### example terminal input given below ###

    """pysimsat.testPlots.plot_background_components(mission, geo, background, "outputs")"""
    
    output_dir = Path(output_dir)    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    energy = np.linspace(mission.energymin, mission.energymax, 500)
    cxb = np.array([background.dcxr(e, geo.fov_sr) for e in energy])
    albedo = np.array([background.albedo(e, geo.fov_sr) for e in energy])
    particles = 0.33 * np.array([background.dcxr(e, 0.67) for e in energy])
    cralbedo = np.array([background.cosmicrayalbedo(e, geo.fov_sr, 0.5) for e in energy])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.step(energy, cxb, color="black", label="diffuse X-ray background")
    ax.step(energy, albedo, color="red", label="albedo photons")
    ax.step(energy, particles, color="blue", label="cosmic rays")
    ax.step(energy, cralbedo, color="green", label="cosmic ray induced albedo")
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlabel(r"Energy (keV)")
    ax.set_ylabel(r"$\frac{dN}{dt}\;(\mathrm{photons\ cm^{-2}\ s^{-1}\ keV^{-1}})$")
    ax.set_title("Expected Background Components")

    # Put legend outside the plot, on the right
    ax.legend(fontsize=9, frameon=True, framealpha=0.9)
    # Make room for the legend
    plt.tight_layout()
    plt.savefig(output_dir / "sampleBKG.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f'done: saved as sampleBKG.png')


def plot_effective_area_arf(mission, resp_dir, output_dir):
### See what is in your arf file. MAKE SURE TO HAVE THE ARF IN THE RIGHT PLACE BEFORE RUNNING THIS. ###
### example terminal input given below ###

    """pysimsat.testPlots.plot_effective_area_arf(mission, "response_files", "outputs")"""

    output_dir = Path(output_dir)
    resp_dir = Path(resp_dir)
        
    output_dir.mkdir(parents=True, exist_ok=True)
    resp_dir.mkdir(parents=True, exist_ok=True)

    with fits.open(resp_dir / f"{mission.name}.arf") as hdul:
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
    print("done. Saved as ARFeffectiveArea.png")


def plot_effective_area(mission, detector, mask, output_dir, start, end, num, ylow, yhigh):
### Plot the effective area on a given range. Also graphs the optics transmission and the detector attenuation ###
### example terminal input for BAT given below ###

    """pysimsat.testPlots.plot_effective_area(mission, detector, mask, "outputs", 10, 1000, 1000, 0, 1500)"""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    energy_vals = np.linspace(start, end, num)

    effective_area = np.array([detector.effective_area(energy=e) for e in energy_vals])
    mask_trans = np.array([mask.transmission(energy=e) for e in energy_vals])
    det_abs = np.array([detector.det_absorption(energy=e) for e in energy_vals])

    fig, ax1 = plt.subplots(figsize=(8, 5))

    # Left y-axis: Effective area
    ax1.plot(energy_vals, effective_area, color="black", linewidth=2, label="Effective Area")
    ax1.set_xlabel(r"Energy ($keV$)")
    ax1.set_ylabel(r"Effective Area ($cm^2$)", color="black")
    ax1.set_ylim(ylow, yhigh)
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
    ax1.legend(lines,labels,loc = "upper left",fontsize=8) #type: ignore

    #plt.title(f"{mission.name} Effective Area and Efficiencies")
    plt.tight_layout()
    plt.savefig(output_dir / "TotalEffectiveArea.png", dpi=300, bbox_inches="tight")
    plt.show()
    print("done. Saved as TotalEffectiveArea")
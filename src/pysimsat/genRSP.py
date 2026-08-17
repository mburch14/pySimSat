from xspec import *
from . import MissionClasses as mc
import matplotlib.pyplot as plt
import numpy as np
import commentjson
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


def load_instrument(instrument_name, instrument_json):
    with open(instrument_json) as f:
        instruments = commentjson.load(f)
    return instruments[instrument_name]


def load_source(source_name, source_json):
    with open(source_json) as f:
        sources = commentjson.load(f)

    return sources[source_name]


def build_instrument(chars, sourcechars, instrumentname):

    #only add solmod if it exists in source.json
    kwargs = {}
    if "solmod" in sourcechars:
        kwargs["solmod"] = sourcechars["solmod"]

    #This is for our specific Cubesat
    orb = mc.Orbit(altitude = chars["altitude"], inclination = chars['inclination'])
    geo = mc.geometry( config = chars['config'])
    mission = mc.Mission(instrumentname, chars['e_min'], chars['e_max'])
    mask = mc.optics(thickness = chars['config']['maskh'], mask_material = chars['mask_material'], mask_density = chars['mask_material_density'], localized = sourcechars["localized"])
    detector = mc.detector(geometry = geo, orbit = orb, mission = mission, optics = mask, res = chars["spec_resolution"], grad = chars["spec_gradient"], low_ecut = chars["low_ecut"], material = chars["detector_material"], mat_density = chars["det_material_density"])
    background = mc.BackgroundModel(detector = detector, orbit = orb, **({"solmod": sourcechars["solmod"]} if "solmod" in sourcechars else {}))

    return orb, geo, mission, detector, background, mask


def main(geo, mission, detector, output_dir, resp_dir):

    print(f'field of view: {geo.fov_sr} steradians')

    try:
        print(f'half coded field of view: {geo.half_coded_fov} steradians')
    except:
        pass

    #Generates the .arf and the .rsp file to be used by xspec. Then, generates a ASCII file for the background spectrum.
    arfname = detector.gen_arf(energy_lo = detector.energy_low, energy_hi = detector.energy_high, arf=resp_dir / f"{mission.name}.arf")
    detector.gen_rsp(arfname, rsp = resp_dir / f"{mission.name}.rsp")

    energy_vals = np.linspace(mission.energymin, mission.energymax, mission.energymax - mission.energymin + 1)
    effective_area = [detector.effective_area(energy=e) for e in energy_vals]

    plt.figure(figsize=(8,5))
    plt.plot(energy_vals, effective_area, color="black")
    plt.xlabel("Energy (keV)")
    plt.ylabel(r"Effective Area ($cm^2$)")
    plt.xscale('log')
    plt.title(f'effective area of {mission.name}')
    plt.savefig(output_dir / "effectiveArea.png", dpi=300, bbox_inches="tight")
    plt.close()

def gen_rsp(instrument, source, instrument_json, source_file, output_dir, resp_dir):
    chars = load_instrument(instrument, instrument_json)
    sourcechars = load_source(source, source_file)
    orb, geo, mission, detector, background, mask = build_instrument(chars, sourcechars, instrument)
    main(geo, mission, detector, output_dir, resp_dir)

    return chars, sourcechars, background, mission
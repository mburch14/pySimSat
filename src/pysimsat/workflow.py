from .genRSP import gen_rsp
from .genBKG import gen_background
from .genOBS import gen_observation
from pathlib import Path

def run_all(instrument, source, instrument_json, source_file, output_dir, spec_dir, resp_dir):
###    Run the complete PySimSat simulation pipeline.
###
###    Steps:
###    1. Generate response files
###    2. Generate background spectrum
###    3. Generate observation spectrum
###
###    Example terminal input is given for Swift/BAT (top) and CZTI (bottom)
###
###    pysimsat.workflow.run_all("SWIFTBAT", "CRAB-SWIFTBAT", "example_sat_source/instrumentCharacteristics.json", "example_sat_source/source.json", "outputs", "spectrum_files", "response_files")
###    pysimsat.workflow.run_all("ASTROSAT", "CRAB-ASTROSAT", "example_sat_source/instrumentCharacteristics.json", "example_sat_source/source.json", "outputs", "spectrum_files", "response_files")

    output_dir = Path(output_dir)
    spec_dir = Path(spec_dir)
    resp_dir = Path(resp_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    spec_dir.mkdir(parents=True, exist_ok=True)
    resp_dir.mkdir(parents=True, exist_ok=True)

    # Generate response files
    print("\nGenerating response files...\n")
    chars, sourcechars, background, mission = gen_rsp(instrument = instrument, source = source, instrument_json = instrument_json, source_file = source_file, output_dir = output_dir, resp_dir = resp_dir)
    print(f"\nResponse files generated for {instrument} and {source} in the {resp_dir} directory.\n")

    # Generate background spectrum
    print("\nGenerating background spectrum...\n")
    gen_background(background, mission, sourcechars, chars["num_det_pixels"], output_dir, spec_dir, resp_dir)
    print(f"\nBackground spectrum generated for {instrument} and {source} in the {spec_dir} and {output_dir} directory.\n")

    # Generate observation spectrum
    print("\nGenerating observation spectrum...\n")
    gen_observation(mission, sourcechars, output_dir, spec_dir, resp_dir, chars=chars)
    print(f"\nObservation spectrum generated for {instrument} and {source} in the {spec_dir} and {output_dir} directory.\n")
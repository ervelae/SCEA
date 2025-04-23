# Now working with the new SMARTCARB_modified dataset

import os
import csv
import numpy as np
import pandas as pd
from tqdm import tqdm  # progress bar
import sys
import xarray as xr

# INITIALIZE PARAMETERS
SMARTCARB_DATA_PATH = "/Users/eliaserv/Documents/VSCode_projects/SCEA/SCEA/SMARTCARB_combined.nc"
OUTPUT_PATH = "/Users/eliaserv/Documents/VSCode_projects/SCEA/SMARTCARB_plumes_with_SCEA"
SCRIPTS_PATH = "/Users/eliaserv/Documents/VSCode_projects/SCEA/SCEA"

SCEA_stds = 3.0
SCEA_radius_func_sigmas_threshold = 1.8
SMARTCARB_no2_noise_scenario = "noisefree"
output_filename_appendix = "_test"

# Add the directory containing the .py file to the system path
sys.path.append(SCRIPTS_PATH)
import SCEA

# Load dataset once
SMARTCARB_combined = xr.open_dataset(
    SMARTCARB_DATA_PATH,
    decode_cf=False,
    mask_and_scale=False
)

# For testing, take only firs 50 files
SMARTCARB_combined = SMARTCARB_combined.isel(file=slice(0, 50))


print("CPU count: "+str(os.cpu_count()))

def process_file(i):
    """Process file #i and return (i, clusters)."""
    data = SMARTCARB_combined.isel(file=i)
    vals = data[f"NO2_{SMARTCARB_no2_noise_scenario}"].values.ravel()
    lon = data["lon"].values.ravel()
    lat = data["lat"].values.ravel()

    mask = ~np.isnan(vals)
    points = np.vstack((lon[mask], lat[mask])).T
    values = vals[mask]

    if len(points) < 20:
        return i, np.array([])

    clusters = SCEA.scea(
        points,
        values,
        radius_func="default",
        n_clusters="auto",
        point_value_threshold="stds_from_median",
        stds=SCEA_stds,
        distance_matrix="euclidean",
        radius_func_sigmas_threshold=SCEA_radius_func_sigmas_threshold,
        max_points_in_start_radius=6,
        local_box_size=3,
        verbose=False,
    )
    return i, clusters

if __name__ == "__main__":
    print("SCEA clustering run on SMARTCARB data.")
    print(f"SMARTCARB data loaded from {SMARTCARB_DATA_PATH}.")
    print(f"Output will be saved in {OUTPUT_PATH}.")

    n_files = len(SMARTCARB_combined.file)
    my_clusters_per_file = [None] * n_files

    from multiprocessing import Pool

    with Pool() as pool:
        # chunksize=1 hands one file at a time to each worker
        results = pool.imap_unordered(process_file, range(n_files), chunksize=1)
        for idx, clusters in tqdm(results, total=n_files):
            my_clusters_per_file[idx] = clusters

    # Save results
    date_now = pd.Timestamp.now().strftime("%Y-%m-%d_%H-%M")
    out_fname = (
        f"SCEA_clusters_on_SMARTCARB_{SMARTCARB_no2_noise_scenario}"
        f"_noise_{date_now}{output_filename_appendix}.csv"
    )
    out_path = os.path.join(OUTPUT_PATH, out_fname)

    with open(out_path, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        for row in my_clusters_per_file:
            writer.writerow(row)

    print(f"\nClusters saved in {out_path}")

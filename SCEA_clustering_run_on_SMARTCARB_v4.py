# Now working with the new SMARTCARB_modified dataset

import os
import csv
import numpy as np
import pandas as pd
from tqdm import tqdm  # progress bar
import sys
import xarray as xr
import multiprocessing as mp
import dask
from dask.distributed import Client, LocalCluster


# INITIALIZE PARAMETERS


SMARTCARB_DATA_PATH = (
    "/Users/eliaserv/Documents/VSCode_projects/SCEA/SCEA/SMARTCARB_combined.nc"
)
OUTPUT_PATH = (
    "/Users/eliaserv/Documents/VSCode_projects/SCEA/SMARTCARB_plumes_with_SCEA"
)
SCRIPTS_PATH = "/Users/eliaserv/Documents/VSCode_projects/SCEA/SCEA"
"""
SMARTCARB_DATA_PATH = "/scratch/project_2008159/eliaserv/SMARTCARB/SMARTCARB_combined.nc"
OUTPUT_PATH = "/scratch/project_2008159/eliaserv/SMARTCARB/outputs"
SCRIPTS_PATH = "/projappl/project_2008159/eliaserv"
"""

SCEA_stds = 3.0
SCEA_radius_func_sigmas_threshold = 1.8
SMARTCARB_no2_noise_scenario = "lownoise"
output_filename_appendix = "test"

# Add the directory containing the .py file to the system path
sys.path.append(SCRIPTS_PATH)
import SCEA

"""
# Load dataset once
SMARTCARB_combined = xr.open_dataset(
    SMARTCARB_DATA_PATH,
    decode_cf=False,
    mask_and_scale=False
)
"""

SMARTCARB_combined = xr.open_zarr(
    "SMARTCARB.zarr",
    consolidated=True,
    decode_cf=False,
    mask_and_scale=False,
    chunks={"file": 1}
)

# For testing, take only first 50 files
#SMARTCARB_combined = SMARTCARB_combined.isel(file=slice(0, 50))
#SMARTCARB_combined.load()

# SMARTCARB_list = [SMARTCARB_combined.isel(file=i) for i in range(len(SMARTCARB_combined.file))]

print("SCEA clustering run on SMARTCARB data.")
print(f"SMARTCARB data loaded from {SMARTCARB_DATA_PATH}.")
print(f"Output will be saved in {OUTPUT_PATH}.")
print("SCEA parameters:")
print(f"  SCEA_stds: {SCEA_stds}")
print(f"  SCEA_radius_func_sigmas_threshold: {SCEA_radius_func_sigmas_threshold}")
print(f"  SMARTCARB_no2_noise_scenario: {SMARTCARB_no2_noise_scenario}")
print(f"  output_filename_appendix: {output_filename_appendix}")
print("CPU count: " + str(os.cpu_count()))


def process_file(i):
    """Process file #i and return (i, clusters)."""
    data = SMARTCARB_combined.isel(file=i)
    vals = data[f"NO2_{SMARTCARB_no2_noise_scenario}"].data.ravel()
    lon = data["lon"].data.ravel()
    lat = data["lat"].data.ravel()

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
    # 1) Start Dask
    from dask.distributed import Client, LocalCluster
    import os
    nthreads = os.cpu_count()
    cluster = LocalCluster(
        n_workers=1,
        threads_per_worker=nthreads,
        memory_limit="30GB",
        local_directory="/scratch/dask-space",
    )
    client = Client(cluster)

    # 2) Prepare tasks
    n_files = len(SMARTCARB_combined.file)
    tasks = [dask.delayed(process_file)(i) for i in range(n_files)]

    # 3) Execute
    results = dask.compute(*tasks, scheduler="threads")

    # 4) Save incrementally
    import csv, os, pandas as pd
    date_now = pd.Timestamp.now().strftime("%Y-%m-%d_%H-%M")
    out_fname = (
        f"SCEA_clusters_{SMARTCARB_no2_noise_scenario}"
        f"_{SCEA_stds}_{SCEA_radius_func_sigmas_threshold}"
        f"_{date_now}_{output_filename_appendix}.csv"
    )
    out_path = os.path.join(OUTPUT_PATH, out_fname)
    with open(out_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        for _, clusters in results:
            writer.writerow(clusters.tolist())

    print(f"Clusters saved in {out_path}")

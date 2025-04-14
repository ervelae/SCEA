# Import libraries

import os
import csv
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import haversine_distances
from tqdm.notebook import tqdm  # progress bar

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from cmcrameri import cm
from cartopy import crs as ccrs, feature as cfeature
import seaborn as sns
import sys
import time


# INITIALIZE PARAMETERS

"""
SMARTCARB_DATA_PATH = "/Volumes/One Touch/SMARTCARB/Sentinel_7_CO2"
OUTPUT_PATH = (
    "/Users/eliaserv/Documents/VSCode_projects/SCEA/SMARTCARB_plumes_with_SCEA"
)
SCRIPTS_PATH = "/Users/eliaserv/Documents/VSCode_projects/SCEA/SCEA"
"""

SMARTCARB_DATA_PATH = "/scratch/project_2008159/eliaserv/SMARTCARB/Sentinel_7_CO2"
OUTPUT_PATH = "/scratch/project_2008159/eliaserv/SMARTCARB/outputs"
SCRIPTS_PATH = "/projappl/project_2008159/eliaserv"


SCEA_stds = 3.1
SCEA_radius_func_sigmas_threshold = 1.0
SMARTCARB_no2_noise_scenario = "high"
output_filename_appendix = "_test"


# THE MAIN PART

# Add the directory containing the .py file to the system path
sys.path.append(SCRIPTS_PATH)

import smartcarb_modified
import SCEA


filenames_all_SMARTCARB = os.listdir(SMARTCARB_DATA_PATH)
filenames_all_SMARTCARB.sort()


my_clusters_per_file = []
my_time_spent_per_file = []


for i in tqdm(range(10)):
    # for i in range(len(filenames_all_SMARTCARB)):

    # Print progress
    print(f"{i+1}/{len(filenames_all_SMARTCARB)} file", end="\t")

    # Find the data
    filename = os.path.join(SMARTCARB_DATA_PATH, filenames_all_SMARTCARB[i])
    # data = ddeq.smartcarb.read_level2(filename, no2_noise_scenario='high', only_observations=False)
    data = smartcarb_modified.read_level2(
        filename,
        no2_noise_scenario=SMARTCARB_no2_noise_scenario,
        only_observations=False,
        # no2_cloud_threshold=1,
    )

    # Data in proper format
    lon = np.array(data["lon"]).flatten()
    lat = np.array(data["lat"]).flatten()
    no2 = np.array(data["NO2"]).flatten()
    not_nan = np.logical_not(np.isnan(no2))
    lon = lon[not_nan]
    lat = lat[not_nan]
    no2 = no2[not_nan]
    points = np.transpose([lon, lat])

    # Time
    start_time = time.time()

    if len(points) < 20:
        my_clusters_per_file.append(np.array([]))
        my_time_spent_per_file.append(0)
        print()
        continue

    # My clustering algorithm
    clusters = SCEA.scea(
        points,
        no2,
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

    # Append clusters to the list
    my_clusters_per_file.append(clusters)

    # Time
    my_time_spent_per_file.append(time.time() - start_time)
    print(f"Execution time: {time.time() - start_time:.2f} seconds.")


date_now = pd.Timestamp.now().strftime("%Y-%m-%d_%H-%M")
print(date_now)

# Save the inhomogeneous list as a CSV
with open(
    OUTPUT_PATH
    + f"/SCEA_clusters_on_SMARTCARB_{SMARTCARB_no2_noise_scenario}_noise_{date_now}{output_filename_appendix}.csv",
    mode="w",
    newline="",
) as csv_file:
    writer = csv.writer(csv_file)
    for row in my_clusters_per_file:
        writer.writerow(row)

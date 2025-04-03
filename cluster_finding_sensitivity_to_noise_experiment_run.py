import numpy as np
import pandas as pd
import SCEA
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import cmcrameri.cm as cmc
from matplotlib import colormaps
from tqdm import tqdm
import datetime
import os
import sys
from pathlib import Path

import helper_functions
from helper_functions import calculate_metrics_for_multiclusters


# Set the default colormap
plt.rcParams["image.cmap"] = "cmc.batlow"

# Paths
current_directory = Path(Path.cwd())
parent_directory = current_directory.parent
outputs_path = Path("/Users/eliaserv/Documents/VSCode_projects/SCEA/outputs")
plots_path = Path("/Users/eliaserv/Documents/VSCode_projects/SCEA/plots")
filename = (
    "scores_per_stds_noise_linear_"
    + datetime.datetime.now().strftime("%y-%m-%d_%H-%M")
    + ".npy"
)

# ===========


linear_function_polar = lambda r: np.maximum(-r * 4 + 16, 1)

meshgrid = np.meshgrid(
    np.linspace(-12, 12, 96, endpoint=False), np.linspace(-12, 12, 96, endpoint=False)
)
x = meshgrid[0].flatten()
y = meshgrid[1].flatten()

# Convert Cartesian coordinates to polar coordinates
R = np.sqrt(x**2 + y**2)  # How far the grid point is from the origin

values = linear_function_polar(R)

points_linear_polar_2 = np.array([x + 24, y]).T
x_linear_polar_2 = x + 24
y_linear_polar_2 = y
meshgrid_linear_polar_2 = meshgrid
values_linear_polar_2 = values
labels_linear_polar_2 = values > 1

# linear_function_polar = lambda r: np.maximum(-r*4 + 16, 1)

X = []
Y = []
VALUES = []
LABELS = []
POINTS = []

k = 0
for i in range(4):
    for j in range(4):

        k = k + 1
        divide_by = 16 / k
        linear_function_polar = lambda r: np.maximum((-r * 4 + 16) / divide_by, 0)

        meshgrid = np.meshgrid(
            np.linspace(-12, 12, 96, endpoint=False),
            np.linspace(-12, 12, 96, endpoint=False),
        )
        x = meshgrid[0].flatten()
        y = meshgrid[1].flatten()

        # Convert Cartesian coordinates to polar coordinates
        R = np.sqrt(x**2 + y**2)  # How far the grid point is from the origin

        values = linear_function_polar(R)

        x_linear_polar = x + 24 * i
        y_linear_polar = y + 24 * j
        points_linear_polar = np.array([x_linear_polar, y_linear_polar]).T
        meshgrid_linear_polar = meshgrid
        values_linear_polar = values
        labels_linear_polar = (values > 0) * k

        X.append(x_linear_polar)
        Y.append(y_linear_polar)
        VALUES.append(values_linear_polar)
        LABELS.append(labels_linear_polar)
        POINTS.append(points_linear_polar)

x_linear_polar = np.concatenate(X)
y_linear_polar = np.concatenate(Y)
values_linear_polar = np.concatenate(VALUES)
labels_linear_polar = np.concatenate(LABELS)
points_linear_polar = np.concatenate(POINTS)



## Big run

values = values_linear_polar
points = points_linear_polar
labels = labels_linear_polar

radius_func_sigmas_threshold = 0.5
noise_levels = np.arange(0, 2.1, 1)  # Standard deviations of the noise
n_repetitions_per_noise_level = 2
stds = np.arange(4, 5.1, 0.5)


# Initialize lists
f1_general_list = []
precision_general_list = []
recall_general_list = []
f1_scores_list = []
n_unique_found_clusters_in_true_cluster_list = []
n_true_positive_clusters_list = []
n_false_positive_clusters_list = []
n_false_negative_clusters_individual_runs = []
f1_scores_whole_clusters_list = []
mean_true_positive_clusters_size_list = []
median_true_positive_clusters_size_list = []
mean_false_positive_clusters_size_list = []
median_false_positive_clusters_size_list = []


# First standardize data set
values_standardized = (values - values.mean()) / values.std()

for k, std in enumerate(tqdm(stds)):

    f1_general_averages = [0] * len(noise_levels)
    precision_general_averages = [0] * len(noise_levels)
    recall_general_averages = [0] * len(noise_levels)
    f1_scores_averages = [0] * len(noise_levels)
    n_unique_found_clusters_in_true_cluster_averages = [0] * len(noise_levels)
    n_true_positive_clusters_averages = [0] * len(noise_levels)
    n_false_positive_clusters_averages = [0] * len(noise_levels)
    n_false_negative_clusters_averages = [0] * len(noise_levels)
    f1_scores_whole_clusters_averages = [0] * len(noise_levels)
    mean_true_positive_clusters_size_averages = [0] * len(noise_levels)
    median_true_positive_clusters_size_averages = [0] * len(noise_levels)
    mean_false_positive_clusters_size_averages = [0] * len(noise_levels)
    median_false_positive_clusters_size_averages = [0] * len(noise_levels)
    
    
    # Iterate over noise levels
    for i, noise_sigma in enumerate(noise_levels):

        f1_general_individual_runs = [0] * n_repetitions_per_noise_level
        precision_general_individual_runs = [0] * n_repetitions_per_noise_level
        recall_general_individual_runs = [0] * n_repetitions_per_noise_level
        f1_scores_individual_runs = [0] * n_repetitions_per_noise_level
        n_unique_found_clusters_in_true_cluster_individual_runs = [0] * n_repetitions_per_noise_level
        n_true_positive_clusters_individual_runs = [0] * n_repetitions_per_noise_level
        n_false_positive_clusters_individual_runs = [0] * n_repetitions_per_noise_level
        n_false_negative_clusters_individual_runs = [0] * n_repetitions_per_noise_level
        f1_scores_whole_clusters_individual_runs = [0] * n_repetitions_per_noise_level
        mean_true_positive_clusters_size_individual_runs = [0] * n_repetitions_per_noise_level
        median_true_positive_clusters_size_individual_runs = [0] * n_repetitions_per_noise_level
        mean_false_positive_clusters_size_individual_runs = [0] * n_repetitions_per_noise_level
        median_false_positive_clusters_size_individual_runs = [0] * n_repetitions_per_noise_level


        # Repeat the experiment multiple times
        for j in range(n_repetitions_per_noise_level):
            
            # add noise
            values_noisy = values_standardized + np.random.normal(0, noise_sigma, values.shape)

            # cluster
            clusters = SCEA.scea(
                points,
                values_noisy,
                radius_func="default",
                n_clusters="auto",
                point_value_threshold="stds_from_median",
                stds=std,
                distance_matrix="euclidean",
                radius_func_sigmas_threshold=radius_func_sigmas_threshold,
                max_points_in_start_radius=6,
                local_box_size=24,
                verbose=False,
            )

            
            # calculate metrics
            (
                f1_general,
                precision_general,
                recall_general,
                f1_scores,
                n_unique_found_clusters_in_true_cluster,
                n_true_positive_clusters,
                n_false_postive_clusters,
                n_false_negative_clusters,
                f1_scores_whole_clusters,
                mean_true_positive_clusters_size,
                median_true_positive_clusters_size,
                mean_false_positive_clusters_size,
                median_false_positive_clusters_size,
            ) = calculate_metrics_for_multiclusters(clusters, labels, verbose=False)

            

            f1_general_individual_runs[j] = f1_general
            precision_general_individual_runs[j] = precision_general
            recall_general_individual_runs[j] = recall_general
            f1_scores_individual_runs[j] = f1_scores
            n_unique_found_clusters_in_true_cluster_individual_runs[j] = n_unique_found_clusters_in_true_cluster
            n_true_positive_clusters_individual_runs[j] = n_true_positive_clusters
            n_false_positive_clusters_individual_runs[j] = n_false_postive_clusters
            n_false_negative_clusters_individual_runs[j] = n_false_negative_clusters
            f1_scores_whole_clusters_individual_runs[j] = f1_scores_whole_clusters
            mean_true_positive_clusters_size_individual_runs[j] = mean_true_positive_clusters_size
            median_true_positive_clusters_size_individual_runs[j] = median_true_positive_clusters_size
            mean_false_positive_clusters_size_individual_runs[j] = mean_false_positive_clusters_size
            median_false_positive_clusters_size_individual_runs[j] = median_false_positive_clusters_size

        # Average over the repetitions
        f1_general_averages[i] = np.mean(f1_general_individual_runs)
        precision_general_averages[i] = np.mean(precision_general_individual_runs)
        recall_general_averages[i] = np.mean(recall_general_individual_runs)
        f1_scores_averages[i] = np.mean(f1_scores_individual_runs, axis=0)
        n_unique_found_clusters_in_true_cluster_averages[i] = np.mean(n_unique_found_clusters_in_true_cluster_individual_runs)
        n_true_positive_clusters_averages[i] = np.mean(n_true_positive_clusters_individual_runs)
        n_false_positive_clusters_averages[i] = np.mean(n_false_positive_clusters_individual_runs)
        n_false_negative_clusters_averages[i] = np.mean(n_false_negative_clusters_individual_runs)
        f1_scores_whole_clusters_averages[i] = np.mean(f1_scores_whole_clusters_individual_runs)
        mean_true_positive_clusters_size_averages[i] = np.mean(mean_true_positive_clusters_size_individual_runs)
        median_true_positive_clusters_size_averages[i] = np.mean(median_true_positive_clusters_size_individual_runs)
        mean_false_positive_clusters_size_averages[i] = np.mean(mean_false_positive_clusters_size_individual_runs)
        median_false_positive_clusters_size_averages[i] = np.mean(median_false_positive_clusters_size_individual_runs)


    # Append the averages to the lists
    f1_general_list.append(np.array(f1_general_averages))
    precision_general_list.append(np.array(precision_general_averages))
    recall_general_list.append(np.array(recall_general_averages))
    f1_scores_list.append(np.array(f1_scores_averages))
    n_unique_found_clusters_in_true_cluster_list.append(np.array(n_unique_found_clusters_in_true_cluster_averages))
    n_true_positive_clusters_list.append(np.array(n_true_positive_clusters_averages))
    n_false_positive_clusters_list.append(np.array(n_false_positive_clusters_averages))
    n_false_negative_clusters_individual_runs.append(np.array(n_false_negative_clusters_individual_runs))
    f1_scores_whole_clusters_list.append(np.array(f1_scores_whole_clusters_averages))
    mean_true_positive_clusters_size_list.append(np.array(mean_true_positive_clusters_size_averages))
    median_true_positive_clusters_size_list.append(np.array(median_true_positive_clusters_size_averages))
    mean_false_positive_clusters_size_list.append(np.array(mean_false_positive_clusters_size_averages))
    median_false_positive_clusters_size_list.append(np.array(median_false_positive_clusters_size_averages))

    
all_in_one_list = [
    f1_general_list,
    precision_general_list,
    recall_general_list,
    f1_scores_list,
    n_unique_found_clusters_in_true_cluster_list,
    n_true_positive_clusters_list,
    n_false_positive_clusters_list,
    n_false_negative_clusters_individual_runs,
    f1_scores_whole_clusters_list,
    mean_true_positive_clusters_size_list,
    median_true_positive_clusters_size_list,
    mean_false_positive_clusters_size_list,
    median_false_positive_clusters_size_list,
]


np.save(outputs_path / filename, np.array(all_in_one_list, dtype=object))




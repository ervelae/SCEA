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



# ==================================
# Paths
# ==================================
current_directory = Path(Path.cwd())
parent_directory = current_directory.parent
outputs_path = Path("/Users/eliaserv/Documents/VSCode_projects/SCEA/outputs")
plots_path = Path("/Users/eliaserv/Documents/VSCode_projects/SCEA/plots")
filename = (
    "scores_per_stds_noise_linear_"
    + datetime.datetime.now().strftime("%y-%m-%d_%H-%M")
    + ".npy"
)

# =====================================
# Parameters
# =====================================
radius_func_sigmas_threshold = 0.5
noise_levels = np.arange(0, 2.1, 1)  # Standard deviations of the noise
n_repetitions_per_noise_level = 2
stds = np.arange(3, 5.1, 1)
data_set = "linear_polar"

# Set the default colormap
plt.rcParams["image.cmap"] = "cmc.batlow"


# ====================================
# Functions
# ====================================

def single_experiment(
    points,
    values,
    labels,
    std,
    radius_func_sigmas_threshold,
):
    """
    Run a single experiment with the given parameters.
    """

    # Cluster
    clusters = SCEA.scea(
        points,
        values,
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
    
    metrics_dict = calculate_metrics_for_multiclusters(
        clusters,
        labels,
        verbose=False,
    )

    return metrics_dict








def single_experiment_with_repetitions(
    points,
    values,
    labels,
    noise_sigma,
    std,
    radius_func_sigmas_threshold,
    n_repetitions_per_noise_level,
):
    """
    Run a single experiment with the given parameters.
    """

    # Create a noisy version of the values for all repetitions
    values_noisy_all = values + np.random.normal(
        0, noise_sigma, (n_repetitions_per_noise_level, len(values))
    )

    # Initialize lists to store the results for each repetition
    f1_general_array = np.zeros(n_repetitions_per_noise_level)
    precision_general_array = np.zeros(n_repetitions_per_noise_level)
    recall_general_array = np.zeros(n_repetitions_per_noise_level)
    f1_scores_array = np.zeros((n_repetitions_per_noise_level, len(np.unique(labels))-1))
    n_unique_found_clusters_in_true_cluster_array = np.zeros((n_repetitions_per_noise_level, len(np.unique(labels))-1))
    n_true_positive_clusters_array = np.zeros(n_repetitions_per_noise_level)
    n_false_positive_clusters_array = np.zeros(n_repetitions_per_noise_level)
    n_false_negative_clusters_array = np.zeros(n_repetitions_per_noise_level)
    f1_scores_whole_clusters_array = np.zeros(n_repetitions_per_noise_level)
    mean_true_positive_clusters_size_array = np.zeros(n_repetitions_per_noise_level)
    median_true_positive_clusters_size_array = np.zeros(n_repetitions_per_noise_level)
    mean_false_positive_clusters_size_array = np.zeros(n_repetitions_per_noise_level)
    median_false_positive_clusters_size_array = np.zeros(n_repetitions_per_noise_level)

    for i in range(n_repetitions_per_noise_level):
        
        # Store the results for this run
        metrics_dict = single_experiment(
            points,
            values_noisy_all[i],
            labels,
            std,
            radius_func_sigmas_threshold,
            )
        
        f1_general_array[i] = metrics_dict["f1_general"]
        precision_general_array[i] = metrics_dict["precision_general"]
        recall_general_array[i] = metrics_dict["recall_general"]
        f1_scores_array[i] = metrics_dict["f1_scores"]
        n_unique_found_clusters_in_true_cluster_array[i] = metrics_dict["n_found_clusters_in_true_cluster"]
        n_true_positive_clusters_array[i] = metrics_dict["n_true_positive_clusters"]
        n_false_positive_clusters_array[i] = metrics_dict["n_false_positive_clusters"]
        n_false_negative_clusters_array[i] = metrics_dict["n_false_negative_clusters"]
        f1_scores_whole_clusters_array[i] = metrics_dict["f1_scores_whole_clusters"]
        mean_true_positive_clusters_size_array[i] = metrics_dict["mean_true_positive_clusters_size"]
        median_true_positive_clusters_size_array[i] = metrics_dict["median_true_positive_clusters_size"]
        mean_false_positive_clusters_size_array[i] = metrics_dict["mean_false_positive_clusters_size"]
        median_false_positive_clusters_size_array[i] = metrics_dict["median_false_positive_clusters_size"]


    # Calculate the average metrics over all repetitions
    f1_general = np.mean(f1_general_array)
    precision_general = np.mean(precision_general_array)
    recall_general = np.mean(recall_general_array)
    f1_scores = np.mean(f1_scores_array, axis=0)
    n_unique_found_clusters_in_true_cluster = np.mean(n_unique_found_clusters_in_true_cluster_array)
    n_true_positive_clusters = np.mean(n_true_positive_clusters_array)
    n_false_positive_clusters = np.mean(n_false_positive_clusters_array)
    n_false_negative_clusters = np.mean(n_false_negative_clusters_array)
    f1_scores_whole_clusters = np.mean(f1_scores_whole_clusters_array)
    mean_true_positive_clusters_size = np.mean(mean_true_positive_clusters_size_array)
    median_true_positive_clusters_size = np.mean(median_true_positive_clusters_size_array)
    mean_false_positive_clusters_size = np.mean(mean_false_positive_clusters_size_array)
    median_false_positive_clusters_size = np.mean(median_false_positive_clusters_size_array)

    # Calculate the standard deviation of the metrics over all repetitions ?
    # TODO ?

    return (
        f1_general,
        precision_general,
        recall_general,
        f1_scores,
        n_unique_found_clusters_in_true_cluster,
        n_true_positive_clusters,
        n_false_positive_clusters,
        n_false_negative_clusters,
        f1_scores_whole_clusters,
        mean_true_positive_clusters_size,
        median_true_positive_clusters_size,
        mean_false_positive_clusters_size,
        median_false_positive_clusters_size,
    )


# ====================================
# Creating the data set
# ====================================

def create_data_set(data_set="linear_polar"):
    """
    Create a data set for the experiment.
    """

    if data_set == "linear_polar":
        data_function = lambda r: np.maximum(-r * 4 + 16, 1)

    meshgrid = np.meshgrid(
        np.linspace(-12, 12, 96, endpoint=False), np.linspace(-12, 12, 96, endpoint=False)
    )
    x = meshgrid[0].flatten()
    y = meshgrid[1].flatten()

    # Convert Cartesian coordinates to polar coordinates
    R = np.sqrt(x**2 + y**2)  # How far the grid point is from the origin

    values = data_function(R)

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
            data_function = lambda r: np.maximum((-r * 4 + 16) / divide_by, 0)

            meshgrid = np.meshgrid(
                np.linspace(-12, 12, 96, endpoint=False),
                np.linspace(-12, 12, 96, endpoint=False),
            )
            x = meshgrid[0].flatten()
            y = meshgrid[1].flatten()

            # Convert Cartesian coordinates to polar coordinates
            R = np.sqrt(x**2 + y**2)  # How far the grid point is from the origin

            values = data_function(R)

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

    return (
        values_linear_polar,
        points_linear_polar,
        labels_linear_polar,
    ) 




# ====================================
# The experiment
# ====================================

values, points, labels = create_data_set(data_set="linear_polar") 

# First standardize data set
values_standardized = (values - values.mean()) / values.std()

# Initialize lists
f1_general_array = np.zeros((len(stds), len(noise_levels)))
precision_general_array = np.zeros((len(stds), len(noise_levels)))
recall_general_array = np.zeros((len(stds), len(noise_levels)))
f1_scores_per_label_array = np.zeros((len(stds), len(noise_levels), len(np.unique(labels))-1)) # 3D array
n_unique_found_clusters_in_true_cluster_array = np.zeros((len(stds), len(noise_levels), len(np.unique(labels))-1))
n_true_positive_clusters_array = np.zeros((len(stds), len(noise_levels)))
n_false_positive_clusters_array = np.zeros((len(stds), len(noise_levels)))
n_false_negative_clusters_array = np.zeros((len(stds), len(noise_levels)))
f1_scores_whole_clusters_array = np.zeros((len(stds), len(noise_levels)))
mean_true_positive_clusters_size_array = np.zeros((len(stds), len(noise_levels)))
median_true_positive_clusters_size_array = np.zeros((len(stds), len(noise_levels)))
mean_false_positive_clusters_size_array = np.zeros((len(stds), len(noise_levels)))
median_false_positive_clusters_size_array = np.zeros((len(stds), len(noise_levels)))


# Iterate over the parameter std
for i, std in enumerate(tqdm(stds)):

    # Iterate over noise levels
    for j, noise_sigma in enumerate(noise_levels):

        # Run the experiment with repetitions
        (
            f1_general,
            precision_general,
            recall_general,
            f1_score_per_label,
            n_unique_found_clusters_in_true_cluster,
            n_true_positive_clusters,
            n_false_positive_clusters,
            n_false_negative_clusters,
            f1_scores_whole_clusters,
            mean_true_positive_clusters_size,
            median_true_positive_clusters_size,
            mean_false_positive_clusters_size,
            median_false_positive_clusters_size,
        ) = single_experiment_with_repetitions(
            points,
            values_standardized,
            labels,
            noise_sigma=noise_sigma,
            std=std,
            radius_func_sigmas_threshold=radius_func_sigmas_threshold,
            n_repetitions_per_noise_level=n_repetitions_per_noise_level
        )

        # Store the results for this run
        f1_general_array[i, j] = f1_general
        precision_general_array[i, j] = precision_general
        recall_general_array[i, j] = recall_general
        f1_scores_per_label_array[i, j] = f1_score_per_label
        n_unique_found_clusters_in_true_cluster_array[i, j] = n_unique_found_clusters_in_true_cluster
        n_true_positive_clusters_array[i, j] = n_true_positive_clusters
        n_false_positive_clusters_array[i, j] = n_false_positive_clusters
        n_false_negative_clusters_array[i, j] = n_false_negative_clusters
        f1_scores_whole_clusters_array[i, j] = f1_scores_whole_clusters
        mean_true_positive_clusters_size_array[i, j] = mean_true_positive_clusters_size
        median_true_positive_clusters_size_array[i, j] = median_true_positive_clusters_size
        mean_false_positive_clusters_size_array[i, j] = mean_false_positive_clusters_size
        median_false_positive_clusters_size_array[i, j] = median_false_positive_clusters_size


# Save the results
results = {
    "f1_general": f1_general_array,
    "precision_general": precision_general_array,
    "recall_general": recall_general_array,
    "f1_scores_per_label": f1_scores_per_label_array,
    "n_unique_found_clusters_in_true_cluster": n_unique_found_clusters_in_true_cluster_array,
    "n_true_positive_clusters": n_true_positive_clusters_array,
    "n_false_positive_clusters": n_false_positive_clusters_array,
    "n_false_negative_clusters": n_false_negative_clusters_array,
    "f1_scores_whole_clusters": f1_scores_whole_clusters_array,
    "mean_true_positive_clusters_size": mean_true_positive_clusters_size_array,
    "median_true_positive_clusters_size": median_true_positive_clusters_size_array,
    "mean_false_positive_clusters_size": mean_false_positive_clusters_size_array,
    "median_false_positive_clusters_size": median_false_positive_clusters_size_array,
    "stds": stds,
    "noise_levels": noise_levels,
    "n_repetitions_per_noise_level": n_repetitions_per_noise_level,
    "radius_func_sigmas_threshold": radius_func_sigmas_threshold,
    "data_set": data_set,
}

np.savez(outputs_path / filename, **results)
import numpy as np
from tqdm import tqdm
import datetime
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import sys

"""
This script is now supercomputer friendly.
It runs the experiment in parallel using ProcessPoolExecutor.
"""


# =====================================
# Parameters
# =====================================
radius_func_sigmas_threshold = 0.5
noise_levels = np.arange(0, 2.1, 1)  # Standard deviations of the noise
n_repetitions_per_noise_level = 6
stds = np.arange(3, 5.1, 1)
data_set = "linear_polar"

# ==================================
# Paths
# ==================================
"""
current_directory = Path(Path.cwd())
parent_directory = current_directory.parent
OUTPUT_PATH = Path("/Users/eliaserv/Documents/VSCode_projects/SCEA/outputs")
#plots_path = Path("/Users/eliaserv/Documents/VSCode_projects/SCEA/plots")
"""

OUTPUT_PATH = Path("/scratch/project_2008159/eliaserv/SMARTCARB/outputs")
SCRIPTS_PATH = Path("/projappl/project_2008159/eliaserv")

output_filename_appendix = "_test"

filename = (
    "scores_per_stds_noise"
    + data_set
    + "_"
    + str(radius_func_sigmas_threshold)
    + "_"
    + datetime.datetime.now().strftime("%y-%m-%d_%H-%M")
    + output_filename_appendix
    + ".npz"
)

sys.path.append(SCRIPTS_PATH)
from helper_functions import calculate_metrics_for_multiclusters
import SCEA

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


def _one_replication(args):
    """
    Helper to run one noisy realization.
    args is a tuple: (points, values, labels, noise_sigma, std,
                      radius_func_sigmas_threshold, rep_index)
    """
    points, values, labels, noise_sigma, std, radius_thr, rep_i = args

    # make this rep’s noisy values
    vals_noisy = values + np.random.normal(0, noise_sigma, size=values.shape)
    # run the core experiment
    metrics = single_experiment(points, vals_noisy, labels, std, radius_thr)
    return metrics  # it’s a dict


def single_experiment_with_repetitions(
    points,
    values,
    labels,
    noise_sigma,
    std,
    radius_func_sigmas_threshold,
    n_repetitions_per_noise_level,
    n_workers=None,  # NEW: how many processes
):
    """
    Run `n_repetitions_per_noise_level` reps in parallel.
    """

    # build argument tuples for each repetition
    args_list = [
        (points, values, labels, noise_sigma, std, radius_func_sigmas_threshold, i)
        for i in range(n_repetitions_per_noise_level)
    ]

    # store all metrics dicts here
    all_metrics = []

    # launch pool
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        # map returns in-order; as_completed lets you track progress
        futures = [pool.submit(_one_replication, args) for args in args_list]
        for fut in as_completed(futures):
            all_metrics.append(fut.result())

    # Now all_metrics is a list of length n_repetitions,
    # each entry is that metrics_dict.  We can vectorize extraction:
    def stack(key):
        return np.array([m[key] for m in all_metrics])

    f1_general_array = stack("f1_general")
    precision_general_array = stack("precision_general")
    recall_general_array = stack("recall_general")
    f1_scores_array = stack("f1_scores")  # shape = (n_reps, n_labels)
    n_unique_found_clusters_in_true_cluster_array = stack(
        "n_found_clusters_in_true_cluster"
    )
    n_true_positive_clusters_array = stack("n_true_positive_clusters")
    n_false_positive_clusters_array = stack("n_false_positive_clusters")
    n_false_negative_clusters_array = stack("n_false_negative_clusters")
    f1_scores_whole_clusters_array = stack("f1_scores_whole_clusters")
    mean_true_positive_clusters_size_array = stack("mean_true_positive_clusters_size")
    median_true_positive_clusters_size_array = stack(
        "median_true_positive_clusters_size"
    )
    mean_false_positive_clusters_size_array = stack("mean_false_positive_clusters_size")
    median_false_positive_clusters_size_array = stack(
        "median_false_positive_clusters_size"
    )

    print(f"f1_general_array {f1_general_array} with noise_level {noise_sigma}")

    # Finally compute means (and stds if you like)
    f1_general = f1_general_array.mean()
    precision_general = precision_general_array.mean()
    recall_general = recall_general_array.mean()
    f1_scores = f1_scores_array.mean(axis=0)
    n_unique_found_clusters_in_true_cluster = (
        n_unique_found_clusters_in_true_cluster_array.mean()
    )
    n_true_positive_clusters = n_true_positive_clusters_array.mean()
    n_false_positive_clusters = n_false_positive_clusters_array.mean()
    n_false_negative_clusters = n_false_negative_clusters_array.mean()
    f1_scores_whole_clusters = f1_scores_whole_clusters_array.mean()
    mean_true_positive_clusters_size = mean_true_positive_clusters_size_array.mean()
    median_true_positive_clusters_size = median_true_positive_clusters_size_array.mean()
    mean_false_positive_clusters_size = mean_false_positive_clusters_size_array.mean()
    median_false_positive_clusters_size = (
        median_false_positive_clusters_size_array.mean()
    )

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
        np.linspace(-12, 12, 96, endpoint=False),
        np.linspace(-12, 12, 96, endpoint=False),
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


if __name__ == "__main__":

    values, points, labels = create_data_set(data_set="linear_polar")

    # First standardize data set
    values_standardized = (values - values.mean()) / values.std()

    # Initialize lists
    f1_general_array = np.zeros((len(stds), len(noise_levels)))
    precision_general_array = np.zeros((len(stds), len(noise_levels)))
    recall_general_array = np.zeros((len(stds), len(noise_levels)))
    f1_scores_per_label_array = np.zeros(
        (len(stds), len(noise_levels), len(np.unique(labels)) - 1)
    )  # 3D array
    n_unique_found_clusters_in_true_cluster_array = np.zeros(
        (len(stds), len(noise_levels), len(np.unique(labels)) - 1)
    )
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
        tqdm.write(f"Processing std = {std}")

        # Iterate over noise levels
        for j, noise_sigma in enumerate(noise_levels):
            # tqdm.write(f"Processing noise level = {noise_sigma}")

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
                n_repetitions_per_noise_level=n_repetitions_per_noise_level,
                n_workers=n_repetitions_per_noise_level,
            )

            # Store the results for this run
            f1_general_array[i, j] = f1_general
            precision_general_array[i, j] = precision_general
            recall_general_array[i, j] = recall_general
            f1_scores_per_label_array[i, j] = f1_score_per_label
            n_unique_found_clusters_in_true_cluster_array[i, j] = (
                n_unique_found_clusters_in_true_cluster
            )
            n_true_positive_clusters_array[i, j] = n_true_positive_clusters
            n_false_positive_clusters_array[i, j] = n_false_positive_clusters
            n_false_negative_clusters_array[i, j] = n_false_negative_clusters
            f1_scores_whole_clusters_array[i, j] = f1_scores_whole_clusters
            mean_true_positive_clusters_size_array[i, j] = (
                mean_true_positive_clusters_size
            )
            median_true_positive_clusters_size_array[i, j] = (
                median_true_positive_clusters_size
            )
            mean_false_positive_clusters_size_array[i, j] = (
                mean_false_positive_clusters_size
            )
            median_false_positive_clusters_size_array[i, j] = (
                median_false_positive_clusters_size
            )

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

    np.savez(OUTPUT_PATH / filename, **results)

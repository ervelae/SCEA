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
"""
radius_func_sigmas_threshold = 0.5
noise_levels = np.arange(0, 8.1, 0.1)  # Standard deviations of the noise
n_repetitions_per_noise_level = 16
stds = np.arange(2.5, 5.3, 0.1)
data_set = "linear_polar"
output_filename_appendix = ""
"""
# Test parameters
radius_func_sigmas_threshold = 0.5
noise_levels = np.arange(0, 3.1, 1.)  # Standard deviations of the noise
n_repetitions_per_noise_level = 16
stds = np.arange(2.5, 5.3, 1.)
data_set = "linear_polar"
output_filename_appendix = "_test"


# ==================================
# Paths
# ==================================
"""
current_directory = Path(Path.cwd())
parent_directory = current_directory.parent
OUTPUT_PATH = Path("/Users/eliaserv/Documents/VSCode_projects/SCEA/outputs")
SCRIPTS_PATH = "/Users/eliaserv/Documents/VSCode_projects/SCEA/SCEA"
# plots_path = Path("/Users/eliaserv/Documents/VSCode_projects/SCEA/plots")
"""
OUTPUT_PATH = Path("/scratch/project_2008159/eliaserv/multicluster_experiment/outputs")
SCRIPTS_PATH = "/projappl/project_2008159/eliaserv"



filename = (
    "scores_per_stds_noise"
    + "_"
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
    clusters = SCEA.SCEA(
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
    rng = np.random.default_rng()  # Automatically uses entropy source
    vals_noisy = values + rng.normal(0, noise_sigma, size=values.shape)
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

    pixelwise_f1_global_array = stack("pixelwise_f1_global")
    pixelwise_precision_global_array = stack("pixelwise_precision_global")
    pixelwise_recall_global_array = stack("pixelwise_recall_global")
    pixelwise_recall_per_true_cluster_array = stack(
        "pixelwise_recall_per_true_cluster"
    )  # shape = (n_reps, n_labels)
    pixelwise_precision_per_found_cluster_array = np.empty(n_repetitions_per_noise_level, dtype=object)  # variable length array, so we need to stack it differently
    for i in range(n_repetitions_per_noise_level):
        pixelwise_precision_per_found_cluster_array[i] = all_metrics[i][
            "pixelwise_precision_per_found_cluster"
        ]  # each entry is a variable length array
    clusterwise_precision_array = stack("clusterwise_precision")
    clusterwise_recall_array = stack("clusterwise_recall")
    clusterwise_f1_array = stack("clusterwise_f1")
    tp_true_clusters_array = stack("tp_true_clusters")
    fn_true_clusters_array = stack("fn_true_clusters")
    tp_found_clusters_array = stack("tp_found_clusters")
    fp_found_clusters_array = stack("fp_found_clusters")
    n_found_clusters_in_true_cluster_array = stack("n_found_clusters_in_true_cluster")


    print(f"pixelwise_f1_global_array sample: {pixelwise_f1_global_array[:4]} with noise_level={noise_sigma}  std={std}")

    # Finally compute means (and stds if you like)
    pixelwise_f1_global = pixelwise_f1_global_array.mean()
    pixelwise_precision_global = pixelwise_precision_global_array.mean()
    pixelwise_recall_global = pixelwise_recall_global_array.mean()
    #pixelwise_recall_per_true_cluster = pixelwise_recall_per_true_cluster_array.mean(axis=0)
    #pixelwise_precision_per_found_cluster = (pixelwise_precision_per_found_cluster_array.mean(axis=0))
    clusterwise_precision = clusterwise_precision_array.mean()
    clusterwise_recall = clusterwise_recall_array.mean()
    clusterwise_f1 = clusterwise_f1_array.mean()
    tp_true_clusters = tp_true_clusters_array.mean()
    fn_true_clusters = fn_true_clusters_array.mean()
    tp_found_clusters = tp_found_clusters_array.mean()
    fp_found_clusters = fp_found_clusters_array.mean()
    n_found_clusters_in_true_cluster = n_found_clusters_in_true_cluster_array.mean()

    # Standard deviation of the metrics
    # Finally compute means (and stds if you like)
    pixelwise_f1_global_std = pixelwise_f1_global_array.std()
    pixelwise_precision_global_std = pixelwise_precision_global_array.std()
    pixelwise_recall_global_std = pixelwise_recall_global_array.std()
    #pixelwise_recall_per_true_cluster = pixelwise_recall_per_true_cluster_array.mean(axis=0)
    #pixelwise_precision_per_found_cluster = (pixelwise_precision_per_found_cluster_array.mean(axis=0))
    clusterwise_precision_std = clusterwise_precision_array.std()
    clusterwise_recall_std = clusterwise_recall_array.std()
    clusterwise_f1_std = clusterwise_f1_array.std()
    tp_true_clusters_std = tp_true_clusters_array.std()
    fn_true_clusters_std = fn_true_clusters_array.std()
    tp_found_clusters_std = tp_found_clusters_array.std()
    fp_found_clusters_std = fp_found_clusters_array.std()
    n_found_clusters_in_true_cluster_std = n_found_clusters_in_true_cluster_array.std()

    return {
        "pixelwise_f1_global": pixelwise_f1_global,
        "pixelwise_precision_global": pixelwise_precision_global,
        "pixelwise_recall_global": pixelwise_recall_global,
        "pixelwise_recall_per_true_cluster": pixelwise_recall_per_true_cluster_array,
        "pixelwise_precision_per_found_cluster": pixelwise_precision_per_found_cluster_array,
        "clusterwise_precision": clusterwise_precision,
        "clusterwise_recall": clusterwise_recall,
        "clusterwise_f1": clusterwise_f1,
        "tp_true_clusters": tp_true_clusters,
        "fn_true_clusters": fn_true_clusters,
        "tp_found_clusters": tp_found_clusters,
        "fp_found_clusters": fp_found_clusters,
        "n_found_clusters_in_true_cluster": n_found_clusters_in_true_cluster,
        "pixelwise_f1_global_std": pixelwise_f1_global_std,
        "pixelwise_precision_global_std": pixelwise_precision_global_std,
        "pixelwise_recall_global_std": pixelwise_recall_global_std,
        "clusterwise_precision_std": clusterwise_precision_std,
        "clusterwise_recall_std": clusterwise_recall_std,
        "clusterwise_f1_std": clusterwise_f1_std,
        "tp_true_clusters_std": tp_true_clusters_std,
        "fn_true_clusters_std": fn_true_clusters_std,
        "tp_found_clusters_std": tp_found_clusters_std,
        "fp_found_clusters_std": fp_found_clusters_std,
        "n_found_clusters_in_true_cluster_std": n_found_clusters_in_true_cluster_std,
    }


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
    results = {
        "pixelwise_f1_global": np.zeros((len(stds), len(noise_levels))),
        "pixelwise_precision_global": np.zeros((len(stds), len(noise_levels))),
        "pixelwise_recall_global": np.zeros((len(stds), len(noise_levels))),
        "pixelwise_recall_per_true_cluster": np.zeros((len(stds), len(noise_levels),n_repetitions_per_noise_level, len(np.unique(labels)) - 1)), 
        "pixelwise_precision_per_found_cluster": np.empty((len(stds), len(noise_levels),n_repetitions_per_noise_level), dtype=object), # variable length array
        "clusterwise_precision": np.zeros((len(stds), len(noise_levels))),
        "clusterwise_recall": np.zeros((len(stds), len(noise_levels))),
        "clusterwise_f1": np.zeros((len(stds), len(noise_levels))),
        "tp_true_clusters": np.zeros((len(stds), len(noise_levels))),
        "fn_true_clusters": np.zeros((len(stds), len(noise_levels))),
        "tp_found_clusters": np.zeros((len(stds), len(noise_levels))),
        "fp_found_clusters": np.zeros((len(stds), len(noise_levels))),
        "n_found_clusters_in_true_cluster": np.zeros((len(stds), len(noise_levels))),
        "pixelwise_f1_global_std": np.zeros((len(stds), len(noise_levels))),
        "pixelwise_precision_global_std": np.zeros((len(stds), len(noise_levels))),
        "pixelwise_recall_global_std": np.zeros((len(stds), len(noise_levels))),
        "clusterwise_precision_std": np.zeros((len(stds), len(noise_levels))),
        "clusterwise_recall_std": np.zeros((len(stds), len(noise_levels))),
        "clusterwise_f1_std": np.zeros((len(stds), len(noise_levels))),
        "tp_true_clusters_std": np.zeros((len(stds), len(noise_levels))),
        "fn_true_clusters_std": np.zeros((len(stds), len(noise_levels))),
        "tp_found_clusters_std": np.zeros((len(stds), len(noise_levels))),
        "fp_found_clusters_std": np.zeros((len(stds), len(noise_levels))),
        "n_found_clusters_in_true_cluster_std": np.zeros((len(stds), len(noise_levels))),
        "stds": stds,
        "noise_levels": noise_levels,
        "n_repetitions_per_noise_level": n_repetitions_per_noise_level,
        "radius_func_sigmas_threshold": radius_func_sigmas_threshold,
        "data_set": data_set,
    }    

    # Iterate over the parameter std
    for i, std in enumerate(
        tqdm(
            stds,
            mininterval=10,
            maxinterval=120,
            ncols=80,
            ascii=True,
            dynamic_ncols=False,
            bar_format="{l_bar}{bar}{r_bar}\n",
            leave=True,
        )
    ):
        tqdm.write(f"Processing std = {std}")

        # Iterate over noise levels
        for j, noise_sigma in enumerate(noise_levels):
            # tqdm.write(f"Processing noise level = {noise_sigma}")

            # Run the experiment with repetitions
            metrics = single_experiment_with_repetitions(
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
            results["pixelwise_f1_global"][i, j] = metrics["pixelwise_f1_global"]
            results["pixelwise_precision_global"][i, j] = (metrics["pixelwise_precision_global"])
            results["pixelwise_recall_global"][i, j] = metrics["pixelwise_recall_global"]
            results["pixelwise_recall_per_true_cluster"][i, j] = (metrics["pixelwise_recall_per_true_cluster"])
            results["pixelwise_precision_per_found_cluster"][i, j] = (metrics["pixelwise_precision_per_found_cluster"])
            results["clusterwise_precision"][i, j] = metrics["clusterwise_precision"]
            results["clusterwise_recall"][i, j] = metrics["clusterwise_recall"]
            results["clusterwise_f1"][i, j] = metrics["clusterwise_f1"]
            results["tp_true_clusters"][i, j] = metrics["tp_true_clusters"]
            results["fn_true_clusters"][i, j] = metrics["fn_true_clusters"]
            results["tp_found_clusters"][i, j] = metrics["tp_found_clusters"]
            results["fp_found_clusters"][i, j] = metrics["fp_found_clusters"]
            results["n_found_clusters_in_true_cluster"][i, j] = (
                metrics["n_found_clusters_in_true_cluster"]
            )
            # Store the standard deviations for this run
            results["pixelwise_f1_global_std"][i, j] = metrics["pixelwise_f1_global_std"]
            results["pixelwise_precision_global_std"][i, j] = (
                metrics["pixelwise_precision_global_std"]
            )
            results["pixelwise_recall_global_std"][i, j] = (
                metrics["pixelwise_recall_global_std"]
            )
            results["clusterwise_precision_std"][i, j] = (
                metrics["clusterwise_precision_std"]
            )
            results["clusterwise_recall_std"][i, j] = (
                metrics["clusterwise_recall_std"]
            )
            results["clusterwise_f1_std"][i, j] = metrics["clusterwise_f1_std"]
            results["tp_true_clusters_std"][i, j] = metrics["tp_true_clusters_std"]
            results["fn_true_clusters_std"][i, j] = metrics["fn_true_clusters_std"]
            results["tp_found_clusters_std"][i, j] = metrics["tp_found_clusters_std"]
            results["fp_found_clusters_std"][i, j] = metrics["fp_found_clusters_std"]
            results["n_found_clusters_in_true_cluster_std"][i, j] = (
                metrics["n_found_clusters_in_true_cluster_std"]
            )
            

    # Save the results to a file
    print(f"Saving results to {OUTPUT_PATH / filename}")
    np.savez(OUTPUT_PATH / filename, **results)

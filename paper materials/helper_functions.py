import numpy as np
import pandas as pd
#import SCEA
import matplotlib.pyplot as plt
#from matplotlib.patches import Polygon
#import cmcrameri.cm as cmc
#from matplotlib import colormaps
#from tqdm import tqdm
#import datetime
#import os
#import sys
#from pathlib import Path



##### Functions #####


def classification_outcomes(clusters, labels, reutrn_counts=False):
    is_clustered = clusters > 0
    labels = labels.astype(bool)
    is_true_positive = np.logical_and(is_clustered, labels)
    is_false_positive = np.logical_and(is_clustered, np.logical_not(labels))
    is_true_negative = np.logical_and(
        np.logical_not(is_clustered), np.logical_not(labels)
    )
    is_false_negative = np.logical_and(np.logical_not(is_clustered), labels)
    if reutrn_counts:
        return (
            is_true_positive.sum(),
            is_false_positive.sum(),
            is_true_negative.sum(),
            is_false_negative.sum(),
        )
    return is_true_positive, is_false_positive, is_true_negative, is_false_negative


def calculate_metrics_for_multiclusters(
    clusters: np.ndarray,
    labels: np.ndarray,
    verbose: bool = False,
    thresh_precision: float = 0.5,
    thresh_recall: float = 0.5,
) -> tuple:
    """ TODO elaborate
    Calculate various metrics for multi-cluster evaluation.
    Parameters:
    - clusters: np.ndarray
    - labels: np.ndarray
    - verbose: bool
    - thresh_precision: float
    - thresh_recall: float
    Returns:
    - dict
        Dictionary containing various metrics.
    """

    # Each cluster is a unique label int > 0
    unique_found_clusters = np.unique(clusters[clusters > 0])

    # Calculating the pixel-wise recall for each true cluster
    # Initialize
    pixelwise_recall_per_true_cluster = np.zeros(labels.max(), dtype=float)
    n_found_clusters_per_true_cluster = np.zeros(labels.max(), dtype=int)

    # Iterate over each true cluster
    for true_label in np.unique(labels):
        if true_label == 0:
            continue  # Skip the background label
        
        # Pixel-wise recall for this true cluster
        clusters_in_true_label = clusters[labels == true_label]
        pixelwise_tp = np.sum(clusters_in_true_label > 0)
        pixelwise_fp = np.sum(clusters_in_true_label == 0)
        pixelwise_recall = pixelwise_tp / (pixelwise_tp + pixelwise_fp) if (pixelwise_tp + pixelwise_fp) else 0
        pixelwise_recall_per_true_cluster[int(true_label - 1)] = pixelwise_recall

        n_found_clusters_per_true_cluster = np.sum(clusters_in_true_label > 0)


    # Calculating the cluster-wise recall
    ## How many of the true clusters are found?
    ## - TP: Count a true cluster as true positive if its pixel-wise recall is greater than t_r.
    ## - FN: Count a true cluster as false negative if its pixel-wise recall is less than t_r.
    tp_true_clusters = np.sum(pixelwise_recall_per_true_cluster>=thresh_recall)
    fn_true_clusters = np.sum(pixelwise_recall_per_true_cluster<thresh_recall)
    clusterwise_recall = (
        tp_true_clusters / (tp_true_clusters + fn_true_clusters)
        if (tp_true_clusters + fn_true_clusters)
        else 0
    )



    # Calcluating the pixel-wise precision for each found cluster
    # Initialize, recall for each true cluster
    if unique_found_clusters.size != 0:
        pixelwise_precision_per_found_cluster = np.zeros(int(unique_found_clusters.max()), dtype=float)

        # Iterate over each found cluster
        for found_cluster in unique_found_clusters:
            
            # Pixel-wise precision for this found cluster
            is_found_cluster = clusters == found_cluster
            tp, fp, tn, fn = classification_outcomes(
                clusters=is_found_cluster, labels=(labels > 0), reutrn_counts=True
            )
            pixelwise_precision = tp / (tp + fp) if (tp + fp) else 0
            pixelwise_precision_per_found_cluster[int(found_cluster-1)] = pixelwise_precision
    else:
        pixelwise_precision_per_found_cluster = np.zeros(0, dtype=float)

    # Calculating the cluster-wise precision
    ## How many of the found clusters are co-located with a true cluster?
    ## - TP: Count a found cluster as true positive if it's pixel-wise precision is greater than t_p'.
    ## - FP: Count a found cluster as false positive if it's pixel-wise precision is less than t_p'.
    ## notice that the TP is not the same as the TP in the recall caclulation
    tp_found_clusters = np.sum(pixelwise_precision_per_found_cluster>=thresh_precision)
    fp_found_clusters = np.sum(pixelwise_precision_per_found_cluster<thresh_precision)
    clusterwise_precision = (
        tp_found_clusters / (tp_found_clusters + fp_found_clusters)
        if (tp_found_clusters + fp_found_clusters)
        else 0
    )

    # Cluster-wise F1 score
    clusterwise_f1 = (
        2 * clusterwise_precision * clusterwise_recall
        / (clusterwise_precision + clusterwise_recall)
        if (clusterwise_precision + clusterwise_recall)
        else 0
    )

    # Pixel-wise F1 score, precision, and recall in general
    tp, fp, tn, fn = classification_outcomes(
        clusters > 0, labels > 0, reutrn_counts=True
    )
    pixelwise_f1_global = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0
    pixelwise_precision_global = tp / (tp + fp) if (tp + fp) else 0
    pixelwise_recall_global = tp / (tp + fn) if (tp + fn) else 0


    # Average and median size of each true found cluster
    #TODO?
    """
    true_found_clusters_sizes = [
        np.sum(clusters == true_found_cluster)
        for true_found_cluster in tp_clusters
    ]
    mean_true_positive_clusters_size = np.mean(true_found_clusters_sizes) if true_found_clusters_sizes else 0
    median_true_positive_clusters_size = np.median(true_found_clusters_sizes) if true_found_clusters_sizes else 0
    """

    # Average and median size of each false found cluster
    #TODO?
    """
    false_found_clusters_sizes = [
        np.sum(clusters == false_found_cluster)
        for false_found_cluster in fp_clusters
    ]
    mean_false_positive_clusters_size = np.mean(false_found_clusters_sizes) if false_found_clusters_sizes else 0
    median_false_positive_clusters_size = np.median(false_found_clusters_sizes) if false_found_clusters_sizes else 0
    """



    #if verbose: TODO
        #print(f"F1 score for whole data: {pixelwise_f1_global:.2f}")
        #print(f"Precision for whole data: {pixelwise_precision_global:.2f}")
        #print(f"Recall for whole data: {pielwise_recall_global:.2f}")
        #print(f"F1 score for each true cluster: {[f'{x:.2f}' for x in pixelwise_recall_per_true_cluster]}")
        
        #print(
        #    f"Number of unique found clusters in each true cluster: {n_found_clusters_in_true_cluster}"
        #)
        #print(f"Number of true found clusters: {n_tp_clusters}")
        #print(f"Number of false found clusters: {n_fp_clusters}")
        #print(
        #    f"Size of each true found cluster. Avg: {mean_true_positive_clusters_size:.2f}, Median: {median_true_positive_clusters_size:.2f}"
        #)
        #print(
        #    f"Size of each false found cluster. Avg: {mean_false_positive_clusters_size:.2f}, Median: {median_false_positive_clusters_size:.2f}"
        #)
    
    return {
        "pixelwise_f1_global": pixelwise_f1_global,
        "pixelwise_precision_global": pixelwise_precision_global,
        "pixelwise_recall_global": pixelwise_recall_global,
        "pixelwise_recall_per_true_cluster": pixelwise_recall_per_true_cluster,
        "pixelwise_precision_per_found_cluster": pixelwise_precision_per_found_cluster,
        "clusterwise_precision": clusterwise_precision,
        "clusterwise_recall": clusterwise_recall,   
        "clusterwise_f1": clusterwise_f1,
        "tp_true_clusters": tp_true_clusters, # do we need these?
        "fn_true_clusters": fn_true_clusters, # do we need these?
        "tp_found_clusters": tp_found_clusters, # do we need these? 
        "fp_found_clusters": fp_found_clusters, # do we need these?
        "n_found_clusters_in_true_cluster": n_found_clusters_per_true_cluster,
    }





def plot_clusters(points, values, clusters, labels, **kwargs):
    """
    Plots the points and clusters with customizable plotting parameters.

    Parameters:
    - points: array-like, shape (n_samples, 2)
        The coordinates of the points.
    - values: array-like, shape (n_samples,)
        The values associated with the points (for coloring).
    - clusters: array-like, shape (n_samples,)
        The cluster labels for the points.
    - kwargs: additional keyword arguments for customization.
    """
    fig, axs = plt.subplots(1, 3, figsize=(21, 7))
    axs = axs.flatten()

    # Plot the original data
    scatter1 = axs[0].scatter(points[:, 0], points[:, 1], c=values, **kwargs)
    axs[0].set_title("Original Data")
    axs[0].set_xlabel("X")
    axs[0].set_ylabel("Y")
    axs[0].set_aspect("equal")
    # cbar1 = fig.colorbar(scatter1, ax=axs[0])
    # cbar1.set_label('Values')

    # Plot the clusters
    scatter2 = axs[1].scatter(points[:, 0], points[:, 1], c=values, **kwargs)
    unique_clusters = np.unique(clusters)
    for cluster in unique_clusters:
        if cluster == 0:
            continue
        cluster_points = points[clusters == cluster]
        axs[1].scatter(
            cluster_points[:, 0],
            cluster_points[:, 1],
            label=f"Cluster {cluster}",
            **kwargs,
        )
    axs[1].set_title(f"Found {int(unique_clusters.max())} Clusters")
    axs[1].set_xlabel("X")
    # axs[1].set_ylabel("Y")
    axs[1].set_aspect("equal")
    # cbar2 = fig.colorbar(scatter2, ax=axs[1])
    # cbar2.set_label('Values')
    # axs[1].legend()
    # Create a colorbar with its own axis

    scatter1 = axs[2].scatter(points[:, 0], points[:, 1], c=values, **kwargs)
    axs[2].set_title("Original Data")
    axs[2].set_xlabel("X")
    # axs[2].set_ylabel("Y")
    axs[2].set_aspect("equal")

    is_true_positive, is_false_positive, is_true_negative, is_false_negative = (
        classification_outcomes(clusters, labels)
    )
    axs[2].scatter(
        points[is_true_positive, 0],
        points[is_true_positive, 1],
        c="g",
        label="True Positive",
        **kwargs,
    )
    axs[2].scatter(
        points[is_false_positive, 0],
        points[is_false_positive, 1],
        c="r",
        label="False Positive",
        **kwargs,
    )
    # axs[2].scatter(points[true_negative, 0], points[true_negative, 1], c='b', label='True Negative')
    axs[2].scatter(
        points[is_false_negative, 0],
        points[is_false_negative, 1],
        c="y",
        label="False Negative",
        **kwargs,
    )
    axs[2].legend()

    # Plot text box with metrics
    tp = is_true_positive.sum()
    fp = is_false_positive.sum()
    tn = is_true_negative.sum()
    fn = is_false_negative.sum()
    accuracy = (tp + tn) / (tp + fp + tn + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    textstr = f"Accuracy: {accuracy:.2f}\nPrecision: {precision:.2f}\nRecall: {recall:.2f}\nF1 Score: {f1:.2f}"
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
    axs[2].text(
        0.05,
        0.95,
        textstr,
        transform=axs[2].transAxes,
        fontsize=14,
        verticalalignment="top",
        bbox=props,
    )

    cbar = fig.colorbar(
        scatter1, ax=axs, orientation="vertical", fraction=0.02, pad=0.04
    )

    plt.show()

def plot_clusters_imshow(points, values, clusters, labels, **kwargs):
    """
    Plots the points and clusters using imshow so that each (x, y, value) becomes one pixel.

    Subplots:
      1. Original Data only.
      2. Clusters overlay on top of the original image (only nonzero clusters, using a discrete colormap).
      3. Classification outcomes overlay on the original image.
         True Positive = green, False Positive = red, False Negative = yellow.
         A legend is provided for these outcomes.

    One common colorbar (referring to the original image values) is shown for all subplots.

    Parameters:
    - points: array-like, shape (n_samples, 2)
        Coordinates of the points.
    - values: array-like, shape (n_samples,)
        Values associated with the points (for coloring the original image).
    - clusters: array-like, shape (n_samples,)
        Cluster labels for the points (0-cluster is not plotted).
    - labels: array-like, shape (n_samples,)
        True labels used for computing classification outcomes.
    - kwargs: additional keyword arguments (not used in this imshow version)

    Note: Assumes that a function `classification_outcomes(clusters, labels)` exists
    that returns (is_true_positive, is_false_positive, is_true_negative, is_false_negative).
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from matplotlib.patches import Patch

    # Determine grid dimensions from point coordinates.
    unique_x = np.unique(points[:, 0])
    unique_y = np.unique(points[:, 1])
    width = len(unique_x)
    height = len(unique_y)
    
    # Create a mapping from coordinates to grid indices.
    x_to_index = {x: idx for idx, x in enumerate(unique_x)}
    y_to_index = {y: idx for idx, y in enumerate(unique_y)}

    # Initialize images (2D arrays) with NaN so that missing pixels remain blank.
    img_orig = np.full((height, width), np.nan)
    img_clusters = np.full((height, width), np.nan)
    img_class = np.full((height, width), np.nan)

    # Populate the image arrays.
    for i in range(points.shape[0]):
        col = x_to_index[points[i, 0]]
        row = y_to_index[points[i, 1]]
        # Original image: assign the value.
        img_orig[row, col] = values[i]
        # Clusters: only assign if cluster is nonzero.
        if clusters[i] != 0:
            img_clusters[row, col] = clusters[i]

    # Compute classification outcomes.
    is_tp, is_fp, is_tn, is_fn = classification_outcomes(clusters, labels)

    # Populate classification outcomes image:
    # 1 = True Positive, 2 = False Positive, 3 = False Negative.
    # True Negatives are left as NaN.
    for i in range(points.shape[0]):
        col = x_to_index[points[i, 0]]
        row = y_to_index[points[i, 1]]
        if is_tp[i]:
            img_class[row, col] = 1
        elif is_fp[i]:
            img_class[row, col] = 2
        elif is_fn[i]:
            img_class[row, col] = 3

    # Create figure and subplots.
    fig, axs = plt.subplots(1, 3, figsize=(21, 7))

    # --- Subplot 1: Original Data only ---
    im0 = axs[0].imshow(img_orig, origin="lower")
    axs[0].set_title("Original Data")
    axs[0].set_xlabel("X")
    axs[0].set_ylabel("Y")

    # --- Subplot 2: Clusters overlay ---
    # First, plot the original image.
    axs[1].imshow(img_orig, origin="lower")
    # Overlay clusters using a discrete colormap (e.g. 'tab10').
    axs[1].imshow(img_clusters, origin="lower", cmap="tab20", alpha=0.7)
    # Count clusters (ignoring 0).
    unique_clusters = np.unique(clusters[clusters != 0])
    cluster_count = int(unique_clusters.max()) if unique_clusters.size > 0 else 0
    axs[1].set_title(f"Clusters (Found {cluster_count})")
    axs[1].set_xlabel("X")

    # --- Subplot 3: Classification Outcomes overlay ---
    # First, plot the original image.
    axs[2].imshow(img_orig, origin="lower")
    # Define a discrete colormap for classification outcomes:
    # 1: True Positive (green), 2: False Positive (red), 3: False Negative (yellow)
    class_colors = ["green", "red", "yellow"]
    cmap_class = ListedColormap(class_colors)
    norm_class = BoundaryNorm([0.5, 1.5, 2.5, 3.5], len(class_colors))
    axs[2].imshow(
        img_class, origin="lower", cmap=cmap_class, norm=norm_class, alpha=0.7
    )
    axs[2].set_title("Classification Outcomes")
    axs[2].set_xlabel("X")
    # Create a legend for the classification outcomes.
    legend_elements = [
        Patch(facecolor="green", edgecolor="black", label="True Positive"),
        Patch(facecolor="red", edgecolor="black", label="False Positive"),
        Patch(facecolor="yellow", edgecolor="black", label="False Negative"),
    ]
    axs[2].legend(handles=legend_elements, loc="upper left", bbox_to_anchor=(1, 1))

    # Optionally, compute metrics (ignoring true negatives) and add a text box in subplot 3.
    tp = is_tp.sum()
    fp = is_fp.sum()
    fn = is_fn.sum()
    total = tp + fp + fn
    accuracy = (tp) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (
        2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    )
    textstr = (
        f"Accuracy: {accuracy:.2f}\nPrecision: {precision:.2f}\n"
        f"Recall: {recall:.2f}\nF1 Score: {f1:.2f}"
    )
    props = dict(boxstyle="round", facecolor="white", alpha=0.5)
    axs[2].text(
        1.03,
        0.75,
        textstr,
        transform=axs[2].transAxes,
        fontsize=12,
        verticalalignment="top",
        bbox=props,
    )

    # --- Common Colorbar ---
    # Create one colorbar for the entire figure (using the original image from subplot 1).
    cbar = fig.colorbar(im0, ax=axs, orientation="vertical", fraction=0.02, pad=0.13)
    cbar.set_label("Original Image Values")

    plt.show()

def plot_clusters_multicluster(points, values, clusters, labels, **kwargs):
    """
    Plots the points and clusters using imshow so that each (x, y, value) becomes one pixel.

    Subplots:
      1. Original Data only.
      2. Clusters overlay on top of the original image (only nonzero clusters, using a discrete colormap).
      3. Classification outcomes overlay on the original image.
         True Positive = green, False Positive = red, False Negative = yellow.
         A legend is provided for these outcomes.

    One common colorbar (referring to the original image values) is shown for all subplots.

    Parameters:
    - points: array-like, shape (n_samples, 2)
        Coordinates of the points.
    - values: array-like, shape (n_samples,)
        Values associated with the points (for coloring the original image).
    - clusters: array-like, shape (n_samples,)
        Cluster labels for the points (0-cluster is not plotted).
    - labels: array-like, shape (n_samples,)
        True labels used for computing classification outcomes.
    - kwargs: additional keyword arguments (not used in this imshow version)

    Note: Assumes that a function `classification_outcomes(clusters, labels)` exists
    that returns (is_true_positive, is_false_positive, is_true_negative, is_false_negative).
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from matplotlib.patches import Patch

    # Determine grid dimensions from point coordinates.
    unique_x = np.unique(points[:, 0])
    unique_y = np.unique(points[:, 1])
    width = len(unique_x)
    height = len(unique_y)
    
    # Create a mapping from coordinates to grid indices.
    x_to_index = {x: idx for idx, x in enumerate(unique_x)}
    y_to_index = {y: idx for idx, y in enumerate(unique_y)}

    # Initialize images (2D arrays) with NaN so that missing pixels remain blank.
    img_orig = np.full((height, width), np.nan)
    img_clusters = np.full((height, width), np.nan)
    img_class = np.full((height, width), np.nan)

    # Populate the image arrays.
    for i in range(points.shape[0]):
        col = x_to_index[points[i, 0]]
        row = y_to_index[points[i, 1]]
        # Original image: assign the value.
        img_orig[row, col] = values[i]
        # Clusters: only assign if cluster is nonzero.
        if clusters[i] != 0:
            img_clusters[row, col] = clusters[i]

    # Compute classification outcomes.
    is_tp, is_fp, is_tn, is_fn = classification_outcomes(clusters, labels)

    metrics_dict = calculate_metrics_for_multiclusters(clusters, labels, verbose=False)

    f1_general = metrics_dict["f1_general"]
    precision_general = metrics_dict["precision_general"]
    recall_general = metrics_dict["recall_general"]
    f1_scores = metrics_dict["f1_scores"]
    n_found_clusters_in_true_cluster = metrics_dict["n_found_clusters_in_true_cluster"]
    n_true_found_clusters = metrics_dict["n_true_found_clusters"]
    n_false_found_clusters = metrics_dict["n_false_found_clusters"]
    n_false_negative_clusters = metrics_dict["n_false_negative_clusters"]
    f1_scores_whole_clusters = metrics_dict["f1_scores_whole_clusters"]
    mean_true_positive_clusters_size_array = metrics_dict["mean_true_positive_clusters_size_array"]
    median_true_positive_clusters_size = metrics_dict["median_true_positive_clusters_size"]
    mean_false_positive_clusters_size = metrics_dict["mean_false_positive_clusters_size"]
    median_false_positive_clusters_size = metrics_dict["median_false_positive_clusters_size"]



    # Populate classification outcomes image:
    # 1 = True Positive, 2 = False Positive, 3 = False Negative.
    # True Negatives are left as NaN.
    for i in range(points.shape[0]):
        col = x_to_index[points[i, 0]]
        row = y_to_index[points[i, 1]]
        if is_tp[i]:
            img_class[row, col] = 1
        elif is_fp[i]:
            img_class[row, col] = 2
        elif is_fn[i]:
            img_class[row, col] = 3

    # Create figure and subplots.
    fig, axs = plt.subplots(1, 3, figsize=(21, 6))

    # --- Subplot 1: Original Data only ---
    im0 = axs[0].imshow(img_orig, origin="lower")
    axs[0].set_title("Original Data")
    axs[0].set_xlabel("X")
    axs[0].set_ylabel("Y")

    # --- Subplot 2: Clusters overlay ---
    
    # First, plot the original image.
    axs[1].imshow(img_orig, origin="lower")
    # Overlay clusters using a discrete colormap (e.g. 'tab10').
    axs[1].imshow(img_clusters, origin="lower", cmap="tab20")
    # Count clusters (ignoring 0).
    unique_clusters = np.unique(clusters[clusters != 0])
    cluster_count = int(unique_clusters.max()) if unique_clusters.size > 0 else 0
    axs[1].set_title(f"Clusters (Found {cluster_count})")
    axs[1].set_xlabel("X")
    

    
    # --- Subplot 3: Classification Outcomes overlay ---
    # First, plot the original image.
    axs[2].imshow(img_orig, origin="lower", alpha=0.8)
    # Define a discrete colormap for classification outcomes:
    # 1: True Positive (green), 2: False Positive (red), 3: False Negative (yellow)
    class_colors = ["green", "red", "orange"]
    cmap_class = ListedColormap(class_colors)
    norm_class = BoundaryNorm([0.5, 1.5, 2.5, 3.5], len(class_colors))
    axs[2].imshow(
        img_class, origin="lower", cmap=cmap_class, norm=norm_class, alpha=0.9
    )
    axs[2].set_title("Classification Outcomes")
    axs[2].set_xlabel("X")
    # Create a legend for the classification outcomes.
    legend_elements = [
        Patch(facecolor="green", edgecolor="black", label="True Positive"),
        Patch(facecolor="red", edgecolor="black", label="False Positive"),
        Patch(facecolor="orange", edgecolor="black", label="False Negative"),
    ]
    axs[2].legend(handles=legend_elements, loc="upper left", bbox_to_anchor=(1, 1.03))

    # Optionally, compute metrics (ignoring true negatives) and add a text box in subplot 3.
    tp = is_tp.sum()
    fp = is_fp.sum()
    fn = is_fn.sum()
    total = tp + fp + fn
    accuracy = (tp) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (
        2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    )
    textstr = (
        f"F1 for whole data pointwise: {f1_general:.2f}\n"
        f"Precision for whole data: {precision_general:.2f}\n"
        f"Recall for whole data: {recall_general:.2f}\n"
        f"N. clusters in true areas : {n_true_found_clusters}\n"
        f"N. clusters not in true areas: {n_false_found_clusters}\n"
        f"N. clusters not found: {n_false_negative_clusters}\n"
        f"F1 cluster-wise: {f1_scores_whole_clusters:.2f}\n"
        f"True cluster size: mean: {mean_true_positive_clusters_size_array:.1f}, median: {median_true_positive_clusters_size:.1f}\n"
        f"False cluster size: mean: {mean_false_positive_clusters_size:.1f}, median: {median_false_positive_clusters_size:.1f}"
    )
    props = dict(boxstyle="round", facecolor="white", alpha=0.5)
    axs[2].text(
        1.03,
        0.78,
        textstr,
        transform=axs[2].transAxes,
        fontsize=8,
        verticalalignment="top",
        bbox=props,
    )

    #textstr2 = (
    #    f"F1 score for each true cluster: \n{[f'{x:.2f}' for x in f1_scores]}\n \n"
    #    f"Number of found clusters in each true cluster: \n {n_found_clusters_in_true_cluster}\n"
    #)

    """
    axs[2].text(
        .03,
        1.3,
        textstr2,
        transform=axs[2].transAxes,
        fontsize=8,
        verticalalignment="top",
        bbox=props,
    )
    """

    # --- Common Colorbar ---
    # Create one colorbar for the entire figure (using the original image from subplot 1).
    cbar = fig.colorbar(im0, ax=axs, orientation="vertical", fraction=0.02, pad=0.2)
    cbar.set_label("Original Image Values")

    # Table 

    # Only 2 decimals for the F1 scores
    f1_scores_table = [f"{i+1}:\n {x:.2f}" for i, x in enumerate(f1_scores)]    
    f1_scores_table = np.flipud(np.array(f1_scores_table).reshape(4, 4).T)  
    axs[2].table(
        f1_scores_table,
        bbox=[1.02, 0, 0.5, 0.4],  # [x, y, width, height]
        cellLoc="center",
        colWidths=[0.1, 0.1, 0.1, 0.1],
    )
    
    # Add a custom title for the table
    axs[2].text(
        1.02, 0.41, "F1 Scores Table for each cluster", transform=axs[2].transAxes,
        fontsize=10, fontweight='bold', va='bottom'
    )

    plt.show()


def plot_heatmap(
    scores_per_radius_func_s,
    noise_levels,
    radius_func_sigmas_thresholds,
    title="Heatmap of F1 Scores Across Different Noise Levels and Radius Function Sensitivity parameters",
    y_label="Radius_func_sigma -parameter\n(lower value -> more sensitive)",
    best_param_box_size=90,
    save_to=None,
):
    fig, ax = plt.subplots(1, 1, figsize=(25, 5))

    scores_per_radius_func_s = np.array(scores_per_radius_func_s)
    cax = ax.matshow(scores_per_radius_func_s, vmin=0, vmax=1)
    cbar = fig.colorbar(cax)
    cbar.set_label("Average F1 score")

    # Set ticks for every point
    xticks = np.arange(len(noise_levels))
    yticks = np.arange(len(radius_func_sigmas_thresholds))

    ax.set_xticks(xticks)
    ax.set_yticks(yticks)

    # Set labels for every other tick
    ax.set_xticklabels(
        [
            str(label) if i % 2 == 0 else ""
            for i, label in enumerate(noise_levels.round(4))
        ]
    )
    ax.set_yticklabels(
        [
            str(label) if i % 1 == 0 else ""
            for i, label in enumerate(radius_func_sigmas_thresholds.round(4))
        ]
    )

    # Lower the font size of the tick labels and rotate them
    ax.tick_params(axis="both", which="major", labelsize=9)
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center")
    plt.setp(ax.get_yticklabels(), rotation=0, ha="right")
    # Put the median scores for each parameter in the second y axis labels on the right of the heatmap
    # median_scores_per_parameter = scores_per_radius_func_s.median(axis=1)
    median_scores_per_parameter = np.median(scores_per_radius_func_s, axis=1)
    # Add text annotations for median scores
    max_median_score = median_scores_per_parameter.max()
    for i, median_score in enumerate(median_scores_per_parameter):
        if median_score == max_median_score:
            ax.text(
                len(noise_levels),
                i,
                f"{median_score:.2f}",
                va="center",
                ha="left",
                fontsize=9,
                fontweight="bold",
            )
        else:
            ax.text(
                len(noise_levels),
                i,
                f"{median_score:.2f}",
                va="center",
                ha="left",
                fontsize=9,
            )
    # Add a title on the right for the median values
    ax.text(
        len(noise_levels) + 3.1,
        len(radius_func_sigmas_thresholds) / 2,
        "Median scores for each row",
        va="center",
        rotation=90,
        ha="left",
        fontsize=12,
    )
    ax.scatter(
        np.arange(len(scores_per_radius_func_s[0])),
        np.argmax(scores_per_radius_func_s, axis=0),
        marker="s",
        linewidths=0.6,
        edgecolor="black",
        alpha=0.8,
        s=best_param_box_size,
        facecolor="none",
        label=f"Best parameter in each noise level",
    )
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlabel("Noise level (standard deviation of Gaussian noise)")
    ax.set_ylabel(y_label)
    """
    ax.text(
        - 4.4,
        len(radius_func_sigmas_thresholds) / 2,
        f"(Lower value $\\rightarrow$ more sensitive $\\rightarrow$ bigger cluster)",
        va="center",
        rotation=90,
        ha="left",
        fontsize=7,
    )
    """
    # ax.set_ylabel("Radius_func_sigma -parameter\n$\\small{(lower\\ value\\ \\rightarrow\\ more\\ sensitive)}$")
    ax.set_title(title)
    if save_to:
        plt.savefig(save_to, bbox_inches="tight", transparent=True, format="pdf")
    plt.show()
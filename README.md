# SCEA

This repository includes code for the SCEA algortihm, unsupervised hotspot detection method for irregularly gridded data.

Paper available at: TBA

## The SCEA algorithm implementation
syntax is as follows:

````
clusters = SCEA(
    point_coordinates,  
    point_values,  
    growth_limit=2,  
    detection_limit=3.5, 
    radius_func="default",  
    n_clusters="auto",
    point_value_threshold="stds_from_median",
    distance_matrix="euclidean",
    max_points_in_start_radius=7,
    local_box_size=0,
    verbose=True,
)
````
where arguments are as follows:

| Argument                   | Description                                                                                                 |
|---------------------------:|-------------------------------------------------------------------------------------------------------------|
| point_coordinates         | Array with shape (n_points, n_dimension)                                                                    |
| point_values              | Array with shape (n_points)                                                                                 |
| growth_limit              | Smaller value → larger clusters (controls cluster growth)                                                   |
| detection_limit           | Smaller value → more clusters (controls detection sensitivity)                                              |
| radius_func               | Radius function to use (e.g., "default" or a user-defined callable)                                         |
| n_clusters                | Integer to find a fixed number of clusters, or "auto"                                                        |
| point_value_threshold     | Threshold to stop detecting new clusters. Can be absolute or "stds_from_median" (then detection_limit is used) |
| distance_matrix           | Precomputed distance matrix or "euclidean"                                                                   |
| max_points_in_start_radius| Condition for killing a "radiating point" (max points allowed in start radius)                              |
| local_box_size            | Local area size considered (0 = global)                                                                      |
| verbose                   | If true, print log information                                                                                |

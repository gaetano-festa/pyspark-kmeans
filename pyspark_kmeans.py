#!/usr/bin/env python3

from utils.kmeans_utils import load_data, save_clustering_results, prepare_data,\
    kmeans_scan

from utils.reports import generate_report

scaled_data, features = prepare_data()

# Coalesce the partition to avoid crash due to too many partitions for a single node run (my laptoop).
scaled_data = scaled_data.coalesce(4)

# In a realistic big data scenario the fitting of the model is an expensive task.
# To avoid to refit the model with the chosen k all model are saved in the tmp folder.

tmp_models_dir = 'tmp_models'

# Save the relevant results in a file to choose the optimal number of clusters
clustering_results_file = 'clustering_results.csv'

# Run the k scanning.
centers, silhuette_scores = kmeans_scan(scaled_data, 2, 8, tmp_models_dir)

# Save the clusters centers and the Silhhuette scores for the different k.
save_clustering_results(clustering_results_file, centers, silhuette_scores, features)

generate_report('kmeans_scanning_report.pdf', 'clustering_results.csv')

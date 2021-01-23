#!/usr/bin/env python3

try:
    from utils.kmeans_utils import spark, load_data, save_clustering_results, prepare_data,\
	    kmeans_scan

    from utils.reports import generate_report, generate_email, send_email

    scaled_data, features = prepare_data()

    # Coalesce the partition to avoid crash due to too many partitions for a single node run (my laptoop!).
    scaled_data = scaled_data.coalesce(4)

    # In a realistic big data scenario the fitting of the model is an expensive task.
    # To avoid to refit the model with the chosen k all model are saved in the tmp folder.

    tmp_models_dir = 'tmp_models'

    # Save the relevant results in a file to choose the optimal number of clusters
    clustering_results_file = 'clustering_results.csv'

    # Run the k scanning.
    k_min, k_max = 2, 6
    centers, silhuette_scores = kmeans_scan(scaled_data, k_min, k_max, tmp_models_dir)

    # Save the clusters centers and the Silhhuette scores for the different k.
    save_clustering_results(clustering_results_file, centers, silhuette_scores, features)

    scanning_report_path = 'kmeans_scanning_report.pdf'
    generate_report(scanning_report_path, 'clustering_results.csv')
    
    # Generate the email message
    email_message = generate_email('gaetano@spark.net', # sender
            'gaetano.festa@gmail.com',                  # recipient
            'Kmeans Spark Scanning Done',               # subject
            'Please find enclosed the KMeans scanning report with the results for k from {} to {}'.format(k_min, k_max), # body
               scanning_report_path ) # attachment

    # Send the email assuming a SMTP server is locally configured.
    send_email(email_message)

except:
    # Send an email to notify something is gone wrong.
    # TODO: collect more information on the reason of failure to give some hint where the problem could be
    email_message = generate_email('gaetano.festa@gmail.com', # sender
            'gaetano.festa@gmail.com',                        # recipient
            'ERROR: Kmeans Spark Scanning Failed!',           # subject
            'KMeans Scanning has failed!')                    # body

    # Send the email assuming a SMTP server is locally configured.
    send_email(email_message)

finally:
    # Close Spark Session
    spark.stop()



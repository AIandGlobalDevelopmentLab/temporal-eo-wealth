# Preprocessing

***Note August 15th 2023**: due to constraints imposed by Google Earth Engine on data exports, the script exporting satellite imagery has become excessively slow to run in its current form (estimated >300 h). The code in this directory is what we ran to produce the results in our paper, but for the sake of usability we are currently developing a faster, equivalent setup for exporting and loading images. This script will be included alongside the original one as soon as it is ready.*

This directory contains the code used to export and preprocess the satellite imagery used in our paper, as well as preparing the cross-validation folds and calculating dataset statistics. Please run the scripts in the following order:

0. [0_export_satellite_images.ipynb](0_export_satellite_images.ipynb)

1. [1_preprocess_satellite_images.ipynb](1_preprocess_satellite_images.ipynb)

2. [2_create_folds.ipynb](2_create_folds.ipynb)

3. [3_calculate_band_stats.sh](3_calculate_band_stats.sh)

4. [4_add_features_to_tf_records.sh](4_add_features_to_tf_records.sh)
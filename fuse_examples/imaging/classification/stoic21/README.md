# STOIC21 Classification with Fuse

Example of severe COVID-19 classifier baseline given a Computed-Tomography (CT), age group and gender.
More details about the challenge can be found here: https://stoic2021.grand-challenge.org/

# Data
The STOIC Dataset (see https://pubs.rsna.org/doi/10.1148/radiol.2021210384 for a full description) contains Computed Tomography scans from 10,735 patients.  For this challenge, one CT scan from each patient has been selected, and the dataset has been divided randomly into a public training set (2,000 patients), a test set (~1,000 patients)
and a private training set (7,000+ patients). The training set is available to download in the Grand Challenge while the rest kept private.
More details can be found here: https://stoic2021.grand-challenge.org/stoic-db/

FuseMedML provides an easy to follow implementation of a PyTorch Dataset. The implementation can be found in (fuseimg/datasets/stoic21.py)[https://github.com/IBM/fuse-med-ml/blob/fuse2/fuseimg/datasets/stoic21.py]


# Model
A 3D backbone (ResNet18) followed by an binary classification head based on the imaging features extracted by the backbone and clinical data (age group and gender)

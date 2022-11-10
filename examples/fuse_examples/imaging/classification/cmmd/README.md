# The Chinese Mammography Database (CMMD) with Fuse
**Introduction**

The project presents classification of lesion in breast.
It demonstrates the use of FuseMedML for classification of 2D lesions extracted from Mammography.
In this example, we present a binary classification between benign lesions and malignant lesions .


**Dataset**

The Chinese Mammography Database (CMMD) has been published by The Cancer Imaging Archive (TCIA) in 2021.
We used the public Chinese Mammography Database (CMMD) dataset:

https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId%3D70230508&hl=en&gl=il&strip=1&vwsrc=0

To download the data one should install the NBIA data retriever application, instructions located under:

https://wiki.cancerimagingarchive.net/display/NBIA/Downloading+TCIA+Images#DownloadingTCIAImages-InstallingtheNBIADataRetriever

It includes 3,728 mammography studies from 1,775 patients, acquired between 2012
and 2016, in various Chinese institutions (including Sun Yat-sen University Cancer Center and Nanhai Hospital of Southern Medical University in Foshan).
Dataset images are accompanied by biopsy-proven breast-level benign and malignant labels.
Dataset authors also provided age and finding type (calcification,mass or both) for all patients as well as immunohistochemical markers for 749 patients with invasive carcinoma.
Each patient might provide up to 4 possible Mammography screenings ; Breast side can be either right or left and view can be either be medio-lateral oblique (MLO) or craniocaudal (CC) .

Several classification tasks were explored using this dataset. In this example we present a binary classification between benign lesions and malignant lesions .

**Data Citation**

Cui, Chunyan; Li Li; Cai, Hongmin; Fan, Zhihao; Zhang, Ling; Dan, Tingting; Li, Jiao; Wang, Jinghua. (2021) The Chinese Mammography Database (CMMD): An online mammography database with biopsy confirmed types for machine diagnosis of breast. The Cancer Imaging Archive. DOI: https://doi.org/10.7937/tcia.eqde-4b16


**Pre-processing**

The pre-processing is preformed using the pipeline in cmmd.py.
The static pipeline extracts the relevant breast area from the mammography scan, standardize it to appear to the left and resize it to fit a standard GPU memory.


**CNN**

This project is an implementation on the InceptionResnetV2 as was presented at
"Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning".

**INPUT**

The network is trained using cross-validation.
The data was split into 5 folds insuring no patient is overlapped between the different folds. The data was split in
a way that saves the frequency of the two classes in each of the folds.
The training set was composed of 3 folds , validation set from 1 fold and test set from 1 fold.



**MODEL TRAINING**

Follow these steps to configure the dataset  -

    1. CMMD_clinicaldata_revision.csv which is a converted version of CMMD_clinicaldata_revision.xlsx

    2. Folder named CMMD which is the downloaded data folder

    3. Enviroment variable CMMD_DATA_PATH to the main folder containing 1+2

* In runner.py / config.yaml hydra file:
    - configure the working dirs and file names for the current run
    - specify NUM_GPUS and number of workers to run
    - specify hyperparameters for the run ( learning rate, weight_decay , num of folds and more )
* Run:
    python3 runner.py

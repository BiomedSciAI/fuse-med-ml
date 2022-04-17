# Duke-Breast-Cancer-MRI with Fuse
**Introduction**

The project presents lesions classification of Lesion stage (size) score in breast.
It demonstrates the use of FuseMedML for classification of 3D lesions extracted from multi-MRI which includes several types 
of series imaging.
The size of the tumor is significant in terms of patient treatment and prognosis. Patient with small tumor grade size 
will be treated differently than large tumor patients.  In this example, we present a binary classification between 
small lesions (tumor stage<=1) and large lesions (tumor stage>1).

  
**Dataset**

We used the public Duke-Breast-Cancer-MRI dataset:
https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=70226903
To download the data one should install the NBIA data retriever application, instructions located under:
https://wiki.cancerimagingarchive.net/display/NBIA/Downloading+TCIA+Images#DownloadingTCIAImages-InstallingtheNBIADataRetriever

It contains 922 patients with invasive breast cancer.  
Following MRI sequences are shared: a non-fat saturated T1-weighted sequence, 
a fat-saturated gradient echo T1-weighted pre-contrast sequence, and mostly three to four post-contrast sequences.
Also, Annotation boxes provided by radiologists that indicate locations of the lesions in the images.
In addition, the data contains Demographic, clinical, pathology, treatment, outcomes, and genomic data. 

The train data was split into 5 folds insuring no patient is overlapped between the different folds. The data was split in 
a way that saves the frequency of the two classes in each of the folds.
Several classification tasks were explored using this dataset. In this example we present the Tumor Staging Size label classification.

**Pre-processing**
The pre-processing is preformed using two major functions: processor_dicom_mri.py and processor.py.
The processor_dicom_mri.py extract the relevant MRI sequences and stack them together, registrated by world coordinates 
system and resampled with common resolution.
Then, the processor.py crops the subvolume of the image based on the given lesion mask volume.
The sequences that are in use in this example are DCE - phase2.
The output of the pre-processing is 4d volume of size (MRI series(ch),z,x,y) around lesion location point.

**CNN**

This project is an implementation on the 3D ResNet as was presented at 
"Semi-automatic classification of prostate cancer on multi-parametric MR imaging using a multi-channel 3D convolutional 
neural network". 

**INPUT**

The network is trained using cross-validation. 
In each iteration one fold ('fold_no') is set as validation and the other folds as training data.
The different folds are extracted from dataset.pickle.
The folds data and labels are extracted from Lesion Information inside Clinical_and_Other_Features.csv that can be found in 
the dataset's site. 



**MODEL TRAINING**
* In run_train_3dpatch.py:
    - fill in the code with paths that are labeled as TODO
    - specify fold_no to run
* Run:
    python3 run_train_3dpatch.py   

**Results**

For the Tumor Stage (Size) classification, the average result before adding the tabular features is AUC of 0.766; 
after adding the tabular features the average result was AUC of 0.782    
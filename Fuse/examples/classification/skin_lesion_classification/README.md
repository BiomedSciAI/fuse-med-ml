# Skin Lesion Classification with Fuse 

This project deals with two skin lesion classification challenges, from the International Skin Imaging Collaboration (ISIC):

1. **ISIC 2016 Challenge - Part 3**  
        Training data: 900 images  
        Test data: 350 images  

2. **ISIC 2017 Challenge - Part 3**  
        Training data: 2000 images  
        Validation data: 150 images  
        Test data: 600 images  

In both cases, the goal is to train a model able to classify whether a tumor is benign or malignant.  
Check these links for more details:  
https://challenge.isic-archive.com/landing/2016  
https://challenge.isic-archive.com/landing/2017  

This code provides a simple implementation of these tasks using FuseMedML.  

The actual configuration is built for 2017 dataset.  
The model used is an InceptionResnetV2 pretrained on ImageNet.  
We evaluate it on AUC metric. After 15 epochs, the results expected are around 0.81 of AUC and 0.83 of accuracy on the testset.  

Here are some papers dealing with these challenges:  
Pham, Tri-Cong & Luong, Chi & Visani, Muriel & Dung, Hoang Van. (2018). Deep CNN and Data Augmentation for Skin Lesion Classification. 10.1007/978-3-319-75420-8_54.  
Brinker TJ, Hekler A, Enk AH, von Kalle C (2019) Enhanced classifier training to improve precision of a convolutional neural network to identify images of skin lesions.PLoSONE 14(6):e0218713.  
Hasan, Md & Elahi, Md & Alam, Md. Ashraful. (2021). DermoExpert: Skin lesion classification using a hybrid convolutional neural network through segmentation, transfer learning, and augmentation. 10.1101/2021.02.02.21251038.   

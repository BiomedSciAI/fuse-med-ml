# Skin Lesion Classification with Fuse

This project deals with skin lesion classification challenge, from the International Skin Imaging Collaboration (ISIC).

## ISIC 2019 Challenge

The goal is to train a model which able to classify demoscropic images among nine different diagnostic categories. Check [this link](https://challenge.isic-archive.com/landing/2019/) for more details.

This code provides a simple implementation of these tasks using FuseMedML.
The model used is an InceptionResnetV2 pretrained on ImageNet and we evaluate it on AUC metric.

### Train a model

```sh
python isic_runner.py
```

### Related papers

Here are some papers dealing ISIC challenges from previous years:

Pham, Tri-Cong & Luong, Chi & Visani, Muriel & Dung, Hoang Van. (2018). Deep CNN and Data Augmentation for Skin Lesion Classification. 10.1007/978-3-319-75420-8_54.
Brinker TJ, Hekler A, Enk AH, von Kalle C (2019) Enhanced classifier training to improve precision of a convolutional neural network to identify images of skin lesions.PLoSONE 14(6):e0218713.
Hasan, Md & Elahi, Md & Alam, Md. Ashraful. (2021). DermoExpert: Skin lesion classification using a hybrid convolutional neural network through segmentation, transfer learning, and augmentation. 10.1101/2021.02.02.21251038.

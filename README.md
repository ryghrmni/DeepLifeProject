# Cell Image Segmentation Project

## Overview
Cell segmentation is a crucial step in downstream single-cell analysis in microscopy image-based biology and biomedical research. This project aims to benchmark cell segmentation methods using deep learning techniques. The project utilizes the "2018 Data Science Bowl" Kaggle competition dataset, which provides cell images and their masks for training cell/nuclei segmentation models.

## Project Description
In the field of (bio-medical) image processing, segmentation of images is typically performed via U-Nets [1,2].

A U-Net consists of an encoder - a series of convolution and pooling layers which reduce the spatial resolution of the input, followed by a decoder - a series of transposed convolution and upsampling layers which increase the spatial resolution of the input. The encoder and decoder are connected by a bottleneck layer which is responsible for reducing the number of channels in the input. The key innovation of U-Net is the addition of skip connections that connect the contracting path to the corresponding layers in the expanding path, allowing the network to recover fine-grained details lost during downsampling.

![image](https://github.com/ryghrmni/DeepLifeProject/assets/111413160/53330d06-48db-40bc-9f3e-6d8e0b884645)


## Project Aim
The aim of the project is to download the cell images (preferably from the “2018 Data Science Bowl” competition) and assess the performance of a U-Net or any other deep model for cell segmentation. Participants are free to choose any model, as long as they are able to explain their rationale, architecture, strengths, and weaknesses.

## References
[1] Ronneberger, O., Fischer, P., Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. In: Navab, N., Hornegger, J., Wells, W., Frangi, A. (eds) Medical Image Computing and Computer-Assisted Intervention – MICCAI 2015. MICCAI 2015. Lecture Notes in Computer Science(), vol 9351. Springer, Cham. [Link](https://doi.org/10.1007/978-3-319-24574-4_28)

[2] Long, F. Microscopy cell nuclei segmentation with enhanced U-Net. BMC Bioinformatics 21, 8 (2020). [Link](https://doi.org/10.1186/s12859-019-3332-1)

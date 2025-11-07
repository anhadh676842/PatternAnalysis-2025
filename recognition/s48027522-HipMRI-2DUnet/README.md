# Prostate MRI Segmentation with 2D U-Net

## Project Overview

This project implements a 2D UNet CNN architecture for segmenting prostates from MRI scans using a **2D U-Net** architecture. The workflow is designed to handle **pre-processed 2D slices of prostate MRIs**, enabling efficient training and evaluation of segmentation models.  

The goal is to accurately delineate the prostate from surrounding tissues, which is critical for clinical applications such as **radiotherapy planning, disease diagnosis, and progression monitoring**.

---

## Features

- **2D Slice-Based Training**: Uses individual 2D slices extracted from 3D MRI volumes, enabling faster training and lower memory requirements.  
- **U-Net Architecture**: Lightweight **2D U-Net** with encoder-decoder structure and skip connections for high-resolution segmentation.  
- **Binary Dice Loss**: Implements a **Dice similarity coefficient loss** for robust training in binary segmentation (prostate vs. background).  
- **Data Augmentation**: Supports **random flips, rotations, and z-score normalization** for better generalization.  
- **Early Stopping**: Training halts automatically when the **Dice coefficient of the prostate exceeds 0.75**, avoiding overfitting.  
- **Visualization Tools**: Includes slice-level visualizations of input images, ground truth masks, and model predictions.  
- **Modular Design**: Easily replaceable components for experimenting with different models, loss functions, and transforms.

---
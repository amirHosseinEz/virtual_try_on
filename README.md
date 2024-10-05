
# High-Resolution Virtual Try-On via Misalignment-Aware Normalization

## Abstract
The goal of image-based virtual try-on is to overlay a chosen clothing item onto the appropriate area of a person. This process typically involves adjusting the item to fit the target body part and seamlessly blending it with the person's image. Despite numerous studies in this field, the generated images remain at a low resolution (e.g., 256×192), which significantly hinders the ability to meet the expectations of online shoppers.

VITON-HD introduces an innovative virtual try-on technique that generates high-resolution images at 1024×768 pixels. The process begins with creating a segmentation map to guide the synthesis. Then, the target clothing item is roughly adjusted to fit the person's body. To address misalignments and maintain the details of the high-resolution inputs, the method incorporates ALIgnment-Aware Segment (ALIAS) normalization and an ALIAS generator. Extensive comparisons with existing methods show that VITON-HD significantly outperforms them in both qualitative and quantitative measures of image quality.

In this project, we worked with the VTON-HD model designed for virtual try-on applications. Our primary focus was on refining the pre-processing pipeline. This involved optimizing pose estimation, clothing segmentation, and training a human parsing model. We focused on making the process better and faster by trying out different ideas and testing quicker models.

## Table of Contents
- [Introduction](#introduction)
- [Methods Overview](#methods-overview)
  - [Pre-Processing](#pre-processing)
    - [Pose Estimation](#pose-estimation)
      - [Body25 (OpenPose)](#body25-openpose)
      - [COCO (Common Objects in Context)](#coco-common-objects-in-context)
      - [MPII (Max Planck Institute for Informatics)](#mpii-max-planck-institute-for-informatics)
    - [Clothes Segmentation](#clothes-segmentation)
    - [Image Parsing](#image-parsing)
  - [Dataset](#dataset)
  - [Training](#training)
  - [Validation](#validation)
  - [Performance](#performance)
- [Clothing-Agnostic Person Representation](#clothing-agnostic-person-representation)
- [Segmentation Generation](#segmentation-generation)
- [Clothing Image Deformation](#clothing-image-deformation)
- [Try-On Synthesis via ALIAS Normalization](#try-on-synthesis-via-alias-normalization)

## Introduction
The main focus of this project is on the pre-processing phase of a virtual try-on system, aimed at optimizing the speed and performance. The pre-processing stage includes three key steps:
1. Pose Estimation
2. Image Parsing
3. Clothes Segmentation

In each step, the best methods were used to achieve the project's goals.

## Methods Overview

### Pre-Processing

#### Pose Estimation
Pose estimation is a computer vision technique used to determine the position and orientation of a person in an image. It involves tracking key points, such as joints, to understand the spatial configuration. Several pose estimation formats exist, including:

##### Body25 (OpenPose)
- Key Points: 25 key points.
- Detailed representation of the whole body.
- Used for sports analysis, dance, and gesture recognition.

##### COCO (Common Objects in Context)
- Key Points: 17 key points.
- General-purpose applications like human-computer interaction and surveillance.

##### MPII (Max Planck Institute for Informatics)
- Key Points: 16 key points.
- Focuses on body parts crucial for action recognition.

Between these, the VTON-HD model works with the Body25 format. Instead of using OpenPose due to its drawbacks, we opted for MediaPipe, which is faster and easier to use.

#### Clothes Segmentation
Clothes segmentation identifies different articles of clothing within an image. Using machine learning techniques like U-Net or Mask R-CNN, we opted for the "cloths_segmentation" model, which is both fast and accurate.

#### Image Parsing
Image parsing divides an image into parts, assigning a class label to each pixel. After trying different approaches, we trained our own model using YOLO V8 for instance segmentation, which produced better results.

### Dataset
The dataset includes 1000 labeled images for training and 200 images for validation. Each image is segmented into body parts like head, neck, upper clothes, and hands.

### Training
We trained the model on Google Colab using an NVIDIA T4 GPU. The training process took over 10 hours, with performance metrics recorded across multiple epochs.

### Validation
Validation results showed that the model achieved the best performance at epoch 45, after which it began to overfit.

### Performance
The YOLO-V8x segmentation model achieved strong results in terms of precision, recall, and mAP metrics.

## Clothing-Agnostic Person Representation
To train the model for virtual try-on, we used a clothing-agnostic person representation. This method removes clothing information while retaining the body shape, pose, and identifiable features of the person.

## Segmentation Generation
The segmentation generator predicts the segmentation map of the person wearing the target clothing item. A U-Net architecture was used for this purpose, trained using both cross-entropy and adversarial losses.

## Clothing Image Deformation
In this stage, the clothing item is deformed to fit the person's body. This is achieved using a geometric matching module, which aligns the clothing based on a correlation matrix.

## Try-On Synthesis via ALIAS Normalization
The final synthetic image is generated using the outputs from the previous stages. ALIAS normalization helps preserve semantic information and remove misleading information from misaligned areas, producing a high-quality virtual try-on result.

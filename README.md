# Multi-Model-Deep-Learning-Framework-for-Autonomous-Perception-and-Scene-Understanding


This repository contains the implementation of a drivable area segmentation model using ResNet-50 as the backbone of a Feature Pyramid Network (FPN). The model is enhanced with Monte Carlo (MC) Dropout and Spatial Dropout to estimate uncertainty in segmentation.

**Features**
**ResNet-50 Backbone:** Pretrained on ImageNet for transfer learning.

**Feature Pyramid Network (FPN):** Multi-scale feature extraction for segmentation.

**Monte Carlo Dropout (MC Dropout):** Uncertainty estimation through stochastic forward passes.

**Spatial Dropout:** Applied before segmentation output for regularization.

**IoU Evaluation:** Measures segmentation performance.

**Supports BDD100K Dataset:** For drivable area segmentation.

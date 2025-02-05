# Multi-Model-Deep-Learning-Framework-for-Autonomous-Perception-and-Scene-Understanding


This repository contains the implementation of a drivable area segmentation model using ResNet-50 as the backbone of a Feature Pyramid Network (FPN). The model is enhanced with Monte Carlo (MC) Dropout and Spatial Dropout to estimate uncertainty in segmentation.

### **Features**
- **_ResNet-50 Backbone:_** Pretrained on ImageNet for transfer learning.

- **_Feature Pyramid Network (FPN):_** Multi-scale feature extraction for segmentation.

- **_Monte Carlo Dropout (MC Dropout):_** Uncertainty estimation through stochastic forward passes.

- **_Spatial Dropout:_** Applied before segmentation output for regularization.

- **_IoU Evaluation:_** Measures segmentation performance.

- **_Supports BDD100K Dataset:_** For drivable area segmentation.

### **Installation**
```
# Clone the repository
git clone https://github.com/your-username/drivable-area-segmentation.git
cd drivable-area-segmentation

# Create virtual environment and install dependencies
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

pip install -r requirements.txt
```

### **Dataset Setup**
Download the [BDD100K](https://bair.berkeley.edu/blog/2018/05/30/bdd/) dataset and organize it as follows:
```
/mnt/nvme0n1p4/ML_Datasets/BDD100k/
    ├── train/
    │   ├── images/
    │   ├── masks/
    ├── val/
    │   ├── images/
    │   ├── masks/
```
Update the dataset paths in ```train.py``` if necessary.

### **Training**
Run the training script with: 
``` python train.py ```
Training will run for 90 epochs with a batch size of 10 and a learning rate 1e-4.


### **Test**


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
Run the training script with: <br>
``` python train.py ```
Training will run for 90 epochs with a batch size of 10 and a learning rate 1e-4.


### **Test**
Run the test script with <br>
``` python test.py ```


### **Inference**
To run inference on new images: <br>
``` python inference.py --image_path path/to/image.jpg --model_path drivable_area_segmentation_model.pth ```

### **Results**
A binary predicted mask was returned along with uncertainty estimation and YOLO predictions. 
![Output_Day2](https://github.com/user-attachments/assets/078df22a-effc-4a99-9cb9-2c0c0ea8b107)
![Output_Day1](https://github.com/user-attachments/assets/84bb6e35-87ad-4f9a-ab8e-f9789e216797)
![Output_Night1](https://github.com/user-attachments/assets/b605edce-6c6f-45a0-8d6b-3dddd1bf3f81)
![Output_Night2](https://github.com/user-attachments/assets/b89e4032-48c1-4a93-88e0-0f0fd7684504)

### **Future Work**
This model leverages uncertainty estimation and dropout to enhance the prediction of drivable areas for autonomous vehicles, achieving outstanding results on the BDD100K dataset. Integrating an attention mechanism improves focus on relevant features; however, the reliance on the computationally intensive ResNet architecture and repeated evaluations significantly limit real-time applicability, particularly for video processing. While the model effectively performs drivable area segmentation and object detection, it does not address other crucial tasks like depth estimation. Future work should focus on developing real-time, multi-task frameworks that balance speed and accuracy, offering a more comprehensive solution for autonomous perception.




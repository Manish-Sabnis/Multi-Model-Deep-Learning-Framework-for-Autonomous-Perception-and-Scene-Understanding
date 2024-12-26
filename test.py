import torch
import cv2
import numpy as np
import torchvision.transforms as transforms
from torchvision import models
from matplotlib import pyplot as plt
from ultralytics import YOLO
from PIL import Image
from FPN import FPNWithMCAndSpatialDropout

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "/home/manish/Documents/ML Lab programs/test/drivable_area_segmentation_model.pth"  
model = FPNWithMCAndSpatialDropout(models.resnet50(weights=None), num_classes=1)
model.load_state_dict(torch.load(model_path))
model = model.to(device)
model.eval()


def enable_mc_dropout(model):
    for module in model.modules():
        if module.__class__.__name__.startswith("Dropout"):
            module.train()

enable_mc_dropout(model)

yolo_model = YOLO('yolo11n.pt')


def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    orig_size = image.shape[:2]  
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = transform(image).unsqueeze(0).to(device)
    return input_tensor, orig_size, image


def get_uncertainty(model, input_tensor, n_iterations=10):
    model.eval() 
    enable_mc_dropout(model) 
    preds = []
    with torch.no_grad():
        for _ in range(n_iterations):
            preds.append(torch.sigmoid(model(input_tensor)).cpu().numpy())
    preds = np.stack(preds, axis=0)
    mean_pred = np.mean(preds, axis=0)
    uncertainty = np.std(preds, axis=0) 
    return mean_pred, uncertainty


image_path = "/mnt/nvme0n1p4/ML_Datasets/BDD100k/val/images/b4c86653-a5abaf4e.jpg"  
input_tensor, orig_size, original_image = preprocess_image(image_path)


mean_pred, uncertainty = get_uncertainty(model, input_tensor)
mean_pred_resized = cv2.resize(mean_pred[0, 0], (orig_size[1], orig_size[0]))
uncertainty_resized = cv2.resize(uncertainty[0, 0], (orig_size[1], orig_size[0]))

results = yolo_model.predict(source=original_image)


yolo_annotated_image = results[0].plot()  


fig, ax = plt.subplots(1, 4, figsize=(20, 10))
ax[0].imshow(original_image)
ax[0].set_title("Original Image")
ax[1].imshow(mean_pred_resized, cmap="gray")
ax[1].set_title("Segmentation Prediction")
ax[2].imshow(uncertainty_resized, cmap="hot")
ax[2].set_title("Uncertainty")
ax[3].imshow(cv2.cvtColor(yolo_annotated_image, cv2.COLOR_BGR2RGB))
ax[3].set_title("YOLO Detection")
plt.show()


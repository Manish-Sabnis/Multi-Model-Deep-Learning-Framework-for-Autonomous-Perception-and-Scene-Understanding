from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from dataloader import BDD100KDataset
import torch
import torch.nn as nn
from FPN import FPNWithMCAndSpatialDropout
from tqdm import tqdm
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision.models import resnet50, ResNet50_Weights
import numpy as np



train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

mask_transform = transforms.Compose([
    transforms.Resize((56, 56)),
    transforms.ToTensor()
])

# Dataset and DataLoader
train_dataset = BDD100KDataset(
    images_dir="/mnt/nvme0n1p4/ML_Datasets/BDD100k/train/images",
    masks_dir="/mnt/nvme0n1p4/ML_Datasets/BDD100k/mask_train",
    train_transform=train_transform,
    mask_transform=mask_transform
)
val_dataset = BDD100KDataset(
    images_dir="/mnt/nvme0n1p4/ML_Datasets/BDD100k/val/images",
    masks_dir="/mnt/nvme0n1p4/ML_Datasets/BDD100k/mask_val",
    train_transform=train_transform,
    mask_transform=mask_transform
)

train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=10, shuffle=True)


def mc_dropout_uncertainty(model, input_tensor, num_samples=100):
    """
    Perform Monte Carlo Dropout to calculate uncertainty.
    
    Args:
        model (nn.Module): The model with Monte Carlo Dropout.
        input_tensor (torch.Tensor): The input data.
        num_samples (int): Number of stochastic forward passes.
    
    Returns:
        mean_prediction (torch.Tensor): Mean of the predictions.
        uncertainty (torch.Tensor): Standard deviation of the predictions.
    """
    model.train()  
    predictions = []

    with torch.no_grad():
        for _ in range(num_samples):
            predictions.append(model(input_tensor))

    predictions = torch.stack(predictions)  
    
  
    mean_prediction = predictions.mean(dim=0)
    uncertainty = predictions.std(dim=0) 

    return mean_prediction, uncertainty

def calculate_iou(pred, mask, threshold=0.5):
    """
    Calculate IoU for a batch of predictions and ground truth masks.
    """
    pred = (torch.sigmoid(pred) > threshold).float()  
    intersection = (pred * mask).sum(dim=(1, 2))  
    union = (pred + mask - pred * mask).sum(dim=(1, 2))  
    iou = intersection / (union + 1e-7)  
    return iou.mean().item() 




def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    running_iou = 0.0
    
    for images, masks in tqdm(loader, desc="Training"):
        images, masks = images.to(device), masks.to(device)
        masks = masks.float()
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        running_iou += calculate_iou(outputs, masks)
    
    avg_loss = running_loss / len(loader)
    avg_iou = running_iou / len(loader)
    return avg_loss, avg_iou


def plot_predictions(model, loader, device):
    """Plot a random image with its mask and prediction."""
    model.eval()
    with torch.no_grad():
        images, masks = next(iter(loader))
        images, masks = images.to(device), masks.to(device)
        
        outputs = model(images)  # Get predictions
        outputs = torch.sigmoid(outputs)  # Apply sigmoid to get probabilities
        
        # Select a random sample
        idx = np.random.randint(0, len(images))
        image = images[idx].cpu().permute(1, 2, 0).numpy()
        mask = masks[idx].cpu().squeeze().numpy()
        pred_mask = outputs[idx].cpu().squeeze().numpy()
        
        # Plot the image, mask, and prediction
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.title("Image")
        plt.imshow(image)
        plt.axis("off")
        
        plt.subplot(1, 3, 2)
        plt.title("Ground Truth Mask")
        plt.imshow(mask, cmap="gray")
        plt.axis("off")
        
        plt.subplot(1, 3, 3)
        plt.title("Predicted Mask")
        plt.imshow(pred_mask, cmap="gray")
        plt.axis("off")
        
        plt.show()


def validate_one_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_iou = 0.0
    
    with torch.no_grad():
        for images, masks in tqdm(loader, desc="Validation"):
            images, masks = images.to(device), masks.to(device)
            masks = masks.float()
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            running_loss += loss.item()
            running_iou += calculate_iou(outputs, masks)
    
    avg_loss = running_loss / len(loader)
    avg_iou = running_iou / len(loader)
    return avg_loss, avg_iou





device = "cuda" if torch.cuda.is_available() else "cpu"
epochs = 90
backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
model = FPNWithMCAndSpatialDropout(backbone, num_classes=1)
model = nn.DataParallel(model)
model = model.to("cuda" if torch.cuda.is_available() else "cpu")
criterion = torch.nn.BCEWithLogitsLoss()  # Combine BCE with logits
optimizer = optim.Adam(model.parameters(), lr=1e-4)


for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")    
    train_loss, train_iou = train_one_epoch(model, train_loader, optimizer, criterion, device)
    val_loss, val_iou = validate_one_epoch(model, val_loader, criterion, device)    
    print(f"Train Loss: {train_loss:.4f}, Train mIoU: {train_iou:.4f}")
    print(f"Val Loss: {val_loss:.4f}, Val mIoU: {val_iou:.4f}")
    plot_predictions(model, val_loader, device)
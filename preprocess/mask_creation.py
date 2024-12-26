import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import random
import cv2
from PIL import Image 
import json 
import pandas as pd


DATASET_DIR = '/mnt/nvme0n1p4/ML_Datasets/BDD100k'
TRAIN_IMAGES_DIR = os.path.join(DATASET_DIR, 'train/images')
VALIDATION_IMAGES_DIR = os.path.join(DATASET_DIR, 'val/images')
TEST_IMAGES_DIR = os.path.join(DATASET_DIR, 'test')
MASK_TRAIN_DIR = os.path.join(DATASET_DIR, 'mask_train')
MASK_VALIDATION_DIR = os.path.join(DATASET_DIR, 'mask_val')
TRAIN_ANNOTATIONS_FILE = os.path.join(DATASET_DIR, 'train/annotations/bdd100k_labels_images_train.json')
VALIDATION_ANNOTATIONS_FILE = os.path.join(DATASET_DIR, 'val/annotations/bdd100k_labels_images_val.json')


def load_json_file(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)


def filter_drivable_area_vertices(labels):
    filtered = [label for label in labels if label['category'] == 'drivable area']
    result = []
    for label in filtered:
        if 'poly2d' in label:
            for poly in label['poly2d']:
                if 'vertices' in poly:
                    result.append({'vertices': poly['vertices']})
    return result


def create_binary_mask(vertices, image_shape=(720, 1280)):
    mask = np.zeros(image_shape, dtype=np.uint8)    
    polygon = np.array(vertices, dtype=np.int32)      
    cv2.fillPoly(mask, [polygon], 255)    
    return mask


def create_black_mask(image_path, mask_path):
    img = Image.open(image_path)
    width, height = img.size
    black_mask = np.zeros((height, width), dtype=np.uint8)
    black_mask_img = Image.fromarray(black_mask)
    black_mask_img.save(mask_path)


def process_train_data():
    train_data = load_json_file(TRAIN_ANNOTATIONS_FILE)
    train_item_array = [item for item in train_data]
    train_df = pd.DataFrame(train_item_array)
    train_df['labels'] = train_df['labels'].apply(filter_drivable_area_vertices)
    train_df.drop(columns=['attributes', 'timestamp'], inplace=True)
    
    for idx, row in train_df.iterrows():
        image_name = row['name']
        vertices = row['labels'][0]['vertices'] if row['labels'] else []
        if vertices:
            mask = create_binary_mask(vertices)
            mask_filename = os.path.join(MASK_TRAIN_DIR, f"{image_name.split('.')[0]}.png")
            cv2.imwrite(mask_filename, mask)
    
    train_images = os.listdir(TRAIN_IMAGES_DIR)
    mask_images = os.listdir(MASK_TRAIN_DIR)
    mask_image_set = set(mask_images)
    
    for image_name in train_images:
        mask_name = image_name.split('.')[0] + '.png'
        if mask_name not in mask_image_set:
            image_path = os.path.join(TRAIN_IMAGES_DIR, image_name)
            mask_path = os.path.join(MASK_TRAIN_DIR, mask_name)
            create_black_mask(image_path, mask_path)
            print(f"Created black mask for {image_name}")


def process_validation_data():
    validation_data = load_json_file(VALIDATION_ANNOTATIONS_FILE)
    validation_item_array = [item for item in validation_data]
    validation_df = pd.DataFrame(validation_item_array)
    validation_df['labels'] = validation_df['labels'].apply(filter_drivable_area_vertices)
    validation_df.drop(columns=['attributes', 'timestamp'], inplace=True)
    
    for idx, row in validation_df.iterrows():
        image_name = row['name']
        vertices = row['labels'][0]['vertices'] if row['labels'] else []
        if vertices:
            mask = create_binary_mask(vertices)
            mask_filename = os.path.join(MASK_VALIDATION_DIR, f"{image_name.split('.')[0]}.png")
            cv2.imwrite(mask_filename, mask)
    
    validation_images = os.listdir(VALIDATION_IMAGES_DIR)
    mask_images = os.listdir(MASK_VALIDATION_DIR)
    mask_image_set = set(mask_images)
    
    for image_name in validation_images:
        mask_name = image_name.split('.')[0] + '.png'
        if mask_name not in mask_image_set:
            image_path = os.path.join(VALIDATION_IMAGES_DIR, image_name)
            mask_path = os.path.join(MASK_VALIDATION_DIR, mask_name)
            create_black_mask(image_path, mask_path)
            print(f"Created black mask for {image_name}")

if __name__ == "__main__":
    process_train_data()
    process_validation_data()

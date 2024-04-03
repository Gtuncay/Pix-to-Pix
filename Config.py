import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
if DEVICE == "cpu":
    print("GPU unavailable")

TRAIN_DIR = "Training/CorruptedDataset/MixedDataset/" #DIRECTORY OF CORRUPTED IMAGES
#/Training/CorruptedDataset/CorruptedDataset
VAL_DIR = "Training/CutDataset/CutDataset/" #DIRECTORY OF UNCORRUPTED IMAGES
lr= 2e-4
batch_size = 16
num_workers = 4
img_size = 128
channels_img = 1 
L1_Lambda = 150 
num_epochs = 251 
load_model = False 
save_model = True  


both_transform = A.Compose(
    [A.Resize(width = 128, height = 128)],
    additional_targets={"image0": "image"},
)

#### Mean calculated to be = 0.5093056629614863; Std calculated to be = 0.10034612044559393 for uncorrupted, cut images

transform_only_input = A.Compose(
    [
        A.Normalize(mean = [0.509], std = [0.1], max_pixel_value = 255.0,), ##max pixel value, calculate mean and std of input
        ToTensorV2(),
    ]
)

transform_only_mask = A.Compose(
    [
        A.Normalize(mean = [0.509], std = [0.1], max_pixel_value = 255.0,), ##
        ToTensorV2(),
    ]
)


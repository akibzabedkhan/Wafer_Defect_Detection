import numpy as np
import torch
from torch.utils.data import Dataset
import sys
import os

sys.path.append(os.path.dirname(__file__))
from SpatialConfig import (WM38K_classes, background_label, good_die_label, defect_label, image_size)

def build_spatial_masks(wafer_map, label_vector):
    """
    Building per die wafer map
    """

    H, W = wafer_map.shape

    mask = np.zeros((len(WM38K_classes) - 1, H, W), dtype = np.float32)

    #sets all bakground wafers to -1 on all masks
    background = (wafer_map == 0)
    mask[:, background] = background_label

    #Goes through and labels each defect to 1
    defective = (wafer_map == 2)
    for defect_class in range(len(WM38K_classes) - 1):
        if label_vector[defect_class] == 1:
            mask[defect_class, defective] = defect_label
    
    return mask




class WM38KDataset(Dataset):
    """ dataset for stage 1 spatial segmentation"""

    def __init__(self, images, onehot_labels, augment = False):
        self.images = images
        self.labels = onehot_labels
        self.augment = augment

        print("Making spatial masks...")
        self.masks = np.stack([
            build_spatial_masks(images[i], onehot_labels[i, :len(WM38K_classes) - 1])
            for i in range(len(images))
        ])
        print(f"Masks bulit: {self.masks.shape}")

    
    def __len__(self):
        return len(self.images)
    

    def _augment(self, image, mask, wafer_map):
        """Function to flip wafers during training so that a way array of orientation can occur and be learned"""
        #These were directly copied, further review needed
        #50% hori. flip
        if torch.rand(1).item() > 0.5:
            image = torch.flip(image, dims=[2])
            mask = torch.flip(mask, dims=[2])
            wafer_map = torch.flip(wafer_map, dims=[1])

        # 50% vertical flip
        if torch.rand(1).item() > 0.5:
            image = torch.flip(image, dims=[1])
            mask = torch.flip(mask, dims=[1])
            wafer_map = torch.flip(wafer_map, dims=[0])

        # 75% chance of 90/180/270 rotation
        k = torch.randint(0, 4, (1,)).item()
        if k > 0:
            image = torch.rot90(image, k, dims=[1, 2])
            mask = torch.rot90(mask, k, dims=[1, 2])
            wafer_map = torch.rot90(wafer_map, k, dims=[0, 1])

        return image, mask, wafer_map
    

    def __getitem__(self, idx):
        # Normalise: 0→0.0, 1→0.5, 2→1.0
        image = torch.tensor(
            self.images[idx] / 2.0, dtype=torch.float32).unsqueeze(0)  # [1, H, W]

        mask = torch.tensor(self.masks[idx], dtype=torch.float32) # [num_defect_classes, H, W]

        label = torch.tensor(self.labels[idx], dtype=torch.float32)  # [number_of_classes]

        # Raw wafer map — returned directly so yield code needs no reverse normalisation
        wafer_map = torch.tensor(self.images[idx], dtype=torch.long)  # [H, W], values {0, 1, 2}

        if self.augment:
            image, mask, wafer_map = self._augment(image, mask, wafer_map)

        return image, mask, label, wafer_map


class WM811KDataset(Dataset):
    """Dataset for stage 1 on 811K"""
    


"""Adding the pseudo label masking down here, need to check validity of code"""
class WM38KPseudoDataset(Dataset):

    def __init__(self, images, onehot_labels, pseudo_masks, augment=False):
        self.images = images
        self.labels = onehot_labels
        self.masks = pseudo_masks
        self.augment = augment

        print(f"WM38KPseudoDataset ready: {len(images)} wafers, "
              f"masks shape: {pseudo_masks.shape}")

    def __len__(self):
        return len(self.images)

    def _augment(self, image, mask, wafer_map):
        if torch.rand(1).item() > 0.5:
            image = torch.flip(image, dims = [2])
            mask = torch.flip(mask, dims = [2])
            wafer_map = torch.flip(wafer_map, dims = [1])

        if torch.rand(1).item() > 0.5:
            image = torch.flip(image,     dims = [1])
            mask = torch.flip(mask,      dims = [1])
            wafer_map = torch.flip(wafer_map, dims = [0])

        k = torch.randint(0, 4, (1,)).item()
        if k > 0:
            image = torch.rot90(image, k, dims = [1, 2])
            mask = torch.rot90(mask, k, dims = [1, 2])
            wafer_map = torch.rot90(wafer_map, k, dims = [0, 1])

        return image, mask, wafer_map

    def __getitem__(self, idx):
        image = torch.tensor(
            self.images[idx] / 2.0, dtype = torch.float32
        ).unsqueeze(0)

        mask = torch.tensor(self.masks[idx], dtype = torch.float32)

        label = torch.tensor(self.labels[idx], dtype = torch.float32)

        wafer_map = torch.tensor(self.images[idx], dtype = torch.long) 

        if self.augment:
            image, mask, wafer_map = self._augment(image, mask, wafer_map)

        return image, mask, label, wafer_map
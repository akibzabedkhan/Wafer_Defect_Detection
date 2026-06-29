import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
import sys
import os

sys.path.append(os.path.dirname(__file__))
from SpatialConfig import (WM38K_data, batch_size, WM38K_classes, Class_threshold)

from SpatialData import WM38KDataset, WM38KPseudoDataset

def weighted_sampler(onehot_labels):
    #goes through each class counting amount of samples and gives more weight to those with less samples
    class_count = onehot_labels[:, :len(WM38K_classes) - 1].sum(axis=0).clip(min=1)
    class_weights = 1.0 / class_count

    sample_weights = np.array([
        class_weights[np.where(onehot_labels[i, :len(WM38K_classes) - 1] == 1)[0]].max()
        if onehot_labels[i, :len(WM38K_classes) - 1].sum() > 0
        else class_weights.min()
        for i in range(len(onehot_labels))
    ])

    return WeightedRandomSampler(weights = torch.tensor(sample_weights, dtype=torch.float32), 
                                   num_samples = len(sample_weights), replacement = True)



def print_class_distribution(labels, indices, split_name):
    print(f"\n {split_name} class distribution:")
    for i, name in enumerate(WM38K_classes[:8]):
        count = int((labels[indices, i].sum()))
        print(f"{name}: {count}  wafers")


    
def getWM38KLoaders(seed = 42):
    print("Loading the dataset")
    data = np.load(WM38K_data)

    raw_images = data["arr_0"]
    raw_labels = data["arr_1"]

    #adds the none label to our defect labels
    no_defect = (raw_labels.sum(axis = 1, keepdims = True) == 0).astype(np.float32)
    labels = np.concatenate([raw_labels, no_defect], axis = 1)

    """Making seperations for single and mixed defects to train"""
    defect_count = raw_labels.sum(axis = 1)
    single = np.where(defect_count == 1)[0]
    mixed = np.where(defect_count >= 2)[0]
    none = np.where(defect_count == 0)[0]

    #split training on single defects 70% train, 15% val, 15% test
    dominant_label_single = np.argmax(labels[single, :8], axis = 1)
    #The dominant labels ensure that every defect type is in the ratio
    single_train, single_temp = train_test_split(
        single, test_size = 0.30, random_state = seed, stratify = dominant_label_single
    )

    dominant_label_temp = np.argmax(labels[single_temp, :8], axis = 1)

    single_val, single_test = train_test_split(
        single_temp, test_size = 0.50, random_state = seed, stratify = dominant_label_temp
    )

    print(f"Stage 1: single defects splits")
    print(f"Train: {len(single_train)} 70%")
    print(f"Validation: {len(single_val)} 15%")
    print(f"Test: {len(single_test)} 15%")

    #Build each dataset
    train_dataset = WM38KDataset(raw_images[single_train], labels[single_train], augment = True)
    val_dataset = WM38KDataset(raw_images[single_val], labels[single_val], augment = False)
    test_dataset = WM38KDataset(raw_images[single_test], labels[single_test], augment = False)

    training_sampler = weighted_sampler(labels[single_train])

    train_loader = DataLoader(train_dataset, batch_size = batch_size,
                              sampler = training_sampler, num_workers = 0, pin_memory = True)
    
    val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle = False, num_workers = 0, pin_memory = True)

    test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False, num_workers = 0, pin_memory = True)

    mixed_data = (raw_images[mixed], labels[mixed])

    return train_loader, val_loader, test_loader, mixed_data

"""This starts the stage 2 loading to deal with the mixed defects.
First it needs to generate pseudo labels using the learned data from stage 1"""

def MixedLoaders(stage1_model, seed = 42):
    from SpatialConfig import stage2_batch

    

    print("Loading Mixed dataset for stage 2")
    data = np.load(WM38K_data)
    raw_images = data["arr_0"]
    raw_labels = data["arr_1"]

    no_defect = (raw_labels.sum(axis = 1, keepdims = True) == 0).astype(np.float32)
    labels = np.concatenate([raw_labels, no_defect], axis = 1)

    defect_count = raw_labels.sum(axis = 1)
    mixed_index = np.where(defect_count >= 2)[0]

    print("\n── Raw label verification ──")
    print(f"Total mixed wafers: {len(mixed_index)}")
    for i, name in enumerate(WM38K_classes[:8]):
        count = int(labels[mixed_index, i].sum())
        print(f"  col {i} ({name}): {count} wafers")

    # Also check what multilabel_stratify is assigning
    keys = multilabel_stratify(labels[mixed_index])
    print("\nStratification key distribution:")
    for k in range(8):
        print(f"  key={k} ({WM38K_classes[k]}): {(keys==k).sum()} wafers")

    print(f"Number of mixed defects in the dataset: {len(mixed_index)}")

    print("Making labels using stage 1 data")
    stage1_model.eval()
    #Changed to deal with new psedo mask making
    pseudo_masks = make_pseudo_labels(stage1_model, raw_images[mixed_index], labels[mixed_index])
    print(f"Pseudo mask shape {pseudo_masks.shape}")

    #recommended stratified split on the dominant classes for training
    dominant_mixed = multilabel_stratify(labels[mixed_index])

    new_mixed_index = np.arange(len(mixed_index))
    mixed_train, mixed_temp = train_test_split(new_mixed_index, test_size = 0.30, random_state = seed, stratify = dominant_mixed)

    dominant_temp = multilabel_stratify(labels[mixed_index[mixed_temp]])

    mixed_val, mixed_test = train_test_split(mixed_temp, test_size=0.5, random_state=seed, stratify=dominant_temp)

    print(f"\nStage 2 split (mixed-type only):")
    print(f"  Train : {len(mixed_train):>6} (70%)")
    print(f"  Val   : {len(mixed_val):>6} (15%)")
    print(f"  Test  : {len(mixed_test):>6} (15%)")

    """Building the new datasets with our pseudolabels"""

    train_dataset = WM38KPseudoDataset(raw_images[mixed_index[mixed_train]], labels[mixed_index[mixed_train]],
                                       pseudo_masks[mixed_train], augment = True)
    val_dataset = WM38KPseudoDataset(raw_images[mixed_index[mixed_val]], labels[mixed_index[mixed_val]], 
                                     pseudo_masks[mixed_val], augment = False)
    test_dataset = WM38KPseudoDataset(raw_images[mixed_index[mixed_test]], labels[mixed_index[mixed_test]], 
                                      pseudo_masks[mixed_test], augment = False)
    
    train_sampler = weighted_sampler(labels[mixed_index[mixed_train]])

    train_loader = DataLoader(
        train_dataset, batch_size = stage2_batch,
        sampler = train_sampler, num_workers=0, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size = stage2_batch,
        shuffle = False, num_workers = 0, pin_memory = True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size = stage2_batch,
        shuffle = False, num_workers = 0, pin_memory = True,
    )

    return train_loader, val_loader, test_loader


def make_pseudo_labels(model, images, label_vectors, batch_size = 32, threshold = 0.5):
    """Runs the stage 1 model on the mixed wafers to make a new mask
    Needed to make adjustments to ensure that pseudo labels dont hallucinate labels not present in wafer
    """

    from SpatialConfig import device, background_label

    number_defect_classes = 8
    N, H, W = images.shape
    pseudo_masks = np.zeros((N, number_defect_classes, H, W), dtype = np.float32)

    actual_active = (label_vectors[:, :number_defect_classes] > 0).astype(np.float32)

    with torch.no_grad():
        for i in range(0, N, batch_size):
            end = min(i + batch_size, N)
            batch = images[i:end]
            #normalizes and moves to the gpu
            imgs = torch.tensor(batch / 2.0, dtype = torch.float32).unsqueeze(1).to(device)

            segment_scans, _, _ = model(imgs)

            #These take in [B, 8, H, W]
            probability = torch.sigmoid(segment_scans)
            thresholds = torch.tensor(
                [Class_threshold[c] for c in range(number_defect_classes)], dtype = torch.float32).view(1, -1, 1, 1).to(device)

            predictions = (probability > thresholds).float()

            #Added to deal with the hallucinating classes
            active_batch = torch.tensor(actual_active[i:end], dtype =  torch.float32).unsqueeze(-1).unsqueeze(-1).to(device)

            predictions *= active_batch

            #Applies a die mask 
            valid_mask = torch.tensor((batch != 0).astype(np.float32)).unsqueeze(1).expand_as(predictions).to(device)

            #changes background to -1 by default
            result = predictions * valid_mask + (background_label * (1 - valid_mask))

            pseudo_masks[i:end] = result.cpu().numpy()
    return pseudo_masks

def multilabel_stratify(labels, num_classes = 8):
    """ Takes care of rarer classes to ensure that they show up in single wafer training well"""
    keys = np.argmax(labels[:, :num_classes], axis = 1)

    for rare_index in [7, 6, 5]:
        present = labels[:, rare_index] == 1
        keys[present] = rare_index

    return keys




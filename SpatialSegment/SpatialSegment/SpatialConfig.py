"""New Class for handling parameters for the per die segmentation task
"""

#Uses GPU instead of CPU
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

WM811K_data = "datasets/WM811K.pkl"

WM38K_data = "datasets/MixedWM38.npz"
image_size = 52

background_label = -1
good_die_label = 0
defect_label = 1

"""Known class labels for each defect type"""
WM38K_classes = [
    'Center',       
    'Donut',       
    'Edge-Loc',    
    'Edge-Ring',   
    'Loc',    
    'Near-full',      
    'Scratch',     
    'Random',           
    'none'
]

number_of_classes = 9

Class_threshold = {
    0: 0.80,  # Center
    1: 0.6,  # Donut
    2: 0.75,  # Edge-Loc
    3: 0.90,  # Edge-Ring
    4: 0.5,  # Loc
    5: 0.5,  # Near-Full
    6: 0.15, # Scratch 
    7: 0.5,  # Random
}

#Variables for training
batch_size = 32
epochs = 85
learning_rate = 0.0001
weight_decay = 1e-4
loss_ignore_index = -1
patience = 999
mixed_precision = True
checkpoint_dir = "checkpoints/"
Dropout_amount = 0.00

#Variables for stage 2
stage2_epochs = 100
stage2_lr = 0.0001
stage2_batch = 32
stage2_checkpoint = "checkpoints/spatial/best_spatial_model_stage2.pt"


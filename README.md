# Wafer_Defect_Detection
Machine Learning for Linking Manufacturing Defects to Yield Prediction in Semiconductor Wafer Maps

Requirements for running the model:
python 3.12.10
torch
torchvision
numpy
scikit-learn
matplotlib

To run the model:
1. Download the necessary dataset at https://github.com/Junliangwangdhu/WaferMap (WM811K is non functional currently)
2 Place the MixedWM38.npz file in a new folder labeled datasets outside of the SpatialSegment folder
3. Run the file SpatialTrain.py, This will run both stages 1 and 2 sequentially
4. The model will create a checkpoints folder to use for the creation of the pseudo wafer maps 
5. After stage 1 is trained, the checkpoint will be saved allowing for the SpatialVisual.py file to be run

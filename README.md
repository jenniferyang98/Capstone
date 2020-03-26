## Improving the Robustness of Neural Network on Road Segmentation via Stability Training

### Project directory layout

    .
    ├── Data                          	# Raw and processed datasets
    │	├── AOI_2_Vegas					# One of the four area of interests (AOIs)
    │	│	├── 8Bit					# Processed 8-bit images from 16-bit RGB images 
    │	│	├── geojson					# Ground truth vectors provided by SpaceNet
    │	│	├── Mask					# Processed segmentation masks from the geojson files
    │	│	├── Perturb4				# Gaussian noise perturbed (sigma=0.04) 8-bit images
    │	│	└── PS-RGB					# Original images provided by SpaceNet
    │	├── AOI_3_Paris					# The rest of the three AOIs have the same folder structure
    │	├── ...
    │	├── JPEG10						# JPEG compressed version of the test set with varying quality
    │	├── JPEG20
    │	├── ...
    │	└── Indices						# Train, validation, and test set indices
    ├── Code  							# All code files
    │	├── pytorch_zoo					# Res-Net and U-Net implementation
    │	├── Datapreprocessing.ipynb
    │	├── Dataset.py		
    │	├── LossFunctions.py					
    │	├── Estimator.py					
    │	├── Training.ipynb					
    │	├── Evaluation.ipynb
    │	├── ConstantFormat.py
    │	└── ModelConsant.json
    ├── Result							# Model outputs
    │	├── Base_Out					# Using the base model
    │	│	├── PNG						# With the original images
    │	│	├── JPEG10					# With JPEG compressed images
    │	│	├── JPEG20
    │	│	└── ...
    │	├── Stable_Out					# Using the stability model
    │	│	├── PNG
    │	│	├── JPEG10
    │	│	├── JPEG20
    │	│	└── ...
    │	└──  Metrics                    # Dictionaries that collects the dice loss of the test set images
    └── Weights                    		# The weights of the base and the stability model

### Code files
1. Datapreprocessing.ipynb
> Creates the dataset for baseline training from rasters and geojsons
> Generates the Gaussian perturbed images for stability training
> Produces the JPEG compressed version of the test set for evaluation
> Pickles the train/validation/test indices to share across files
2. Dataset.py
> Implements the training, validation, and test dataset class where data augmentation is done
3. LossFunctions.py
> Implements the dice loss
4. Estimator.py
> Performs model training. The fit method calls run_one_epoch, which calls run_one_batch, which calls evaluate to determine the loss and backpropagate
5. Training.ipynb
> Defines all the necessary components (model, optimizer, data loaders) and calls the estimator to train the model
6. Evaluation and Postprocess.ipynb
> Runs the inference process after the model is trained. Visualizes the model performance.
7. ConstantFormat.py and ModelConsant.json
> These two files together define the constants shared across all the files, providing a centralized way to alter configuration.

### Author
**Jennifer Yang** - jennifer.yang@minerva.kgi.edu

# Parkinson's Disease Detection System

This project focuses on detecting Parkinson's disease from MRI scans using a deep learning ensemble model combining EfficientNet and Vision Transformer (ViT).

## Important Notice

This project has only been tested on Linux. If you wish to use it on Windows, you must run it using WSL (Windows Subsystem for Linux).

## Setup & Installation

### Install Dependencies

Ensure you have [Pixi](https://pixi.sh/) installed, then run:

```bash
pixi run setup
```

### Download Dataset

Run the following command to download the dataset from Kaggle:

```bash
./scripts/download_dataset.sh
```
This script downloads and extracts the images into `backend/data`.

### Train the model:

Run the following command to create the ensemble model.
   
```bash
pixi run train 
```

The generated model files are placed in `backend/models`.

### Running the Servers

Run the following command to start the servers
```bash
pixi run start
```

## Model Architecture

- **EfficientNet B0**: Extracts spatial features from MRI scans.
- **Vision Transformer (ViT)**: Captures global dependencies in the image.
- **Fully Connected Layer**: Merges both feature maps and classifies the image.
- **Softmax Activation**: Outputs final class probabilities.

## Results & Evaluation

The model is evaluated using **categorical cross-entropy loss** and **accuracy metrics** on test data.

## Citation

### Datasets
- [Kaggle - Neurodegenerative Diseases](https://www.kaggle.com/datasets/toshall/neurodegenerative-diseases)
- [Kaggle - parkinsons_brain_mri_dataset](https://www.kaggle.com/datasets/irfansheriff/parkinsons-brain-mri-dataset/data)
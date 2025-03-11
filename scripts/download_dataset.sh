#!/bin/bash

# Get directory of script and project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Define variables for paths and URLs
DATA_DIR="${DATA_DIR:-$PROJECT_ROOT/backend/data}"
DATASET_FILENAME="parkinsons-brain-mri-dataset.zip"
DATASET_PATH="${DATA_DIR}/${DATASET_FILENAME}"
EXTRACT_DIR="${DATA_DIR}/parkinsons-dataset"
KAGGLE_URL="https://www.kaggle.com/api/v1/datasets/download/irfansheriff/parkinsons-brain-mri-dataset"

# Create data directory if it doesn't exist
mkdir -p ${DATA_DIR}

# Download the dataset
curl -L -o ${DATASET_PATH} ${KAGGLE_URL}

# Extract the dataset
unzip -o ${DATASET_PATH} -d ${EXTRACT_DIR}

# Clean up the zip file
rm ${DATASET_PATH}

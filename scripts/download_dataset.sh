#!/bin/bash
mkdir -p src/data
curl -L -o ./src/data/parkinsons-brain-mri-dataset.zip \
  https://www.kaggle.com/api/v1/datasets/download/irfansheriff/parkinsons-brain-mri-dataset

unzip -o ./src/data/parkinsons-brain-mri-dataset.zip -d ./src/data/parkinsons-dataset
rm ./src/data/parkinsons-brain-mri-dataset.zip

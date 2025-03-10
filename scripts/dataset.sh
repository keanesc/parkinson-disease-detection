#!/bin/bash
mkdir -p src/data
curl -L -o ../src/data/neurodegenerative-diseases.zip \
  https://www.kaggle.com/api/v1/datasets/download/toshall/neurodegenerative-diseases

unzip -o ../src/data/neurodegenerative-diseases.zip -d ../src/data/neurodegenerative-diseases/
rm assets/neurodegenerative-diseases.zip

#!/bin/bash
mkdir -p assets
curl -L -o assets/neurodegenerative-diseases.zip \
  https://www.kaggle.com/api/v1/datasets/download/toshall/neurodegenerative-diseases

unzip -o assets/neurodegenerative-diseases.zip -d assets/neurodegenerative-diseases/
rm assets/neurodegenerative-diseases.zip

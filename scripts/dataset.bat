@echo off
set datasetUrl=https://www.kaggle.com/api/v1/datasets/download/toshall/neurodegenerative-diseases
set zipPath=assets\neurodegenerative-diseases.zip
set extractPath=assets\neurodegenerative-diseases

:: Ensure assets directory exists
if not exist assets mkdir assets

:: Download dataset
powershell -Command "Invoke-WebRequest -Uri '%datasetUrl%' -OutFile '%zipPath%'"

:: Extract dataset
powershell -Command "Expand-Archive -Path '%zipPath%' -DestinationPath '%extractPath%' -Force"

:: Remove zip file
del /Q "%zipPath%"


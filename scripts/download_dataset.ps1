# PowerShell script to download and extract the dataset
$datasetUrl = "https://www.kaggle.com/api/v1/datasets/download/irfansheriff/parkinsons-brain-mri-dataset"
$zipPath = ".\backend\data\neurodegenerative-diseases.zip"
$extractPath = ".\backend\data\neurodegenerative-diseases"

# Ensure assets directory exists
if (!(Test-Path "assets")) {
    New-Item -ItemType Directory -Path "assets" | Out-Null
}

# Download dataset
Invoke-WebRequest -Uri $datasetUrl -OutFile $zipPath

# Extract dataset
Expand-Archive -Path $zipPath -DestinationPath $extractPath -Force

# Remove zip file
Remove-Item $zipPath


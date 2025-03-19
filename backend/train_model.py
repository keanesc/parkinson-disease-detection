# pylint: disable=import-error
"""Module docstring: This script trains an ensemble for diagnosing Parkinson's disease from images using EfficientNet and ViT models."""

import os
import shutil

import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.datasets import ImageFolder
from transformers import ViTFeatureExtractor, ViTModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load and partially freeze EfficientNet
efficientnet = models.efficientnet_b0(
    weights=models.EfficientNet_B0_Weights.DEFAULT
).features.to(device)
for param in efficientnet.parameters():
    param.requires_grad = False

# Load and partially freeze ViT
vit_model = ViTModel.from_pretrained("google/vit-base-patch16-224").to(device)
for param in vit_model.parameters():
    param.requires_grad = False


def create_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


# Get absolute path to project root (assuming the script is in src/)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

# Use absolute paths based on project root
DATA_DIR = os.getenv(
    "DATA_DIR",
    os.path.join(PROJECT_ROOT, "backend", "data", "input", "parkinsons_dataset"),
)
PROCESSED_DIR = os.getenv(
    "PROCESSED_DIR",
    os.path.join(PROJECT_ROOT, "backend", "data", "working", "processed_data"),
)

# Now these will work regardless of where you run the script from
train_dir = os.path.join(PROCESSED_DIR, "train")
val_dir = os.path.join(PROCESSED_DIR, "val")

base_dir = DATA_DIR
normal_dir = os.path.join(base_dir, "normal")
parkinsons_dir = os.path.join(base_dir, "parkinson")

# Create train, validation and test directories
output_base_dir = PROCESSED_DIR
train_dir = os.path.join(output_base_dir, "train")
val_dir = os.path.join(output_base_dir, "val")
test_dir = os.path.join(output_base_dir, "test")

for category in ["Normal", "Parkinson"]:
    os.makedirs(os.path.join(train_dir, category), exist_ok=True)
    os.makedirs(os.path.join(val_dir, category), exist_ok=True)
    os.makedirs(os.path.join(test_dir, category), exist_ok=True)


# Function to preprocess images
def preprocess_images(input_images_dir, output_images_dir, target_size=(224, 224)):
    """Preprocess images from the source_dir by resizing and normalizing them."""
    for filename in os.listdir(input_images_dir):
        img_path = os.path.join(input_images_dir, filename)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, target_size)
            img = img / 255.0  # Normalize to [0, 1]
            dest_img_path = os.path.join(output_images_dir, filename)
            cv2.imwrite(dest_img_path, img * 255)  # Save the preprocessed image


# Helper function to copy files
def copy_file(file_name, src_dir, dest_dir):
    src = os.path.join(src_dir, file_name)
    dest = os.path.join(dest_dir, file_name)
    shutil.copyfile(src, dest)


# Function to split data into train, validation, and test sets
def split_data(
    input_directory,
    train_directory,
    validation_directory,
    test_directory,
    split_size=0.7,
    valid_size=0.15,
    test_size=0.15,
):
    """Split images from source_dir into training, validation, and test sets."""
    all_files = []
    for file_name in os.listdir(input_directory):
        file_path = os.path.join(input_directory, file_name)
        if os.path.getsize(file_path) > 0:
            all_files.append(file_name)
        else:
            print(f"{file_name} is zero length, so ignoring.")

    train_set, temp_set = train_test_split(all_files, test_size=1 - split_size)
    valid_set, test_set = train_test_split(
        temp_set, test_size=test_size / (test_size + valid_size)
    )

    for file_name in train_set:
        copy_file(file_name, input_directory, train_directory)

    for file_name in valid_set:
        copy_file(file_name, input_directory, validation_directory)

    for file_name in test_set:
        copy_file(file_name, input_directory, test_directory)


# Visualize some images
def visualize_images(image_category, image_source_directory, num_images=5):
    plt.figure(figsize=(15, 5))
    for image_index, filename in enumerate(
        os.listdir(image_source_directory)[:num_images]
    ):
        img_path = os.path.join(image_source_directory, filename)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.subplot(1, num_images, image_index + 1)
            plt.imshow(img)
            plt.title(f"{image_category} - {image_index + 1}")
            plt.axis("off")
    plt.show()


# Define an ensemble model.
class EnsembleModel(nn.Module):
    def __init__(self, num_classes):
        super(EnsembleModel, self).__init__()
        self.efficientnet = efficientnet
        self.vit = vit_model
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # Add feature fusion layers instead of direct concatenation
        self.fusion = nn.Sequential(
            nn.Linear(1280 + 768, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x, vit_model_inputs):
        # EfficientNet branch.
        eff_features = self.efficientnet(x)
        eff_features = self.pool(eff_features)
        eff_features = torch.flatten(eff_features, start_dim=1)

        # Use precomputed inputs for ViT
        vit_outputs = self.vit(**vit_model_inputs).last_hidden_state[:, 0, :]

        # Combine features and pass through the final layer.
        features = torch.cat((eff_features, vit_outputs), dim=1)
        features = self.fusion(features)
        output = self.fc(features)
        return output


# Process each category
if __name__ == "__main__":
    # Creating a versioned directory to save the files with a timestamp
    # This helps in organizing and archiving results from different runs
    # for better tracking and reproducibility of experiments.
    import datetime
    import sys

    version_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    SAVE_DIR = os.getenv(
        "SAVE_DIR", os.path.join(PROJECT_ROOT, "results", "archived", version_str)
    )
    os.makedirs(SAVE_DIR, exist_ok=True)

    log_file_path = os.path.join(SAVE_DIR, "train_output.log")
    with open(log_file_path, "w", encoding="utf-8") as f:
        original_stdout = sys.stdout
        sys.stdout = f

        # Everything below runs only when you execute train_model.py directly
        for category, source_dir in [
            ("Normal", normal_dir),
            ("Parkinson", parkinsons_dir),
        ]:
            train_dest = os.path.join(train_dir, category)
            val_dest = os.path.join(val_dir, category)
            test_dest = os.path.join(test_dir, category)

            split_data(source_dir, train_dest, val_dest, test_dest)
            preprocess_images(train_dest, train_dest)
            preprocess_images(val_dest, val_dest)
            preprocess_images(test_dest, test_dest)

            # Visualize sample images
            print(f"Visualizing images from {category}")
            visualize_images(category, train_dest)

        print("Data preprocessing and splitting completed.")

        # Paths to the processed data directories
        train_dir = os.path.join(PROCESSED_DIR, "train")
        val_dir = os.path.join(PROCESSED_DIR, "val")
        test_dir = os.path.join(PROCESSED_DIR, "test")

        # Choose device: GPU if available, else CPU.
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Set basic parameters.
        NUM_CLASSES = 2
        BATCH_SIZE = 32
        EPOCHS = 30
        LEARNING_RATE = 1e-4

        # Define transformations for PyTorch
        transform_train = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(20),  # example "stronger" rotation
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                ),
                transforms.RandomAffine(
                    degrees=0,
                    translate=(0.2, 0.2),
                    scale=(0.8, 1.2),
                    shear=0.2,
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        transform_val = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        # Create datasets and loaders once
        train_dataset = ImageFolder(train_dir, transform=transform_train)
        val_dataset = ImageFolder(val_dir, transform=transform_val)

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

        print(
            f"Training with {len(train_dataset)} images in {len(train_dataset.classes)} classes"
        )
        print(f"Class mapping: {train_dataset.class_to_idx}")

        # Load the pre-trained EfficientNet and freeze its weights.
        efficientnet = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.DEFAULT
        ).features.to(device)
        for param in efficientnet.parameters():
            param.requires_grad = False

        # Load the pre-trained ViT model and its feature extractor, and freeze its weights.
        vit_model = ViTModel.from_pretrained("google/vit-base-patch16-224").to(device)
        vit_feature_extractor = ViTFeatureExtractor.from_pretrained(
            "google/vit-base-patch16-224"
        )
        for param in vit_model.parameters():
            param.requires_grad = False

        # Selectively unfreeze later layers of the models for fine-tuning
        # For EfficientNet
        for i, layer in enumerate(efficientnet):
            if i > 6:  # Unfreeze the last few layers
                for param in layer.parameters():
                    param.requires_grad = True

        # For ViT - unfreeze the last transformer blocks
        for i, layer in enumerate(vit_model.encoder.layer):
            if i >= 9:  # Unfreeze the last 3 transformer blocks
                for param in layer.parameters():
                    param.requires_grad = True

        # Set up the model, loss function, and optimizer.
        model = EnsembleModel(NUM_CLASSES).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(
            [
                {"params": model.fusion.parameters()},
                {"params": model.fc.parameters()},
                {
                    "params": [
                        p for p in model.efficientnet.parameters() if p.requires_grad
                    ],
                    "lr": LEARNING_RATE / 10,
                },
                {
                    "params": [p for p in model.vit.parameters() if p.requires_grad],
                    "lr": LEARNING_RATE / 10,
                },
            ],
            lr=LEARNING_RATE,
            weight_decay=1e-5,
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, "min", patience=3, factor=0.5
        )

        best_val_loss = float("inf")
        EARLY_STOPPING_PATIENCE = 3
        NO_IMPROVEMENT_COUNT = 0

        # Training loop.
        for epoch in range(EPOCHS):
            model.train()
            RUNNING_LOSS = 0.0
            for images, labels in train_loader:
                # Preprocess for ViT before forward pass
                pil_images = []
                for img_tensor in images:
                    pil_images.append(transforms.ToPILImage()(img_tensor.cpu()))
                vit_inputs = vit_feature_extractor(
                    images=pil_images, return_tensors="pt"
                )
                vit_inputs = {k: v.to(device) for k, v in vit_inputs.items()}

                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images, vit_inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                RUNNING_LOSS += loss.item()
            avg_train_loss = RUNNING_LOSS / len(train_loader)

            # Validation phase.
            model.eval()
            VAL_LOSS = 0.0
            CORRECT = 0
            TOTAL = 0
            with torch.no_grad():
                for images, labels in val_loader:
                    pil_images = []
                    for img_tensor in images:
                        pil_images.append(transforms.ToPILImage()(img_tensor.cpu()))
                    vit_inputs = vit_feature_extractor(
                        images=pil_images, return_tensors="pt"
                    )
                    vit_inputs = {k: v.to(device) for k, v in vit_inputs.items()}

                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images, vit_inputs)
                    loss = criterion(outputs, labels)
                    VAL_LOSS += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    TOTAL += labels.size(0)
                    CORRECT += (predicted == labels).sum().item()
            avg_val_loss = VAL_LOSS / len(val_loader)
            val_acc = CORRECT / TOTAL * 100.0
            print(
                f"Epoch {epoch + 1}/{EPOCHS}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%, Strike {NO_IMPROVEMENT_COUNT}"
            )

            # Step the scheduler to adjust LR based on validation loss
            scheduler.step(avg_val_loss)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                NO_IMPROVEMENT_COUNT = 0
            else:
                NO_IMPROVEMENT_COUNT += 1
                if NO_IMPROVEMENT_COUNT >= EARLY_STOPPING_PATIENCE:
                    print("Early stopping triggered.")
                    break

            torch.cuda.empty_cache()

        print("Training finished!")

        # Save the trained model using absolute paths and environment variables
        MODEL_DIR = os.getenv(
            "MODEL_DIR", os.path.join(PROJECT_ROOT, "backend", "models")
        )
        os.makedirs(MODEL_DIR, exist_ok=True)

        torch.save(model, os.path.join(MODEL_DIR, "ensemble_model_full.pt"))
        torch.save(model.state_dict(), os.path.join(MODEL_DIR, "ensemble_model.pth"))
        vit_feature_extractor.save_pretrained(
            os.path.join(MODEL_DIR, "vit_feature_extractor")
        )

        print(f"Model saved to {MODEL_DIR}")

        # Then save your models and extractor there
        torch.save(model, os.path.join(SAVE_DIR, "ensemble_model_full.pt"))
        torch.save(model.state_dict(), os.path.join(SAVE_DIR, "ensemble_model.pth"))
        vit_feature_extractor.save_pretrained(
            os.path.join(SAVE_DIR, "vit_feature_extractor")
        )

        sys.stdout = original_stdout
        print(f"Log file written to {SAVE_DIR}")

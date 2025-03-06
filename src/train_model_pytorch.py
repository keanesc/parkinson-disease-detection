import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from transformers import ViTFeatureExtractor, ViTModel

# Choose device: GPU if available, else CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set basic parameters.
num_classes = 4
batch_size = 4
epochs = 10
learning_rate = 1e-4

# Define image transformations.
transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


# Custom Dataset that works with numpy arrays.
class NumpyDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images  # Expects images as numpy arrays (H, W, C)
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        # Convert to uint8 if not already
        if img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8)
        img = Image.fromarray(img)
        if self.transform:
            img = self.transform(img)
        label = self.labels[idx]
        if isinstance(label, np.ndarray) and label.ndim > 0:
            label = np.argmax(label)
        return img, torch.tensor(label, dtype=torch.long)


# Load preprocessed data.
dataset_path = "./assets/preprocessed_data/dataset.npz"
data = np.load(dataset_path, allow_pickle=True)
X_train = data["X_train"]
X_val = data["X_val"]
y_train = data["y_train"]
y_val = data["y_val"]

# Create dataset objects and data loaders.
train_dataset = NumpyDataset(X_train, y_train, transform=transform)
val_dataset = NumpyDataset(X_val, y_val, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Load the pre-trained EfficientNet and freeze its weights.
efficientnet = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT).features.to(device)
for param in efficientnet.parameters():
    param.requires_grad = False

# Load the pre-trained ViT model and its feature extractor, and freeze its weights.
vit_model = ViTModel.from_pretrained("google/vit-base-patch16-224").to(device)
vit_feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")
for param in vit_model.parameters():
    param.requires_grad = False


# Define an ensemble model.
class EnsembleModel(nn.Module):
    def __init__(self, num_classes):
        super(EnsembleModel, self).__init__()
        self.efficientnet = efficientnet
        self.vit = vit_model
        # Global pooling to reduce the spatial size of EfficientNet's output.
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        # The fully connected layer takes features from both branches.
        self.fc = nn.Linear(1280 + 768, num_classes)

    def forward(self, x):
        # EfficientNet branch.
        eff_features = self.efficientnet(x)
        eff_features = self.pool(eff_features)
        eff_features = torch.flatten(eff_features, start_dim=1)

        # ViT branch: convert each tensor in the batch to a PIL image.
        pil_images = []
        for img_tensor in x:
            img = transforms.ToPILImage()(img_tensor.cpu())
            pil_images.append(img)
        vit_inputs = vit_feature_extractor(images=pil_images, return_tensors="pt")
        vit_inputs = {k: v.to(device) for k, v in vit_inputs.items()}
        vit_outputs = self.vit(**vit_inputs).last_hidden_state[:, 0, :]

        # Combine features and pass through the final layer.
        features = torch.cat((eff_features, vit_outputs), dim=1)
        output = self.fc(features)
        return output


# Set up the model, loss function, and optimizer.
model = EnsembleModel(num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop.
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    avg_train_loss = running_loss / len(train_loader)

    # Validation phase.
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    avg_val_loss = val_loss / len(val_loader)
    val_acc = correct / total * 100.0
    print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%")
    torch.cuda.empty_cache()

print("Training finished!")

# Save the trained model.
model_save_path = "./assets/models/ensemble_model.pth"
torch.save(model.state_dict(), model_save_path)

print(f"Model saved to {model_save_path}")

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix

# Load trained model
model_path = "parkinsons_mri_model.pth"
model = torch.load(model_path)
model.eval()


def evaluate_model(model, dataloader, device):
    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    # Compute metrics
    print("Classification Report:")
    print(classification_report(y_true, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=range(4), yticklabels=range(4))
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()


evaluate_model(model, val_loader, device)


# Grad-CAM Implementation
def grad_cam(model, image, target_class, device):
    image = image.unsqueeze(0).to(device)
    model.zero_grad()

    # Get features from EfficientNet
    features = model.efficientnet(image)
    grad_target = features

    # Compute gradients
    features.retain_grad()
    output = model(image)
    loss = output[0, target_class]
    loss.backward()

    # Get gradients & weight them
    gradients = features.grad[0]
    weights = torch.mean(gradients, dim=(1, 2), keepdim=True)
    activation_map = torch.sum(weights * grad_target[0], dim=0)
    activation_map = torch.clamp(activation_map, min=0).cpu().detach().numpy()

    # Normalize heatmap
    activation_map = (activation_map - activation_map.min()) / (activation_map.max() - activation_map.min())

    # Show image with heatmap
    plt.imshow(image.squeeze().permute(1, 2, 0).cpu().numpy())
    plt.imshow(activation_map, cmap="jet", alpha=0.5)
    plt.title("Grad-CAM Visualization")
    plt.axis("off")
    plt.show()


# Example Grad-CAM usage
data_iter = iter(val_loader)
sample_image, sample_label = next(data_iter)
grad_cam(model, sample_image[0], sample_label[0].item(), device)

# Save model
torch.save(model, "parkinsons_mri_model.pth")
print("Model saved successfully.")

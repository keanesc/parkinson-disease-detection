"""
Preprocess the dataset using TensorFlow, OpenCV, and NumPy.

Steps:
1. Import Required Libraries
2. Define Preprocessing Function
3. Load Dataset
4. Split the Dataset
5. Run the Pipeline
"""

import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

def preprocess_image(image_path, target_size=(224, 224)):
    """Load an image, convert to RGB, resize, and normalize."""
    image = cv2.imread(image_path)  # Read image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
    image = cv2.resize(image, target_size)  # Resize to 224x224
    image = image.astype("float32") / 255.0  # Normalize to [0,1]
    return image

def load_dataset(data_dir):
    """Load images and labels from the dataset folder."""
    images, labels = [], []
    class_names = sorted(os.listdir(data_dir))  # Get class names

    for label, class_name in enumerate(class_names):
        class_path = os.path.join(data_dir, class_name)
        for filename in os.listdir(class_path):
            img_path = os.path.join(class_path, filename)
            images.append(preprocess_image(img_path))  # Process image
            labels.append(label)

    return np.array(images), np.array(labels), class_names

def split_data(images, labels):
    """Split dataset into train (70%), val (20%), test (10%)."""
    labels = to_categorical(labels)  # Convert to one-hot encoding

    X_train, X_temp, y_train, y_temp = train_test_split(
        images, labels, test_size=0.3, stratify=labels, random_state=42
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.33, stratify=y_temp, random_state=42
    )  # 20% validation, 10% test

    return X_train, X_val, X_test, y_train, y_val, y_test

if __name__ == "__main__":
    data_dir = "./assets/neurodegenerative-diseases"
    images, labels, class_names = load_dataset(data_dir)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(images, labels)

    print(f"Dataset Loaded! Classes: {class_names}")
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # Save processed dataset as .npz
    save_dir = "./assets/processed_data"
    os.makedirs(save_dir, exist_ok=True)
    np.savez_compressed(os.path.join(save_dir, "dataset.npz"),
                        X_train=X_train, X_val=X_val, X_test=X_test,
                        y_train=y_train, y_val=y_val, y_test=y_test)
    print(f"Processed dataset saved in {save_dir}/dataset.npz!")

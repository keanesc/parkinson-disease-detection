# Create a new file: inference.py
import cv2
import torch
from torchvision import transforms
from train_model import EnsembleModel  # noqa: F401
from transformers import ViTFeatureExtractor


class ParkinsonsPredictor:
    def __init__(self, model_path, feature_extractor_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create a fresh instance of the model, then load the state dict:
        self.model = EnsembleModel(num_classes=2).to(self.device)
        # self.model = torch.load(model_path, map_location=self.device, weights_only=False)

        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        self.feature_extractor = ViTFeatureExtractor.from_pretrained(
            feature_extractor_path
        )

        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        self.class_mapping = {0: "Normal", 1: "Parkinson's Disease"}

    def predict(self, image_path):
        # Load and preprocess image
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Transform for PyTorch model
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)

        # Prepare ViT inputs
        vit_inputs = self.feature_extractor(images=[img], return_tensors="pt")
        vit_inputs = {k: v.to(self.device) for k, v in vit_inputs.items()}

        # Get prediction
        with torch.no_grad():
            outputs = self.model(img_tensor, vit_inputs)
            _, predicted = torch.max(outputs, 1)

        pred_class = predicted.item()
        pred_label = self.class_mapping[pred_class]
        confidence = torch.softmax(outputs, dim=1)[0][pred_class].item()

        return {
            "prediction": pred_label,
            "class_id": pred_class,
            "confidence": confidence,
        }


# Example usage
if __name__ == "__main__":
    import os

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(BASE_DIR, "models", "ensemble_model.pth")
    EXTRACTOR_PATH = os.path.join(BASE_DIR, "models", "vit_feature_extractor")

    predictor = ParkinsonsPredictor(
        model_path=MODEL_PATH,
        feature_extractor_path=EXTRACTOR_PATH,
    )

    result = predictor.predict("path/to/your/image.jpg")
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']:.2%}")

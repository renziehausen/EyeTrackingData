import torch
import torchvision.models as models
import torch.nn as nn

def load_model(model_path):
    """Load a ResNet model and modify for regression."""
    model = models.resnet18() if "ResNet18" in model_path else models.resnet50()
    model.fc = nn.Linear(model.fc.in_features, 1)
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu"), weights_only=False))
    model.eval()
    return model

# Load models
left_model_18 = load_model("models_storage/ResNet18/left_eye.pt")
right_model_18 = load_model("models_storage/ResNet18/right_eye.pt")
left_model_50 = load_model("models_storage/ResNet50/left_eye.pt")
right_model_50 = load_model("models_storage/ResNet50/right_eye.pt")

def predict_pupil_diameter(model, eye_img):
    """Predict pupil diameter from eye image."""
    if eye_img is None:
        return 0

    eye_img = torch.tensor(eye_img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        output = model(eye_img)
    return output.item()
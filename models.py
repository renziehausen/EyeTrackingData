import torch
import torchvision.models as models
import torch.nn as nn
import cv2

def load_model(model_path):
    """Load a ResNet model and modify for regression."""
    model = models.resnet18() if "ResNet18" in model_path else models.resnet50()

    # Adjust the final layer for regression (1 output neuron)
    model.fc = nn.Linear(model.fc.in_features, 1)  

    # Load the state dict with potential prefix handling
    state_dict = torch.load(model_path, map_location=torch.device("cpu"))

    # Remove the 'resnet.' prefix if it exists
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("resnet."):
            new_state_dict[key[len("resnet."):]] = value  # Strip "resnet."
        else:
            new_state_dict[key] = value

    # Remove fc weights from checkpoint since they don't match
    if "fc.weight" in new_state_dict:
        del new_state_dict["fc.weight"]
    if "fc.bias" in new_state_dict:
        del new_state_dict["fc.bias"]

    # Load the modified state dict
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()
    
    return model

# Load models
left_model_18 = load_model("models_storage/ResNet18/left_eye.pt")
right_model_18 = load_model("models_storage/ResNet18/right_eye.pt")
left_model_50 = load_model("models_storage/ResNet50/left_eye.pt")
right_model_50 = load_model("models_storage/ResNet50/right_eye.pt")

def predict_pupil_diameter(model, eye_img):
    """Predict pupil diameter from eye image."""
    if eye_img is None or len(eye_img) == 0:
        return 0

    # Ensure input is a NumPy array
    if isinstance(eye_img, list):
        eye_img = np.array(eye_img, dtype=np.float32)

    # Resize to 224x224 (expected by ResNet)
    eye_img = cv2.resize(eye_img, (224, 224), interpolation=cv2.INTER_CUBIC)

    # Convert to tensor
    eye_img = torch.tensor(eye_img, dtype=torch.float32)

    # Ensure correct format (batch, channels, height, width)
    if eye_img.ndim == 3 and eye_img.shape[-1] == 3:
        eye_img = eye_img.permute(2, 0, 1)  # Convert (H, W, C) → (C, H, W)

    eye_img = eye_img.unsqueeze(0)  # Add batch dimension → [1, 3, 224, 224]

    with torch.no_grad():
        output = model(eye_img)

    return output.item()
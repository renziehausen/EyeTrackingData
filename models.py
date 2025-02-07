import torch
import torchvision.models as models
import torch.nn as nn

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
    if eye_img is None:
        return 0

    eye_img = torch.tensor(eye_img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        output = model(eye_img)
    return output.item()
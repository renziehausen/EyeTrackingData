import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image

# ----------------------------
# Custom ResNet18 for Regression
# ----------------------------
class CustomResNet18(nn.Module):
    def __init__(self, num_classes=1):
        super(CustomResNet18, self).__init__()
        self.num_classes = num_classes
        # Instantiate ResNet18 without pretrained weights.
        self.resnet = models.resnet18(weights=None)
        # Attach a separate regression head.
        self.regression_head = nn.Linear(1000, self.num_classes)

    def forward(self, x, masks=None):
        # Dynamically pad input so that it becomes 224x224.
        height, width = x.shape[2], x.shape[3]
        pad_height = max(0, (224 - height) // 2)
        pad_width  = max(0, (224 - width) // 2)
        x = F.pad(x, (pad_width, pad_width, pad_height, pad_height), mode="constant", value=0)
        x = self.resnet(x)
        x = self.regression_head(x)
        return x

# ----------------------------
# Custom ResNet50 for Regression
# ----------------------------
class CustomResNet50(nn.Module):
    def __init__(self, num_classes=1):
        super(CustomResNet50, self).__init__()
        self.num_classes = num_classes
        # Instantiate ResNet50 without pretrained weights.
        self.resnet = models.resnet50(weights=None)
        # Attach a separate regression head.
        self.regression_head = nn.Linear(1000, self.num_classes)

    def forward(self, x, masks=None):
        # Dynamically pad input so that it becomes 224x224.
        height, width = x.shape[2], x.shape[3]
        pad_height = max(0, (224 - height) // 2)
        pad_width  = max(0, (224 - width) // 2)
        x = F.pad(x, (pad_width, pad_width, pad_height, pad_height), mode="constant", value=0)
        x = self.resnet(x)
        x = self.regression_head(x)
        return x

# ----------------------------
# Pupil Diameter Prediction Function
# ----------------------------
def predict_pupil_diameter(model, eye_image):
    """
    Preprocesses the cropped eye image and predicts pupil diameter using the provided model.
    """
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(eye_image).unsqueeze(0)  # Add batch dimension.
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
    return output.item()

# ----------------------------
# Model Loading Function with Key Remapping
# ----------------------------
def load_custom_model(model_path, arch='resnet18'):
    """
    Loads a pretrained model from the given checkpoint.
    
    The checkpoint is assumed to have keys that may not have the proper prefixes for our custom model.
    This function remaps keys as follows:
      - Keys starting with "fc." are remapped to "regression_head.".
      - Keys starting with "regression_head." or "resnet." are kept as-is.
      - Any other keys have "resnet." prepended.
    
    Parameters:
        model_path: Path to the saved checkpoint.
        arch: 'resnet18' or 'resnet50'
    
    Returns:
        A model instance with the loaded weights.
    """
    if arch == 'resnet18':
        model = CustomResNet18(num_classes=1)
    elif arch == 'resnet50':
        model = CustomResNet50(num_classes=1)
    else:
        raise ValueError("Unsupported architecture: choose 'resnet18' or 'resnet50'")
    
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("fc."):
            # Remap keys for the final layer.
            new_key = "regression_head." + key[len("fc."):]
            new_state_dict[new_key] = value
        elif key.startswith("regression_head.") or key.startswith("resnet."):
            # Keep these keys as-is.
            new_state_dict[key] = value
        else:
            # For any other key, add the "resnet." prefix.
            new_key = "resnet." + key
            new_state_dict[new_key] = value

    model.load_state_dict(new_state_dict, strict=True)
    return model

# ----------------------------
# Load Models for Main
# ----------------------------
left_model_18  = load_custom_model("models_storage/ResNet18/left_eye.pt", arch='resnet18')
right_model_18 = load_custom_model("models_storage/ResNet18/right_eye.pt", arch='resnet18')
left_model_50  = load_custom_model("models_storage/ResNet50/left_eye.pt", arch='resnet50')
right_model_50 = load_custom_model("models_storage/ResNet50/right_eye.pt", arch='resnet50')
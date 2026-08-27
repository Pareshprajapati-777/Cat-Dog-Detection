import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os

# MobileNetV2 ImageNet class index mapping as fallback
CAT_INDICES = set(range(281, 286))
DOG_INDICES = set(range(151, 269))

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_classifier(model_path="cat_dog_model.pth"):
    """
    Loads fine-tuned PyTorch model if available, or pre-trained MobileNetV2.
    """
    if os.path.exists(model_path):
        model = models.mobilenet_v2(weights=None)
        model.classifier[1] = nn.Sequential(
            nn.Linear(model.last_channel, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2)
        )
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        model.eval()
        return model, "fine_tuned"
    else:
        # Fallback to pre-trained ImageNet MobileNetV2
        weights = models.MobileNet_V2_Weights.DEFAULT
        model = models.mobilenet_v2(weights=weights)
        model.eval()
        return model, "imagenet"

def predict_image(image_input, model, model_type="imagenet"):
    """
    Predicts whether an image is a Cat or a Dog.
    Accepts PIL Image or file path.
    Returns label, confidence (0-100), and dict of scores.
    """
    if isinstance(image_input, str):
        img = Image.open(image_input).convert('RGB')
    else:
        img = image_input.convert('RGB')
        
    tensor = transform(img).unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(tensor)
        
        if model_type == "fine_tuned":
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            cat_score = float(probabilities[0].item())
            dog_score = float(probabilities[1].item())
        else:
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            cat_prob = float(sum(probabilities[idx].item() for idx in CAT_INDICES))
            dog_prob = float(sum(probabilities[idx].item() for idx in DOG_INDICES))
            
            total = cat_prob + dog_prob
            if total > 0:
                cat_score = cat_prob / total
                dog_score = dog_prob / total
            else:
                cat_score, dog_score = 0.5, 0.5
                
    if dog_score > cat_score:
        label = "Dog"
        emoji = "🐶"
        confidence = dog_score * 100
    else:
        label = "Cat"
        emoji = "🐱"
        confidence = cat_score * 100
        
    return {
        "label": label,
        "emoji": emoji,
        "confidence": confidence,
        "cat_score": cat_score * 100,
        "dog_score": dog_score * 100
    }

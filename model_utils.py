import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os

# ImageNet Class Mappings for Open-World Detection
# 1. Domestic cats (281..285) and wild felines (286..293)
CAT_INDICES = set(range(281, 294))

# 2. Domestic dogs (151..268) and canines/wild dogs (269..275)
DOG_INDICES = set(range(151, 276))

# 3. Related small mammals that may visually resemble cats/dogs in ImageNet (foxes, weasels, ferrets)
RELATED_MAMMALS = set(range(276, 281)).union(set(range(356, 362)))

# Pre-processing transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_classifier(model_path="cat_dog_model.pth"):
    """
    Loads fine-tuned PyTorch model if available, together with ImageNet MobileNetV2
    for out-of-distribution (Unknown object) open-world detection.
    """
    # Load ImageNet MobileNetV2 for 1000-class open-world detection
    weights = models.MobileNet_V2_Weights.DEFAULT
    imagenet_model = models.mobilenet_v2(weights=weights)
    imagenet_model.eval()
    categories = weights.meta['categories']

    fine_tuned_model = None
    model_type = "imagenet"

    if os.path.exists(model_path):
        fine_tuned_model = models.mobilenet_v2(weights=None)
        fine_tuned_model.classifier[1] = nn.Sequential(
            nn.Linear(fine_tuned_model.last_channel, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2)
        )
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            fine_tuned_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            fine_tuned_model.load_state_dict(checkpoint)
        fine_tuned_model.eval()
        model_type = "fine_tuned"

    classifier_bundle = {
        "imagenet_model": imagenet_model,
        "fine_tuned_model": fine_tuned_model,
        "categories": categories
    }

    return classifier_bundle, model_type

def predict_image(image_input, model, model_type="fine_tuned"):
    """
    Predicts whether an image is a Cat, Dog, or Unknown (neither Cat nor Dog).
    Accepts PIL Image or file path.
    Returns label, emoji, confidence, cat_score, dog_score, detected_object, and is_cat_or_dog.
    """
    if isinstance(image_input, str):
        img = Image.open(image_input).convert('RGB')
    else:
        img = image_input.convert('RGB')
        
    tensor = transform(img).unsqueeze(0)

    # Unpack models from bundle or handle legacy single model
    if isinstance(model, dict):
        imagenet_model = model.get("imagenet_model")
        fine_tuned_model = model.get("fine_tuned_model")
        categories = model.get("categories")
    else:
        if model_type == "fine_tuned":
            fine_tuned_model = model
            weights = models.MobileNet_V2_Weights.DEFAULT
            imagenet_model = models.mobilenet_v2(weights=weights).eval()
            categories = weights.meta['categories']
        else:
            imagenet_model = model
            fine_tuned_model = None
            weights = models.MobileNet_V2_Weights.DEFAULT
            categories = weights.meta['categories']

    with torch.no_grad():
        # 1. Open-world 1000-class ImageNet inference
        out_imgnet = imagenet_model(tensor)
        probs_imgnet = torch.softmax(out_imgnet[0], dim=0)

        # 2. Fine-tuned 2-class inference if available
        ft_cat = None
        ft_dog = None
        if fine_tuned_model is not None:
            out_ft = fine_tuned_model(tensor)
            probs_ft = torch.softmax(out_ft[0], dim=0)
            ft_cat = float(probs_ft[0].item())
            ft_dog = float(probs_ft[1].item())

    top5_probs, top5_idx = torch.topk(probs_imgnet, 5)
    top1_idx = top5_idx[0].item()
    top1_prob = float(top5_probs[0].item())
    top5_indices = [i.item() for i in top5_idx]

    detected_object = categories[top1_idx].replace('_', ' ').title()

    cat_prob = float(sum(probs_imgnet[i].item() for i in CAT_INDICES))
    dog_prob = float(sum(probs_imgnet[i].item() for i in DOG_INDICES))

    top1_is_cat = top1_idx in CAT_INDICES
    top1_is_dog = top1_idx in DOG_INDICES

    cat_in_top3 = any(i in CAT_INDICES for i in top5_indices[:3])
    dog_in_top3 = any(i in DOG_INDICES for i in top5_indices[:3])

    has_cat_dog_top5 = any(i in CAT_INDICES or i in DOG_INDICES for i in top5_indices)
    top1_is_related = top1_idx in RELATED_MAMMALS

    # ---------------------------------------------------------
    # Detection & Classification Logic
    # ---------------------------------------------------------
    if top1_is_cat:
        label = "Cat"
        emoji = "🐱"
        confidence = (ft_cat * 100) if ft_cat is not None else max(cat_prob * 100, top1_prob * 100)
    elif top1_is_dog:
        label = "Dog"
        emoji = "🐶"
        confidence = (ft_dog * 100) if ft_dog is not None else max(dog_prob * 100, top1_prob * 100)
    elif cat_in_top3 and cat_prob > dog_prob and cat_prob > 0.03:
        label = "Cat"
        emoji = "🐱"
        confidence = (ft_cat * 100) if ft_cat is not None else (cat_prob * 100)
    elif dog_in_top3 and dog_prob > cat_prob and dog_prob > 0.15:
        label = "Dog"
        emoji = "🐶"
        confidence = (ft_dog * 100) if ft_dog is not None else (dog_prob * 100)
    elif (has_cat_dog_top5 or top1_is_related) and ft_cat is not None and (ft_cat > 0.85 or ft_dog > 0.85):
        if ft_cat > ft_dog:
            label = "Cat"
            emoji = "🐱"
            confidence = ft_cat * 100
        else:
            label = "Dog"
            emoji = "🐶"
            confidence = ft_dog * 100
    else:
        # Non-cat / non-dog item: Return Unknown
        label = "Unknown"
        emoji = "❓"
        confidence = top1_prob * 100

    # Probability metrics formatting
    if label == "Cat":
        if ft_cat is not None:
            cat_score = ft_cat * 100
            dog_score = ft_dog * 100
        else:
            total = cat_prob + dog_prob
            cat_score = (cat_prob / total * 100) if total > 0 else 85.0
            dog_score = 100.0 - cat_score
    elif label == "Dog":
        if ft_dog is not None:
            dog_score = ft_dog * 100
            cat_score = ft_cat * 100
        else:
            total = cat_prob + dog_prob
            dog_score = (dog_prob / total * 100) if total > 0 else 85.0
            cat_score = 100.0 - dog_score
    else:
        # Unknown: cat and dog probabilities are naturally near 0
        cat_score = min(cat_prob * 100, 15.0)
        dog_score = min(dog_prob * 100, 15.0)

    return {
        "label": label,
        "emoji": emoji,
        "confidence": confidence,
        "cat_score": cat_score,
        "dog_score": dog_score,
        "detected_object": detected_object,
        "is_cat_or_dog": label in ["Cat", "Dog"]
    }

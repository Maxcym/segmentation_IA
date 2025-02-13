import torch
import cv2
import numpy as np
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Mapping des modèles et leurs encoders correspondants
models_paths = {
    "unet_resnet50": "C:/Users/maxim/PycharmProjects/segmentation_IA/models/unet_resnet50.pth",
    "unet_vgg16": "C:/Users/maxim/PycharmProjects/segmentation_IA/models/unet_vgg16.pth",
}

encoder_mapping = {
    "unet_resnet50": "resnet50",
    "unet_vgg16": "vgg16"
}

def load_model(model_name):
    """ Charge dynamiquement le modèle U-Net avec le bon encoder """
    if model_name not in models_paths:
        raise ValueError("Modèle non trouvé.")

    encoder_name = encoder_mapping[model_name]

    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=None,
        in_channels=3,
        classes=1,
        decoder_attention_type="scse"
    )

    model.load_state_dict(torch.load(models_paths[model_name], map_location=device))
    model.to(device)
    model.eval()  # Mode évaluation
    return model

def preprocess_image(image_path):
    """ Charge et prétraite une image pour la prédiction """
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    transform = A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    augmented = transform(image=image)
    image_tensor = augmented["image"].unsqueeze(0)  # Ajouter la dimension batch
    return image_tensor, image

def predict(model, image_tensor):
    """ Prédit le masque pour une image donnée """
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        output = model(image_tensor)
        output = torch.sigmoid(output)  # Activation sigmoid pour obtenir des valeurs entre 0 et 1
        output = output.cpu().numpy()[0, 0]  # Retirer la dimension batch et canal
    return output

def overlay_mask(image, mask):
    """ Superpose uniquement la segmentation en rouge sur l'image originale """
    mask = (mask > 0.5).astype(np.uint8)  # Seuil pour binariser le masque

    # Créer une image de masque de la même taille que l'image originale
    mask_colored = np.zeros_like(image)

    # Appliquer la couleur rouge uniquement sur les pixels segmentés
    mask_colored[:, :, 2] = mask * 255

    # Fusionner le masque avec l'image originale
    result = image.copy()
    result[mask == 1] = cv2.addWeighted(image[mask == 1], 0.5, mask_colored[mask == 1], 0.5, 0)

    return result

def compute_segmented_surface_from_mask(mask, pixel_to_mm_ratio=0.1):
    """
    Calcule la surface segmentée en pixels et en mm² depuis le masque de sortie du modèle.
    :param mask: Tableau numpy du masque segmenté (valeurs entre 0 et 1)
    :param pixel_to_mm_ratio: Facteur de conversion pixel -> mm² (ajuster selon l'image)
    :return: Nombre de pixels segmentés et surface en mm²
    """
    # Seuil pour binariser le masque (les pixels > 0.5 sont considérés comme segmentés)
    binary_mask = (mask > 0.5).astype(np.uint8)

    # Nombre de pixels segmentés
    segmented_pixels = np.count_nonzero(binary_mask)

    # Conversion en surface réelle
    segmented_surface = segmented_pixels * (pixel_to_mm_ratio ** 2)

    return segmented_pixels, segmented_surface

def dice_score(y_true, y_pred):
    intersection = np.sum(y_true * y_pred)
    return (2. * intersection) / (np.sum(y_true) + np.sum(y_pred))

def iou_score(y_true, y_pred):
    intersection = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred) - intersection
    return intersection / union if union > 0 else 0


import argparse
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
import torch.nn as nn
import torch.optim as optim
import os
import glob
import json
import cv2
import numpy as np
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt


def preprocess_images(xml_dir, image_dir, output_image_dir, output_mask_dir, filtered_output_dir, target_size=(256, 256), padding_factor=1.5):
    """
        Traite les annotations XML et extrait les régions annotées en images et masques.

        Args:
            xml_dir (str): Répertoire contenant les fichiers XML d'annotations.
            image_dir (str): Répertoire contenant les images d'entrée.
            output_image_dir (str): Répertoire pour enregistrer les images recadrées.
            output_mask_dir (str): Répertoire pour enregistrer les masques recadrés.
            target_size (tuple): Taille cible pour les images et masques recadrés.
            padding_factor (float): Facteur d'agrandissement des boîtes englobantes.
    """

    # Création des dossiers de sortie
    os.makedirs(output_image_dir, exist_ok=True)
    os.makedirs(output_mask_dir, exist_ok=True)
    os.makedirs(filtered_output_dir, exist_ok=True)

    # Récupérer la liste des fichiers XML
    xml_files = glob.glob(os.path.join(xml_dir, "*.xml"))
    print(f"{len(xml_files)} fichiers XML trouvés.")

    for xml_file in xml_files:
        try:
            tree = ET.parse(xml_file)
        except Exception as e:
            print(f"Erreur lors du parsing de {xml_file} : {e}")
            continue

        root = tree.getroot()
        case_number_elem = root.find('number')
        if case_number_elem is None or case_number_elem.text is None:
            print(f"Numéro de cas introuvable dans {xml_file}.")
            continue
        case_number = case_number_elem.text.strip()

        for mark in root.findall('mark'):
            image_tag = mark.find('image')
            if image_tag is None or image_tag.text is None:
                continue
            image_number = image_tag.text.strip()

            image_filename = f"{case_number}_{image_number}.jpg"
            image_path = os.path.join(image_dir, image_filename)
            if not os.path.exists(image_path):
                continue

            image = cv2.imread(image_path)
            if image is None:
                continue

            svg_elem = mark.find('svg')
            if svg_elem is None or not svg_elem.text:
                continue

            try:
                svg_data = json.loads(svg_elem.text.strip())
            except Exception as e:
                continue

            if not isinstance(svg_data, list):
                continue

            for region_idx, region in enumerate(svg_data):
                points_data = region.get("points", [])
                if not points_data:
                    continue

                xs, ys = zip(*[(point["x"], point["y"]) for point in points_data if "x" in point and "y" in point])

                if not xs or not ys:
                    continue

                center_x, center_y = int(np.mean(xs)), int(np.mean(ys))
                points = np.array(list(zip(xs, ys)), dtype=np.int32)
                x, y, w, h = cv2.boundingRect(points)

                side_length = int(max(w, h) * padding_factor)
                half_side = side_length // 2

                x1, y1 = max(center_x - half_side, 0), max(center_y - half_side, 0)
                x2, y2 = min(center_x + half_side, image.shape[1]), min(center_y + half_side, image.shape[0])
                cropped = image[y1:y2, x1:x2]

                resized_crop = cv2.resize(cropped, target_size, interpolation=cv2.INTER_LANCZOS4)
                output_filename = f"{case_number}_{image_number}_region{region_idx}_cropped.jpg"
                output_path = os.path.join(output_image_dir, output_filename)
                cv2.imwrite(output_path, resized_crop)

                mask = np.zeros(image.shape[:2], dtype=np.uint8)
                cv2.fillPoly(mask, [points], 255)
                cropped_mask = mask[y1:y2, x1:x2]
                resized_mask = cv2.resize(cropped_mask, target_size, interpolation=cv2.INTER_NEAREST)

                mask_filename = f"{case_number}_{image_number}_region{region_idx}_mask.jpg"
                mask_path = os.path.join(output_mask_dir, mask_filename)
                cv2.imwrite(mask_path, resized_mask)

    # --- Filtrage des images ---
    def load_image(image_path):
        """
            Charge une image en niveaux de gris.

            Args:
                image_path (str): Chemin du fichier image.

            Returns:
                numpy.ndarray: Image en niveaux de gris.
        """
        return cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    def apply_nl_means_filter(img, h=10, template_window_size=7, search_window_size=21):
        """
            Applique un filtre NL-Means pour le débruitage.

            Args:
                img (numpy.ndarray): Image d'entrée.
                h (int): Paramètre de filtrage.
                template_window_size (int): Taille de la fenêtre modèle.
                search_window_size (int): Taille de la fenêtre de recherche.

            Returns:
                numpy.ndarray: Image filtrée.
        """
        return cv2.fastNlMeansDenoising(img, None, h, template_window_size, search_window_size)

    def enhance_contrast(img):
        """
            Améliore le contraste d'une image avec CLAHE.

            Args:
                img (numpy.ndarray): Image d'entrée.

            Returns:
                numpy.ndarray: Image avec un contraste amélioré.
        """
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        return clahe.apply(img)

    image_files = [f for f in os.listdir(output_image_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    for image_file in image_files:
        image_path = os.path.join(output_image_dir, image_file)
        img = load_image(image_path)
        if img is None:
            continue

        denoised_img = apply_nl_means_filter(img)
        enhanced_img = enhance_contrast(denoised_img)

        output_path = os.path.join(filtered_output_dir, image_file)
        cv2.imwrite(output_path, enhanced_img)
        print(f"Image filtrée enregistrée : {output_path}")

    print("Prétraitement terminé !\n")

# --- Annotation des images

def process_annotations(xml_dir, image_dir, output_dir):
    """
        Traite les annotations XML et applique les annotations sur les images correspondantes.

        Args:
            xml_dir (str): Répertoire contenant les fichiers XML d'annotations.
            image_dir (str): Répertoire contenant les images d'entrée.
            output_dir (str): Répertoire où enregistrer les images annotées.
    """
    os.makedirs(output_dir, exist_ok=True)
    xml_files = glob.glob(os.path.join(xml_dir, "*.xml"))
    print(f"{len(xml_files)} fichiers XML trouvés.")

    for xml_file in xml_files:
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
        except Exception as e:
            print(f"Erreur lors du parsing de {xml_file} : {e}")
            continue

        case_number_elem = root.find('number')
        if case_number_elem is None or case_number_elem.text is None:
            print(f"Numéro de cas introuvable dans {xml_file}.")
            continue
        case_number = case_number_elem.text.strip()

        for mark in root.findall('mark'):
            image_tag = mark.find('image')
            if image_tag is None or image_tag.text is None:
                print(f"Balise <image> manquante dans {xml_file}.")
                continue
            image_number = image_tag.text.strip()

            image_filename = f"{case_number}_{image_number}.jpg"
            image_path = os.path.join(image_dir, image_filename)
            if not os.path.exists(image_path):
                print(f"Image {image_path} introuvable.")
                continue

            image = cv2.imread(image_path)
            if image is None:
                print(f"Erreur lors du chargement de l'image {image_path}.")
                continue

            svg_elem = mark.find('svg')
            if svg_elem is None or not svg_elem.text:
                print(f"Pas d'annotation SVG trouvée dans {xml_file} pour {image_filename}.")
                continue

            try:
                svg_data = json.loads(svg_elem.text.strip())
            except Exception as e:
                print(f"Erreur lors du parsing du JSON dans {xml_file} pour {image_filename} : {e}")
                continue

            if not isinstance(svg_data, list):
                print(f"Format JSON inattendu dans {xml_file} pour {image_filename}.")
                continue

            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            for region_idx, region in enumerate(svg_data):
                points_data = region.get("points", [])
                if not points_data:
                    print(f"Aucun point trouvé pour la région {region_idx} dans {image_filename}.")
                    continue

                xs, ys = [], []
                for point in points_data:
                    x_coord = point.get("x")
                    y_coord = point.get("y")
                    if x_coord is not None and y_coord is not None:
                        xs.append(x_coord)
                        ys.append(y_coord)

                if not xs or not ys:
                    print(f"Coordonnées insuffisantes pour la région {region_idx} dans {image_filename}.")
                    continue

                pts = np.array(list(zip(xs, ys)), dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(gray_image, [pts], isClosed=True, color=255, thickness=2)

            annotated_filename = f"{case_number}_{image_number}_annotated.jpg"
            annotated_path = os.path.join(output_dir, annotated_filename)
            cv2.imwrite(annotated_path, gray_image)
            print(f"Image annotée enregistrée : {annotated_path}")

# --- Entrainement ---

# --- 1. Définition du Dataset ---
class SegmentationDataset(Dataset):
    """
        Dataset personnalisé pour la segmentation d'images.

        Args:
            images_paths (list): Liste des chemins des images.
            masks_paths (list): Liste des chemins des masques.
            transform (callable, optional): Transformation à appliquer aux images et aux masques.
    """
    def __init__(self, images_paths, masks_paths, transform=None):
        self.images_paths = images_paths
        self.masks_paths = masks_paths
        self.transform = transform

    def __len__(self):
        return len(self.images_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.images_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.masks_paths[idx], cv2.IMREAD_GRAYSCALE)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        else:
            image = ToTensorV2()(image=image)['image']
            mask = torch.tensor(mask, dtype=torch.float).unsqueeze(0)

        mask = mask.float() / 255.0  # Normalisation 0-1
        return image, mask


# --- 2. Fonctions Utilitaires ---
def get_dataloaders(batch_size=8, num_workers=4):
    """
    Charge les images, divise les datasets en ensembles d'entraînement, validation et test,
    et retourne les DataLoaders correspondants.

    Args:
        batch_size (int): Taille des lots.
        num_workers (int): Nombre de processus pour le chargement des données.

    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    images_paths = sorted(glob.glob(os.path.join("../data/train_data", "cropped_filtered_images", "*.jpg")))
    masks_paths  = sorted(glob.glob(os.path.join("../data/train_data", "cropped_masks", "*.jpg")))

    assert len(images_paths) == len(masks_paths), "Le nombre d'images et de masques doit être identique."

    train_imgs, temp_imgs, train_masks, temp_masks = train_test_split(images_paths, masks_paths, test_size=0.2, random_state=42)
    val_imgs, test_imgs, val_masks, test_masks = train_test_split(temp_imgs, temp_masks, test_size=0.5, random_state=42)

    train_transform = A.Compose([
        A.Resize(256, 256),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    val_transform = A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    train_dataset = SegmentationDataset(train_imgs, train_masks, transform=train_transform)
    val_dataset   = SegmentationDataset(val_imgs, val_masks, transform=val_transform)
    test_dataset  = SegmentationDataset(test_imgs, test_masks, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader


def create_model(encoder_name):
    """
    Crée et retourne un modèle U-Net avec un encodeur ResNet-50.

    Returns:
        torch.nn.Module: Modèle U-Net.
    """
    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        decoder_attention_type="scse"
    )
    return model


def train_model(model, train_loader, val_loader, device, encoder_name, num_epochs=30, lr=1e-4):
    """ Entraîne le modèle et sauvegarde les poids avec le nom de l'encodeur. """
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device).unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)

        train_loss /= len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {train_loss:.4f}")

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device).unsqueeze(1)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item() * images.size(0)

        val_loss /= len(val_loader.dataset)
        print(f"Validation Loss: {val_loss:.4f}")

    # --- Sauvegarde du modèle ---
    model_save_path = f"../models/unet_{encoder_name}.pth"
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save(model.state_dict(), model_save_path)
    print(f"Modèle sauvegardé à : {model_save_path}")



def load_model(model_path, device):
    """
        Charge un modèle U-Net pré-entraîné pour la segmentation.

        Args:
            model_path (str): Chemin vers le fichier des poids du modèle.
            device (torch.device): Périphérique sur lequel charger le modèle (CPU ou GPU).

        Returns:
            torch.nn.Module: Modèle chargé et mis en mode évaluation.
    """
    model = smp.Unet(
        encoder_name="resnet50",
        encoder_weights=None,
        in_channels=3,
        classes=1,
        decoder_attention_type="scse"
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


# --- 2. Préparer une image pour le test ---
def preprocess_image(image_path):
    """
        Pré-traite une image pour l'inférence du modèle.

        Args:
            image_path (str): Chemin de l'image à traiter.

        Returns:
            tuple: Tenseur d'image prêt pour l'inférence, image originale sous forme de tableau NumPy.
    """
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    transform = A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    augmented = transform(image=image)
    image_tensor = augmented["image"].unsqueeze(0)
    return image_tensor, image


# --- 3. Effectuer la prédiction ---
def predict(model, image_tensor, device):
    """
        Effectue une prédiction de segmentation sur une image donnée.

        Args:
            model (torch.nn.Module): Modèle de segmentation.
            image_tensor (torch.Tensor): Image pré-traitée sous forme de tenseur.
            device (torch.device): Périphérique d'inférence.

        Returns:
            numpy.ndarray: Masque prédit sous forme d'un tableau 2D.
    """
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        output = model(image_tensor)
        output = torch.sigmoid(output)
        output = output.cpu().numpy()[0, 0]
    return output


# --- 4. Superposer le masque sur l'image ---
def overlay_mask(image, mask, alpha=0.5):
    """
        Superpose un masque sur une image d'origine.

        Args:
            image (numpy.ndarray): Image originale.
            mask (numpy.ndarray): Masque prédit.
            alpha (float): Facteur de transparence pour la superposition.

        Returns:
            numpy.ndarray: Image avec le masque superposé.
    """
    mask = (mask > 0.5).astype(np.uint8)
    mask_colored = np.zeros_like(image)
    mask_colored[:, :, 0] = mask * 255
    blended = cv2.addWeighted(image, 1 - alpha, mask_colored, alpha, 0)
    return blended


# --- 5. Récupérer le masque de référence ---
def get_ground_truth_mask(original_image_filename):
    """
        Charge le masque de référence (ground truth) correspondant à une image.

        Args:
            original_image_filename (str): Nom du fichier image original.

        Returns:
            numpy.ndarray or None: Masque ground truth normalisé ou None s'il n'existe pas.
    """
    mask_filename = original_image_filename.replace("cropped.jpg", "mask.jpg")
    mask_filepath = os.path.join("../data/train_data/cropped_masks",
                                 mask_filename)
    if os.path.exists(mask_filepath):
        return cv2.imread(mask_filepath, cv2.IMREAD_GRAYSCALE) / 255.0
    return None


# --- 6. Calcul des métriques ---
def dice_score(y_true, y_pred):
    """
        Calcule le Dice Score entre le masque de vérité terrain et le masque prédit.

        :param y_true: Masque de vérité terrain.
        :param y_pred: Masque prédit.
        :return: Dice Score.
    """
    intersection = np.sum(y_true * y_pred)
    return (2. * intersection) / (np.sum(y_true) + np.sum(y_pred))


def iou_score(y_true, y_pred):
    """
        Calcule le score d'intersection sur union (IoU) entre le masque de vérité terrain et le masque prédit.

        :param y_true: Masque de vérité terrain.
        :param y_pred: Masque prédit.
        :return: IoU Score.
    """
    intersection = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred) - intersection
    return intersection / union if union > 0 else 0


# --- 7. Visualisation des résultats ---
def visualize(original, mask, overlay, ground_truth=None):
    """
        Affiche les images et masques pour évaluation visuelle.

        :param original: Image originale.
        :param mask: Masque prédit.
        :param overlay: Image avec masque superposé.
        :param ground_truth: Masque de vérité terrain (optionnel).
    """
    plt.imshow(original)
    plt.title("Image Originale")
    plt.axis("off")
    plt.show()

    plt.imshow(mask, cmap="gray")
    plt.title("Masque Prédit")
    plt.axis("off")
    plt.show()

    if ground_truth is not None:
        plt.imshow(ground_truth, cmap="gray")
        plt.title("Masque Original (Ground Truth)")
        plt.axis("off")
        plt.show()

    plt.imshow(overlay)
    plt.title("Image + Masque")
    plt.axis("off")
    plt.show()


# --- 8. Analyse des performances en fonction de la taille du nodule ---
def analyze_performance(image_paths, model, device):
    """
        Analyse la performance du modèle en fonction de la taille des nodules segmentés.

        :param image_paths: Liste des chemins des images.
        :param model: Modèle de segmentation.
        :param device: Périphérique utilisé pour l'inférence.
    """
    sizes = []  # Taille des nodules en pixels
    dice_scores = []
    iou_scores = []

    for image_path in image_paths:
        image_tensor, original_image = preprocess_image(image_path)
        predicted_mask = predict(model, image_tensor, device)
        ground_truth_mask = get_ground_truth_mask(os.path.basename(image_path))

        if ground_truth_mask is not None:
            dice = dice_score(ground_truth_mask, predicted_mask)
            iou = iou_score(ground_truth_mask, predicted_mask)
            size = np.sum(predicted_mask > 0.5)  # Nombre de pixels segmentés

            sizes.append(size)
            dice_scores.append(dice)
            iou_scores.append(iou)

    # Calculer les statistiques du Dice Score
    dice_mean = np.mean(dice_scores)
    dice_median = np.median(dice_scores)
    dice_std = np.std(dice_scores)
    dice_variance = np.var(dice_scores)

    print(f"Dice Score Moyenne: {dice_mean:.4f}")
    print(f"Dice Score Médiane: {dice_median:.4f}")
    print(f"Dice Score Écart-Type: {dice_std:.4f}")
    print(f"Dice Score Variance: {dice_variance:.4f}")

    iou_mean = np.mean(iou_scores)
    iou_median = np.median(iou_scores)
    iou_std = np.std(iou_scores)
    iou_variance = np.var(iou_scores)

    print(f"Iou Score Moyenne: {iou_mean:.4f}")
    print(f"Iou Score Médiane: {iou_median:.4f}")
    print(f"Iou Score Écart-Type: {iou_std:.4f}")
    print(f"Iou Score Variance: {iou_variance:.4f}")

if __name__ == '__main__':
    # --- Argument parsing ---
    parser = argparse.ArgumentParser(description="Process XML annotations and images.")
    parser.add_argument("--encoder", type=str, default="resnet50", help="Nom de l'encodeur pour le modèle U-Net.")

    args = parser.parse_args()

    # Définition des chemins
    output_image_dir = "../data/cropped_images"
    output_mask_dir = "../data/train_data/cropped_masks"
    filtered_output_dir = "../data/train_data/cropped_filtered_images"

    print("Annotation des images...\n")

    process_annotations("../data/annotations", "../data/images", "../data/annotated_images")

    # Lancer le prétraitement
    preprocess_images("../data/annotations", "../data/images", output_image_dir, output_mask_dir, filtered_output_dir)

    print("Démarrage de l'entraînement...\n")

    # Vérifier CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Utilisation du périphérique : {device}")
    if device.type == "cuda":
        print(f"Nombre de GPU : {torch.cuda.device_count()}")
        print(f"Nom du GPU : {torch.cuda.get_device_name(0)}")

    # Charger les données
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=8, num_workers=4)

    encoder_name = args.encoder

    # Créer le modèle
    model = create_model(encoder_name)

    # Entraîner le modèle
    train_model(model, train_loader, val_loader, device,encoder_name, num_epochs=2)


    print("Entraînement terminé ! \n")

    # --- affichage des métriques ---

    print("Analyse des performances du modèle...\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = f"../models/unet_{encoder_name}.pth"
    model = load_model(model_path, device)

    # Charger plusieurs images pour l'analyse
    image_folder = '../data/cropped_images'
    image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith("_cropped.jpg")]

    # Analyser la performance du modèle en fonction de la taille des nodules
    analyze_performance(image_paths, model, device)

    print("Analyse terminé !")

    print("Fin du pipeline")



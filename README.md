# Segmentation des nodules thyroïdiens

## Contexte

Les **nodules thyroïdiens** sont des masses qui apparaissent dans la glande thyroïde et peuvent être bénignes ou malignes.  
L’échographie est l’examen clé pour leur détection et classification, mais son interprétation reste subjective et dépend de l’expertise du radiologue.  

L'objectif est d'automatiser la segmentation des nodules pour :  
- Améliorer la précision et la reproductibilité du diagnostic
- Réduire le temps d’analyse des images  
- Aider à une meilleure classification des nodules  

## Structure du projet

- **data/**  
  Contient les **images brutes** et les **annotations**  
  (**disponibles ici :** [Google Drive](https://drive.google.com/drive/folders/1wIlOX3atqCiQv7KWhndW3s0eqCqN4K4B))

- **notebooks/**  
  Notebook Jupyter pour tester divers codes en lien avec le projet.

- **models/**  
  Modèles entraînés (**U-Net + ResNet-50**, **U-Net + VGG16**)  
  (**disponibles ici :** [Google Drive](https://drive.google.com/drive/folders/1FAaUSJmr9F6cvXhmgnZb4oPP82qOFjJF))

- **app/**  
  Code de l’application web pour tester la segmentation en ligne.

- **scripts/**  
  Scripts d’automatisation pour :  
  - Le prétraitement des images  
  - L’entraînement des modèles  
  - L’évaluation des performances

## Installation et utilisation

### 1. Cloner le projet et créer les dossiers nécessaires

```bash
git clone 'https://github.com/Maxcym/segmentation_IA'
cd segmentation_IA
mkdir data
mkdir models
```

### 2. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 3. Préparer les données 

- Placez vos images d’échographie dans le dossier data/images/. (Disponibles sur le google drive dont le lien est au dessus)
- Ajoutez les fichiers d’annotations XML dans le dossier data/annotations/. (Disponibles sur le google drive dont le lien est au dessus)

### 4. Lancer le pipeline

Le script principal orchestre le prétraitement, l’entraînement du modèle et l’analyse des performances. Pour lancer la pipeline complète, utilisez les commandes suivantes en précisant l’encodeur souhaité (ici resnet50) :

```bash
cd scripts
./pipeline_segmentation.py --encoder resnet50
```
**Attention** : il se peut que le chemin menant aux fichiers utilisés ne soient pas valides. Si cela est le cas, il faut supprimer les "../" devant le chemin des fichiers pour que le script fonctionne. 

## 5. Visualiser les résultats

Une fois le script exécuté, vous verrez s’afficher dans le terminal :

Les étapes de prétraitement (annotation, recadrage et filtrage des images).
Les statistiques d’entraînement (loss d’entraînement et validation).
Les métriques d’évaluation (Dice Score, IoU, etc.) pour analyser la performance du modèle.

Vous obtiendrez également différents dossiers contenant les données prétraitées et le modèle entraîné :

- ../data/annotated_images/ : Images originales avec annotations superposées.
- ../data/train_data/cropped_images/ : Images recadrées à partir des annotations.
- ../data/train_data/cropped_masks/ : Masques de segmentation correspondants.
- ../data/train_data/cropped_filtered_images/ : Images prétraitées (filtrage et amélioration du contraste).
- ../models/ : Modèles entraînés enregistrés (ex. unet_resnet50.pth).

## 6. Lancer l'application web

Vous pouvez maintenant démarrer l’application web en éxécutant le fichier python app.py se trouvant dans le dossier app



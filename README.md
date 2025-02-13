# Segmentation des nodules thyroïdiens

## Contexte

Les **nodules thyroïdiens** sont des masses qui apparaissent dans la glande thyroïde et peuvent être bénignes ou malignes.  
L’échographie est l’examen clé pour leur **détection et classification**, mais son interprétation reste subjective et dépend de l’expertise du radiologue.  

**Objectif du projet :** Automatiser la segmentation des nodules pour :  
- Améliorer la précision et la reproductibilité du diagnostic
- Réduire le temps d’analyse des images  
- Aider à une meilleure classification des nodules  

## Objectifs

- Détecter et segmenter automatiquement les nodules thyroïdiens sur des images d’échographie
- Améliorer la précision du contour des nodules par rapport aux méthodes manuelles
- Évaluer la performance du modèle par rapport aux annotations d’experts
- Intégrer la segmentation dans une **application web interactive**

---

## Structure du Projet

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

## Installation et Utilisation

### 1. Cloner le projet

```bash
git clone 'https://github.com/Maxcym/segmentation_IA/tree/main'
cd segmentation_IA
```

### 2. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 3. Préparer les données 

- Placez vos images d’échographie dans le dossier data/images/.
- Ajoutez les fichiers d’annotations XML dans le dossier data/annotations/.

## 4. Lancer le pipeline

Le script principal orchestre le prétraitement, l’entraînement du modèle et l’analyse des performances. Pour lancer la pipeline complète, utilisez les commandes suivantes en précisant l’encodeur souhaité (ici resnet50) :

```bash
cd scripts
./pipeline_segmentation.py --encoder resnet50
```

## 5. Visualiser les Résultats

Une fois le script exécuté, vous verrez s’afficher dans le terminal :

Les étapes de prétraitement (annotation, recadrage et filtrage des images).
Les statistiques d’entraînement (loss d’entraînement et validation).
Les métriques d’évaluation (Dice Score, IoU, etc.) pour analyser la performance du modèle.

##6. Lancer l'application web

Vous pouvez démarrer l’application web en éxécutant le fichier python app.py se trouvant dans le dossier app



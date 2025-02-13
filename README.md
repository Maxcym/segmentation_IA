# Segmentation des Nodules Thyroïdiens

## Contexte

Les **nodules thyroïdiens** sont des masses qui apparaissent dans la glande thyroïde et peuvent être bénignes ou malignes.  
L’échographie est l’examen clé pour leur **détection et classification**, mais son interprétation reste subjective et dépend de l’expertise du radiologue.  

🔹 **Objectif du projet :** Automatiser la segmentation des nodules pour :  
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

**data/**  
> Contient les **images brutes** et les **annotations** pour l'entraînement (📌 **disponibles sur Google Drive**)

**notebooks/**  
> Contient des notebooks Jupyter pour tester et analyser les performances des modèles  

**models/**  
> Contient les modèles pré-entraînés (**U-Net + ResNet-50**, **U-Net + VGG16**) (**📌 disponibles sur Google Drive**)  

**app/**  
> Code de l’application web permettant d’exécuter la segmentation  

**scripts/**  
> Contient des scripts d’automatisation pour :  
> 🔹 Prétraitement des images  
> 🔹 Entraînement des modèles  
> 🔹 Évaluation des performances  

**scripts/pipeline_segmentation.py**  
> Un script automatique pour exécuter la segmentation de bout en bout  

**requirements.txt**  
> Liste des bibliothèques Python requises pour exécuter le projet  

---

## Installation et Utilisation

### Cloner le projet

```bash
git clone <URL_DU_REPO>
cd segmentation_IA

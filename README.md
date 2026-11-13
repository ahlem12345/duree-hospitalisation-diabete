# Prédiction de la durée d’hospitalisation des patients diabétiques

Projet de Machine Learning (méthode **CRISP-DM**), réalisé à Esprit (Tunisie), sur le jeu de données *Diabetes 130-US Hospitals (1999-2008)*.

**Objectif métier :** optimiser la gestion des lits hospitaliers en anticipant la durée de séjour des patients diabétiques.
**Objectif data science :** prédire la durée d’hospitalisation (`time_in_hospital`, de 1 à 14 jours) à partir des données disponibles à l’admission. C’est un problème de **régression**.

## Résultats (jeu de test, 14 303 séjours)

| Modèle | MAE (jours) | RMSE | R² |
|---|---|---|---|
| Moyenne (référence) | 2,31 | 2,94 | 0,00 |
| Ridge (interprétable) | 2,17 | 2,80 | 0,09 |
| **HistGradientBoosting (principal)** | **2,13** | **2,76** | **0,12** |

- Le modèle fait mieux que la simple moyenne, avec un gain réel mais modeste.
- **Limite principale :** les séjours longs (8 à 14 jours) sont sous-estimés d’environ 5 jours. Un test de diagnostic montre que l’information disponible à l’admission est limitée (R² de 0,12, contre 0,45 si l’on ajoutait des variables connues seulement à la fin du séjour).
- C’est une **aide à la planification des lits**, pas un outil médical.

## Démarche

1. **Business Understanding** : objectifs métier et data science.
2. **Data Understanding** : 101 766 séjours × 50 colonnes, qualité des données (`weight` manquant à 97 %, patients revenus plusieurs fois).
3. **Data Preparation** : un séjour par patient, retrait des variables qui trichent (fuite de données), valeurs extrêmes plafonnées, sélection de 20 variables sur 38, encodage et normalisation.
4. **Modélisation** : trois modèles comparés en validation croisée (5 plis), réglage par GridSearchCV et RandomizedSearchCV.
5. **Évaluation** : MAE, RMSE, R² sur le test, erreur selon la durée du séjour, importance des variables.
6. **Déploiement** : fonction `predire(patient)` et API FastAPI.

Variables les plus importantes : l’insuline, le diagnostic principal et l’âge.

## Structure du dépôt

```
notebooks/projet_ml_duree_hospitalisation.ipynb   le projet complet, exécuté
api/                                              API FastAPI + modèle final (voir api/README.md)
requirements.txt                                  dépendances du notebook
```

## Lancer le notebook

1. Télécharger le jeu de données **Diabetes 130-US Hospitals for Years 1999-2008** (UCI Machine Learning Repository) et placer `diabetic_data.csv` à côté du notebook. Sur Google Colab, le notebook demande le fichier à l’exécution.
2. Installer les dépendances :
   ```bash
   pip install -r requirements.txt
   ```
3. Ouvrir `notebooks/projet_ml_duree_hospitalisation.ipynb` et choisir « Exécuter tout ».

## Lancer l’API

```bash
cd api
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```
Documentation interactive : http://localhost:8000/docs

Exemple :
```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"age_num": 65, "diag_groupe": "Tumeurs", "insulin": "Up"}'
# {"duree_prevue_jours": 6.8, ...}
```

## Auteure

Ahlem Bensalem, Esprit. Projet réalisé en groupe ; ce dépôt présente la partie modélisation, évaluation et déploiement.

## Données

Source : UCI Machine Learning Repository, *Diabetes 130-US Hospitals for Years 1999-2008* (Strack et al., 2014). Le fichier n’est pas inclus dans ce dépôt.

# Comparaison : Découpage Train/Val/Test vs Validation Croisée Stratifiée K-Fold

Même configuration de modèle pour les deux approches : entrée complète 16×16 (256 valeurs), couche cachée de taille 64, 20 époques, taux d'apprentissage 0,1.

---

## Différence de configuration

| | Découpage Train/Val/Test | Validation croisée 5-Fold stratifiée |
|---|---|---|
| **Échantillons d'entraînement** | 300 (entraînement uniquement) | 320 par fold / 400 pour le modèle final |
| **Échantillons de validation** | 100 (fixe) | 80 par fold (en rotation) |
| **Échantillons de test** | 100 (réservés) | 100 (réservés) |
| **Évaluations** | 1 | 5 folds + 1 test final |
| **Modèle sauvegardé** | Entraîné sur 300 | Entraîné sur 400 |

---

## Métriques sur l'ensemble de test

### Accuracy

| | Split | K-Fold (moyenne CV) | K-Fold (test) |
|---|---|---|---|
| **Accuracy** | 0,96 | 0,9225 ± 0,0436 | 0,95 |

### Précision par fold (K-Fold)

| Fold | Accuracy |
|---|---|
| Fold 1 | 0,9375 |
| Fold 2 | 0,85 |
| Fold 3 | 0,975 |
| Fold 4 | 0,90 |
| Fold 5 | 0,95 |
| **Moyenne** | **0,9225 ± 0,0436** |

---

## Métriques par classe (ensemble de test)

### Précision

| Classe | Split | K-Fold |
|---|---|---|
| **Souriant** | 0,95 | 1,00 |
| **Neutre** | 0,9524 | 0,9048 |
| **Triste** | 0,95 | 0,9048 |
| **En colère** | 0,95 | 0,95 |
| **Surpris** | 1,00 | 1,00 |
| **Macro** | **0,9605** | **0,9519** |

### Rappel

| Classe | Split | K-Fold |
|---|---|---|
| **Souriant** | 0,95 | 0,90 |
| **Neutre** | 1,00 | 0,95 |
| **Triste** | 0,95 | 0,95 |
| **En colère** | 0,95 | 0,95 |
| **Surpris** | 0,95 | 1,00 |
| **Macro** | **0,96** | **0,95** |

### F1

| Classe | Split | K-Fold |
|---|---|---|
| **Souriant** | 0,95 | 0,9474 |
| **Neutre** | 0,9756 | 0,9268 |
| **Triste** | 0,95 | 0,9268 |
| **En colère** | 0,95 | 0,95 |
| **Surpris** | 0,9744 | 1,00 |
| **Macro** | **0,96** | **0,9502** |

---

## Pourquoi les métriques sont artificiellement élevées

Ces résultats (94–96 %) ne reflètent pas une capacité de généralisation réelle — ils sont le produit direct des caractéristiques du jeu de données synthétique.

**1. Données entièrement synthétiques à signature algorithmique.**
Chaque image est un visage cartoon 16×16 dessiné par code (cercles, arcs, lignes droites). Chaque émotion a une règle géométrique fixe et non ambiguë : sourire = arc vers le haut, tristesse = arc vers le bas + larmes bleues, colère = sourcils inclinés vers l'intérieur + ligne droite, surprise = grands yeux + sourcils relevés + bouche ronde. Il n'existe aucune ambiguïté inter-classes.

**2. Bruit très faible.**
Le bruit gaussien appliqué a un écart-type σ = 3–6 sur une échelle 0–255, soit moins de 2 % de la plage dynamique. Les caractéristiques émotionnelles dominent largement le signal.

**3. Visages toujours centrés, droits, sans variation réelle.**
La position est décalée de ±0,7 px au maximum et l'échelle varie de ±10 % seulement. Il n'y a ni rotation, ni occlusion, ni changement d'éclairage, ni diversité morphologique.

**4. Distribution de test identique à la distribution d'entraînement.**
Les images de test sont générées par le même générateur avec des graines distinctes mais la même distribution étroite. Le modèle n'est jamais confronté à un cas hors-distribution.

**En pratique**, un modèle entraîné sur ce jeu de données obtiendrait probablement ~20 % (chance) sur de vraies photos de visages. Les métriques mesurent ici la capacité du réseau à apprendre la géométrie du générateur, pas la reconnaissance d'émotions réelles.

---

## Interprétation

### Pourquoi le split obtient un meilleur score CV (0,96 vs 0,9225)
Le résultat du split est une **évaluation unique sur 100 échantillons**. Avec seulement 20 échantillons par classe, une prédiction correcte ou incorrecte fait varier la précision globale de 1 %. L'estimation par split est donc bruitée — une graine aléatoire différente pourrait facilement inverser le classement. La moyenne K-Fold de **0,9225 ± 0,0436** est une estimation plus fiable de la capacité de généralisation réelle.

Le modèle final K-Fold est entraîné sur 400 échantillons (au lieu de 300 pour le split), ce qui explique que sa précision sur l'ensemble de test remonte à **0,95** — identique au split, malgré une variance CV plus élevée.

### Ce qu'apporte le K-Fold
- **Intervalle de confiance** : l'écart-type de ± 0,044 montre que le modèle est sensible aux sous-ensembles de données (fold 2 à 0,85 contre fold 3 à 0,975).
- **Utilisation de toutes les données** : chaque échantillon est utilisé à la fois pour l'entraînement et la validation au fil des 5 folds, offrant une vision plus complète qu'un découpage fixe.
- **Moins dépendant de la chance** : le résultat du split dépend fortement des 100 échantillons qui se retrouvent dans l'ensemble de test.

### Observations par classe
- **Souriant** : précision parfaite (1,00) en K-Fold — aucune autre classe n'est confondue avec « souriant ». Cependant le rappel descend à 0,90 : 2 images souriantes sur 20 sont mal classées (1 → neutre, 1 → triste).
- **Neutre** : précision de 0,9048 — quelques images d'autres classes sont prédites comme neutres (1 souriant + 1 triste). Rappel de 0,95 : 1 neutre prédit comme en colère.
- **Triste** : mêmes valeurs que neutre (P=0,9048, R=0,95) — la confusion est symétrique.
- **En colère** : précision et rappel équilibrés à 0,95. 1 triste prédit comme en colère, 1 en colère prédit comme triste.
- **Surpris** : classe parfaitement séparée — précision et rappel à 1,00 dans les deux approches.

### Quelle estimation privilégier
Utilisez la **précision CV K-Fold (0,9225 ± 0,044)** comme chiffre de performance principal — elle est statistiquement plus robuste. Utilisez le **résultat sur l'ensemble de test (0,95)** comme évaluation finale non biaisée, puisque cet ensemble n'a jamais été vu lors de l'entraînement d'un fold.

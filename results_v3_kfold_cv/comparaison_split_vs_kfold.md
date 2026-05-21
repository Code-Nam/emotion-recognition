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

## Métriques globales

| Métrique | Split | K-Fold (moyenne CV) | K-Fold (test) |
|---|---|---|---|
| **Précision globale** | 0,96 | 0,943 ± 0,023 | 0,94 |
| **Précision macro** | 0,9605 | — | 0,9489 |
| **Rappel macro** | 0,96 | — | 0,94 |
| **F1 macro** | 0,96 | — | 0,9413 |

### Précision par fold (K-Fold)

| Fold | Précision globale |
|---|---|
| Fold 1 | 0,925 |
| Fold 2 | 0,9125 |
| Fold 3 | 0,975 |
| Fold 4 | 0,9375 |
| Fold 5 | 0,9625 |
| **Moyenne** | **0,943 ± 0,023** |

---

## Métriques par classe (ensemble de test)

| Classe | Précision Split | Précision K-Fold | Rappel Split | Rappel K-Fold | F1 Split | F1 K-Fold |
|---|---|---|---|---|---|---|
| **Souriant** | 0,95 | 0,9444 | 0,95 | 0,85 | 0,95 | 0,8947 |
| **Neutre** | 0,9524 | 0,80 | 1,00 | 1,00 | 0,9756 | 0,8889 |
| **Triste** | 0,95 | 1,00 | 0,95 | 0,95 | 0,95 | 0,9744 |
| **En colère** | 0,95 | 1,00 | 0,95 | 0,95 | 0,95 | 0,9744 |
| **Surpris** | 1,00 | 1,00 | 0,95 | 0,95 | 0,9744 | 0,9744 |

---

## Interprétation

### Pourquoi le split obtient un meilleur score sur cette exécution (0,96 vs 0,94)
Le résultat du split est une **évaluation unique sur 100 échantillons**. Avec seulement 20 échantillons par classe, une prédiction correcte ou incorrecte fait varier la précision globale de 1 %. L'estimation par split est donc bruitée — une graine aléatoire différente pourrait facilement inverser le classement. La moyenne K-Fold de **0,943 ± 0,023** est une estimation plus fiable de la capacité de généralisation réelle.

### Ce qu'apporte le K-Fold
- **Intervalle de confiance** : l'écart-type de ± 0,023 montre que le modèle est régulièrement performant, mais pas parfaitement stable selon les sous-ensembles de données.
- **Utilisation de toutes les données** : chaque échantillon est utilisé à la fois pour l'entraînement et la validation au fil des 5 folds, offrant une vision plus complète qu'un découpage fixe.
- **Moins dépendant de la chance** : le résultat du split dépend fortement des 100 échantillons qui se retrouvent dans l'ensemble de test.

### Observations par classe
- **Neutre** : le modèle split avait une précision quasi parfaite (0,95) alors que le K-Fold descend à 0,80 — le modèle final K-Fold surprédit légèrement la classe neutre. Les deux atteignent un rappel parfait (1,00).
- **Souriant** : l'écart le plus important — le rappel passe de 0,95 (split) à 0,85 (K-Fold). La classe souriant est celle qui est le plus souvent confondue avec surpris ou neutre.
- **Triste, En colère, Surpris** : résultats essentiellement équivalents entre les deux approches.

### Quelle estimation privilégier
Utilisez la **précision CV K-Fold (0,943 ± 0,023)** comme chiffre de performance principal — elle est statistiquement plus robuste. Utilisez le **résultat sur l'ensemble de test (0,94)** comme évaluation finale non biaisée, puisque cet ensemble n'a jamais été vu lors de l'entraînement d'un fold.

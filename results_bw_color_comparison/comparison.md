# Comparaison des modèles : Niveaux de gris vs Couleur

## Description

Ce document compare les performances de deux modèles de reconnaissance d'émotions :

- **Niveaux de gris** : images PGM 16×16 (256 caractéristiques)
- **Couleur** : images PPM sous-échantillonnées à 8×8 (192 caractéristiques, 3 canaux RVB)

Les deux modèles utilisent un réseau de neurones à une couche cachée (64 neurones), entraîné avec une validation croisée à 5 plis sur 20 époques.

---

## Métriques globales

| Métrique             | Niveaux de gris | Couleur | Δ (Couleur − Gris) |
|----------------------|----------------:|--------:|--------------------:|
| Précision (test)     | 0.9400          | 0.8700  | −0.0700             |
| F1 macro             | 0.9413          | 0.8704  | −0.0709             |
| Précision macro      | 0.9489          | 0.8845  | −0.0644             |
| Rappel macro         | 0.9400          | 0.8700  | −0.0700             |
| Acc. CV moyenne      | 0.9425          | 0.9025  | −0.0400             |
| Écart-type CV        | 0.0232          | 0.0166  | −0.0066             |

---

## F1 par classe

| Classe      | Niveaux de gris | Couleur | Δ         |
|-------------|----------------:|--------:|----------:|
| Souriant    | 0.8947          | 0.8571  | −0.0376   |
| Neutre      | 0.8889          | 0.7368  | −0.1521   |
| Triste      | 0.9744          | 0.9756  | +0.0012   |
| En colère   | 0.9744          | 0.7826  | −0.1918   |
| Surpris     | 0.9744          | 1.0000  | +0.0256   |

---

## Précision par pli (validation croisée)

| Pli       | Niveaux de gris | Couleur |
|-----------|----------------:|--------:|
| Pli 1     | 0.9250          | 0.9125  |
| Pli 2     | 0.9125          | 0.8750  |
| Pli 3     | 0.9750          | 0.9000  |
| Pli 4     | 0.9375          | 0.9250  |
| Pli 5     | 0.9625          | 0.9000  |
| **Moyenne**   | **0.9425**  | **0.9025** |
| **Écart-type** | **0.0232** | **0.0166** |

---

## Analyse

Le modèle en niveaux de gris surpasse le modèle couleur sur l'ensemble des métriques, avec une précision de test de **94 %** contre **87 %**.

Les classes **neutre** et **en colère** sont les plus affectées par le passage à la couleur (F1 respectivement −0.15 et −0.19). Le sous-échantillonnage 8×8 efface les détails fins des sourcils et de la bouche, qui sont les indices structurels clés pour distinguer ces deux émotions.

À l'inverse, les classes **triste** et **surpris** maintiennent ou améliorent légèrement leurs scores grâce aux indices chromatiques (larmes bleues, intérieur de la bouche foncé), qui restent perceptibles même après réduction de résolution.

La variabilité entre les plis est plus faible pour le modèle couleur (0.0166 vs 0.0232), ce qui indique une certaine stabilité, mais au prix d'une précision globale inférieure. Cela s'explique en partie par la taille réduite du vecteur d'entrée (192 vs 256 caractéristiques) qui simplifie l'espace d'apprentissage.

En résumé, pour ce jeu de données synthétique, la structure géométrique des visages encodée en niveaux de gris s'avère plus discriminante que l'information chromatique sous-échantillonnée.

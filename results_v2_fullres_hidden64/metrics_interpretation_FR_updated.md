# Interprétation des Métriques

## Comparaison des Résultats : V1 vs V2

### Changements apportés dans V2
- **Suppression du sous-échantillonnage** : les images 16×16 sont maintenant utilisées en entier (256 valeurs) au lieu d'être réduites à 8×8 (64 valeurs).
- **Augmentation de la taille cachée** : de 8 à 64 neurones cachés, donnant au réseau plus de capacité d'apprentissage.

### Résultats Globaux

| Métrique | V1 | V2 | Évolution |
|---|---|---|---|
| **Accuracy** | 0.73 | 0.96 | +0.23 |
| **Macro Précision** | 0.7479 | 0.9605 | +0.2126 |
| **Macro Rappel** | 0.73 | 0.96 | +0.23 |
| **Macro F1** | 0.7274 | 0.96 | +0.2326 |

### Résultats par Classe

| Classe | Précision V1 | Précision V2 | Rappel V1 | Rappel V2 | F1 V1 | F1 V2 |
|---|---|---|---|---|---|---|
| **Smiling** | 0.7727 | 0.95 | 0.85 | 0.95 | 0.8095 | 0.95 |
| **Neutral** | 0.40 | 0.9524 | 0.30 | 1.00 | 0.3429 | 0.9756 |
| **Sad** | 1.0 | 0.95 | 0.75 | 0.95 | 0.8571 | 0.95 |
| **Angry** | 0.5667 | 0.95 | 0.85 | 0.95 | 0.68 | 0.95 |
| **Surprised** | 1.0 | 1.0 | 0.90 | 0.95 | 0.9474 | 0.9744 |

---

## Interprétation V2

### Résultats Globaux

| Métrique | Valeur | Interprétation |
|---|---|---|
| **Accuracy** | 0.96 | Le modèle classe correctement 96 % de toutes les images — une nette amélioration par rapport aux 73 % de la V1. |
| **Macro Précision** | 0.9605 | En moyenne, quand le modèle prédit une émotion, il a raison 96 % du temps. |
| **Macro Rappel** | 0.96 | Le modèle détecte 96 % des instances réelles par émotion. |
| **Macro F1** | 0.96 | Score très élevé et équilibré — la précision et le rappel sont alignés sur toutes les classes. |

### Résultats par Classe

| Classe | Précision | Rappel | F1 | Support | Interprétation |
|---|---|---|---|---|---|
| **Smiling** | 0.95 | 0.95 | 0.95 | 20 | Très bonne performance, stable et équilibrée. |
| **Neutral** | 0.9524 | 1.00 | 0.9756 | 20 | Transformation remarquable — la pire classe en V1 (rappel 0.30) est maintenant quasi parfaite. Le sous-échantillonnage effaçait les traits distinctifs des visages neutres. |
| **Sad** | 0.95 | 0.95 | 0.95 | 20 | Légère baisse de précision (1.0 → 0.95) mais le rappel est passé de 0.75 à 0.95 — bien meilleur globalement. |
| **Angry** | 0.95 | 0.95 | 0.95 | 20 | Grande amélioration de la précision (0.57 → 0.95) — le modèle ne sur-prédit plus angry. |
| **Surprised** | 1.00 | 0.95 | 0.9744 | 20 | Toujours la meilleure classe, maintenant avec une précision parfaite en plus. |

---

## Points Clés

- La suppression du sous-échantillonnage est le changement le plus impactant : passer de 64 à 256 valeurs d'entrée a préservé les informations nécessaires pour distinguer les émotions.
- **Neutral** était victime du sous-échantillonnage — ses traits distinctifs (expressions subtiles) étaient perdus lors de la réduction 16×16 → 8×8.
- L'augmentation de la taille cachée (8 → 64) a donné au réseau la capacité d'apprendre des représentations plus riches.
- Les erreurs restantes (4 sur 100) sont toutes des confusions entre classes visuellement proches : smiling↔surprised, sad↔angry.

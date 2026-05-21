# Interprétation des Métriques

## Définitions

### Accuracy (Précision globale)
Le pourcentage de toutes les prédictions qui étaient correctes. C'est le ratio simple entre les bonnes réponses et le total des échantillons. Peut être trompeur sur des jeux de données déséquilibrés, mais fiable ici puisque toutes les classes ont le même support (20 chacune).

### Précision (Precision)
Parmi toutes les fois où le modèle a prédit une classe donnée, combien de fois avait-il raison ?
> "Quand tu dis que c'est *angry*, à quelle fréquence as-tu raison ?"

Une précision élevée = peu de faux positifs.

### Rappel (Recall)
Parmi toutes les instances réelles d'une classe, combien le modèle en a-t-il détectées ?
> "Parmi tous les visages vraiment *angry*, combien en as-tu trouvé ?"

Un rappel élevé = peu de faux négatifs.

### Score F1
La moyenne harmonique de la précision et du rappel. Utile quand on se soucie des deux — un modèle avec une précision de 1.0 mais un rappel de 0.0 obtiendrait un F1 de 0, pas 0.5. Il pénalise le déséquilibre entre les deux.

> F1 = 2 × (précision × rappel) / (précision + rappel)

### Macro vs Par Classe
- Les métriques **par classe** calculent la précision, le rappel et le F1 indépendamment pour chaque émotion.
- La moyenne **macro** fait la moyenne de ces scores par classe de façon égale, indépendamment du nombre d'échantillons par classe. Comme toutes les classes ont le même support ici (20), la moyenne macro est simplement la moyenne des cinq scores.

### Support
Le nombre d'échantillons réels pour chaque classe dans le jeu de test. Utilisé comme vérification — si le support est très faible pour une classe, ses métriques sont moins fiables.

---

## Résultats Globaux

| Métrique | Valeur | Interprétation |
|---|---|---|
| **Accuracy** | 0.73 | Le modèle classe correctement 73 % de toutes les images toutes émotions confondues. Correct, mais avec une marge d'amélioration. |
| **Macro Précision** | 0.7479 | En moyenne, quand le modèle prédit une émotion, il a raison ~75 % du temps. Traite toutes les classes de façon égale. |
| **Macro Rappel** | 0.73 | En moyenne, le modèle détecte 73 % des instances réelles par émotion. Correspond à l'accuracy car les classes sont équilibrées (20 chacune). |
| **Macro F1** | 0.7274 | Moyenne harmonique de la précision et du rappel. Un F1 de 0.73 est correct globalement, mais le détail par classe révèle une réalité plus nuancée. |

---

## Résultats par Classe

| Classe | Précision | Rappel | F1 | Support | Interprétation |
|---|---|---|---|---|---|
| **Smiling** | 0.7727 | 0.85 | 0.8095 | 20 | Bonne performance. Un rappel élevé signifie qu'il rate rarement un visage souriant, et la précision est respectable. |
| **Neutral** | 0.40 | 0.30 | 0.3429 | 20 | La classe la plus faible. Le modèle rate 70 % des visages neutres et confond souvent cette émotion avec d'autres, notamment sad et angry. |
| **Sad** | 1.0 | 0.75 | 0.8571 | 20 | Précision parfaite — chaque prédiction sad est correcte — mais 25 % des visages tristes sont manqués, probablement classés comme neutral ou angry. |
| **Angry** | 0.5667 | 0.85 | 0.68 | 20 | Rappel élevé : le modèle détecte la plupart des visages angry, mais la faible précision indique qu'il sur-prédit cette classe en y classant d'autres émotions. |
| **Surprised** | 1.0 | 0.90 | 0.9474 | 20 | Meilleure classe. Quasi parfait — le modèle a clairement appris des caractéristiques visuelles distinctives pour l'expression de surprise. |

---

## Points Clés

- **Surprised** et **Sad** ont une précision parfaite, ce qui suggère que ces expressions ont des caractéristiques visuelles distinctes bien apprises par le modèle.
- **Neutral** est le problème principal — faible précision et faible rappel indiquent que le modèle n'a pas de signal fort pour cette émotion, et les images neutres se retrouvent classées dans d'autres catégories.
- **Angry** tend à être sur-prédit (faible précision, rappel élevé), ce qui est probablement causé en partie par des images neutres mal classées comme angry.
- Améliorer **Neutral** est l'optimisation la plus rentable — c'est probablement cette classe qui tire le plus vers le bas l'accuracy et le macro F1.

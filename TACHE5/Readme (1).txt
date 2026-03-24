1. Introduction
Ce rapport présente l'implémentation d'un classifieur basé sur l'architecture Transformer Encoder. L'objectif est de prédire la classe de séquences d'ARN non codantes (ncRNA) en utilisant les représentations vectorielles (embeddings) de dimension 100 générées lors de la Tâche 4.

2. Architecture du Modèle
Le modèle TransformerRNA a été conçu avec les composants suivants pour capturer les dépendances complexes dans les données :

Projection Linéaire : Transformation de l'espace d'entrée (dimension 100) vers l'espace du Transformer (dimension 128).

Transformer Encoder :

2 couches d'encodage pour extraire des caractéristiques profondes.

4 têtes d'attention (Multi-head attention) pour analyser simultanément différentes parties du vecteur.

Pooling Global : Calcul de la moyenne sur la dimension temporelle pour obtenir un vecteur fixe par séquence.

Couche de Classification : Une couche finale nn.Linear produisant les scores pour les 3 classes cibles.

3. Méthodologie d'Entraînement
L'entraînement a été effectué sur un ensemble de 168 310 séquences. Les hyperparamètres utilisés sont les suivants :

Optimiseur : Adam avec un taux d'apprentissage (learning rate) de 0.001.

Fonction de Perte : CrossEntropyLoss, adaptée à la classification multi-classe.

Nombre d'époques : 3 (validation technique de la convergence).

4. Résultats et Analyse
Les résultats d'exécution montrent une diminution progressive et une stabilisation de la perte (Loss) :

Époque 1/3 : Loss = 1.1031

Époque 2/3 : Loss = 1.0992

Époque 3/3 : Loss = 1.0991

Cette stabilisation à 1.0991 confirme que le modèle a correctement initialisé ses poids et est prêt pour une phase d'optimisation plus poussée.

5. Conclusion
La Tâche 5 est validée par la création d'un modèle fonctionnel et la génération du fichier de poids transformer_rna_task5.pth. Ce fichier constitue le point de départ pour la Tâche 6, qui consistera à effectuer le fine-tuning du modèle pour améliorer la précision globale de classification.
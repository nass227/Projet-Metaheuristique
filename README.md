# Maximum Covering Problem (MCP) - Algorithme Génétique

Ce projet implémente un **algorithme génétique** pour résoudre le **Maximum Covering Problem (MCP)** dans le cadre d'un **projet de TP de métaheuristiques**.  
Le MCP est un problème d'optimisation combinatoire où l'objectif est de sélectionner **au plus k sous-ensembles** afin de couvrir **le maximum d'éléments**.

## 1. Présentation du problème  

Le **Maximum Covering Problem (MCP)** est défini comme suit :  
- Soit un ensemble universel `U` contenant `n` éléments.  
- Un ensemble `S` de `m` sous-ensembles de `U`, chacun ayant un coût.  
- Un nombre maximal `k` de sous-ensembles que l'on peut sélectionner.  
- L'objectif est de choisir **au plus k sous-ensembles** de `S` de manière à **couvrir le plus grand nombre d'éléments possibles de `U`**.  

## 2. Algorithme génétique utilisé  

L'algorithme génétique suit les étapes suivantes :  

1. **Initialisation** : Génération aléatoire d'une population de solutions représentées sous forme de vecteurs binaires.  
2. **Évaluation** : Calcul du score de chaque individu (nombre d'éléments couverts).  
3. **Sélection** : Choix des meilleurs individus pour la reproduction (roulette, tournoi, etc.).  
4. **Croisement (crossover)** : Combinaison de deux parents pour créer de nouveaux individus.  
5. **Mutation** : Modification aléatoire d'un gène pour maintenir la diversité.  
6. **Remplacement** : Intégration des nouveaux individus dans la population.  
7. **Critère d'arrêt** : Nombre maximal d'itérations ou convergence des solutions.  

## 2. Membres du projet

- ABDELLAZIZ Nassila
- AKMOUNE Feriel
- BENBOUCHAMA Fatima zahra

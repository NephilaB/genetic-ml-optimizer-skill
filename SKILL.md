---
name: genetic-ml-optimizer
description: "Optimise les hyperparamètres, architectures de réseaux neuronaux et fonctions complexes avec des algorithmes génétiques avancés (PyGAD/DEAP style), opérateurs adaptatifs, élitisme, multi-objectifs et hybrides gradient. Utiliser pour l'optimisation d'algorithmes d'apprentissage, de modèles ML, de NAS ou de tâches évolutives."
type: workflow
lifecycle: active
---

# Genetic ML Optimizer — Optimiseur génétique le plus puissant pour Claude

Skill d'optimisation génétique de pointe pour algorithmes d'apprentissage automatique, hyperparamètres, architectures neurales (NAS) et fonctions non-différentiables. Combine les meilleures pratiques de PyGAD, DEAP, CMA-ES et hybrides GA-gradient.

## Quand utiliser

- Optimisation d'hyperparamètres (XGBoost, scikit-learn, réseaux neuronaux)
- Recherche d'architecture neuronale (NAS)
- Optimisation multi-objectifs (précision vs taille modèle)
- Fonctions black-box, non-convexes ou discrètes
- Quand Bayesian Optimization ou grid search échouent ou sont trop lents

## Workflow principal

1. **Définir le problème**
   - Identifier les gènes (hyperparamètres, poids, architecture)
   - Choisir type de gène : float, int, categorical, binary
   - Définir fitness (maximiser accuracy, minimiser loss, multi-objectif)

2. **Choisir le moteur**
   - **PyGAD** : simple, rapide à prototyper, excellent pour ML (Keras/PyTorch)
   - **DEAP** : contrôle total, multi-objectif (NSGA-II), genetic programming
   - **Hybride** : GA pour structure + Adam/SGD pour poids

3. **Configuration recommandée (puissante par défaut)**

| Paramètre              | Valeur recommandée          | Raison |
|------------------------|-----------------------------|--------|
| Population             | 50–200                      | Bon équilibre exploration/exploitation |
| Générations            | 50–300                      | Selon coût fitness |
| Sélection              | Tournament (k=3) ou SSS     | Robuste |
| Crossover              | Uniform ou Two-point        | Diversité |
| Mutation               | Adaptive (0.1 → 0.01)       | Exploration puis exploitation |
| Élitisime              | 1–5%                        | Préserver meilleurs |
| Parallélisme           | Oui (n_jobs)                | Accélération |

4. **Implémenter ou générer le code**
   - Utiliser le template PyGAD ci-dessous
   - Ajouter early stopping, callbacks, logging
   - Pour NAS : encoder architecture en gènes

5. **Évaluer et itérer**
   - Tracker meilleur fitness, diversité
   - Ajuster mutation/crossover si stagnation
   - Comparer vs Random/Bayesian

## Template PyGAD puissant (copy-paste ready)

```python
import pygad
import numpy as np

def fitness_func(ga_instance, solution, solution_idx):
    # Remplacer par votre évaluation (accuracy, -loss, etc.)
    # Exemple: x, y = solution
    # return -(x-3)**2 - (y+1)**2 + 10
    return np.sum(solution)  # placeholder

ga_instance = pygad.GA(
    num_generations=150,
    num_parents_mating=20,
    fitness_func=fitness_func,
    sol_per_pop=80,
    num_genes=10,
    init_range_low=-5,
    init_range_high=5,
    parent_selection_type="tournament",
    K_tournament=3,
    crossover_type="uniform",
    mutation_type="adaptive",
    mutation_probability=[0.25, 0.05],  # haut → bas
    mutation_num_genes=[3, 1],
    keep_elitism=3,
    gene_space=None,  # ou dict par gène
    stop_criteria=["saturate_15", "reach_0.99"],
    parallel_processing=4,  # si disponible
    random_seed=42,
    on_generation=lambda ga: print(f"Gen {ga.generations_completed} | Best: {ga.best_solution()[1]:.4f}")
)

ga_instance.run()
solution, fitness, _ = ga_instance.best_solution()
print(f"Meilleure solution: {solution}")
print(f"Fitness: {fitness}")
```

## Opérateurs avancés

- **Mutation adaptive** : diminue avec les générations
- **Crossover arithmétique** : pour variables continues
- **NSGA-II** (via DEAP) : multi-objectifs
- **Hybrid local search** : après GA, affiner avec BFGS ou Adam sur les meilleurs
- **Diversity preservation** : penalty si trop de similarité

## Cas d'usage spécifiques

### Hyperparamètres XGBoost / LightGBM
Gènes : learning_rate, max_depth, n_estimators, subsample, colsample_bytree, reg_alpha, reg_lambda
Fitness : cross-val score (accuracy ou AUC)

### Neural Architecture Search (NAS)
Gènes : nombre couches, unités, activation, dropout, learning rate
Encoder : liste d'entiers + contraintes (pas de cycles)

### Optimisation de poids (alternative à gradient)
Utile pour fonctions non-différentiables. Population de matrices de poids.

## Bonnes pratiques

- Toujours utiliser `random_seed` pour reproductibilité
- Logger les stats (best, avg, std) chaque génération
- Normaliser les gènes si échelles très différentes
- Utiliser `gene_space` pour contraindre les domaines
- Pour problèmes coûteux : multi-fidelity (évaluer d'abord sur petit dataset)
- Comparer systématiquement avec Optuna (Bayesian) ou Hyperopt

## Dépannage

| Problème              | Solution |
|-----------------------|----------|
| Stagnation précoce    | Augmenter mutation, réduire élitisme |
| Convergence lente     | Augmenter population ou générations |
| Solutions invalides   | Ajouter contraintes dans fitness ou gene_constraint |
| Trop lent             | Paralléliser fitness, réduire population, early stop |

## Références rapides

- PyGAD docs : pygad.readthedocs.io
- DEAP : deap.readthedocs.io
- Pour multi-objectif : pymoo ou DEAP NSGA-II

Toujours générer du code exécutable, expliquer les choix, et proposer des variantes (plus rapide vs plus précis).

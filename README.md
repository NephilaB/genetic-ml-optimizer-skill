# Genetic ML Optimizer Skill for Claude

**L'optimiseur génétique le plus puissant pour Claude** — optimise hyperparamètres, architectures de réseaux neuronaux (NAS) et fonctions complexes avec des algorithmes génétiques avancés.

## Installation

1. Clone ce repo :
   ```bash
   git clone https://github.com/NephilaB/genetic-ml-optimizer-skill.git
   ```
2. Place le dossier dans tes skills Claude :
   - Claude Code : `~/.claude/skills/genetic-ml-optimizer`
   - Ou uploade le contenu de `SKILL.md` dans Claude.ai Skills

## Ce que ça fait

- Optimisation d'hyperparamètres (XGBoost, scikit-learn, NN)
- Neural Architecture Search (NAS)
- Multi-objectifs (précision vs taille)
- Templates PyGAD prêts à l'emploi avec mutation adaptive, élitisme, parallélisme
- Hybrides GA + gradient (Adam/BFGS)

## Usage

Demande simplement à Claude :
- "Optimise les hyperparamètres de mon XGBoost avec un algorithme génétique"
- "Trouve la meilleure architecture de réseau pour MNIST avec un GA"
- "Optimise cette fonction black-box avec PyGAD"

Claude chargera automatiquement le skill.

## Structure

```
SKILL.md              ← instructions principales
CHANGELOG.md
evals/
  evals.json
  triggers.json
```

Créé le 2026-11-05 — skill optimiseur génétique de pointe pour Claude.

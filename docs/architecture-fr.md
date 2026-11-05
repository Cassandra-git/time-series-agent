# Architecture du projet

## 1. Présentation

Le projet repose sur une architecture modulaire séparant l'agent conversationnel, les outils accessibles au LLM, les services de traitement des données et l'interface utilisateur.

**Flux de fonctionnement :**

```text
Utilisateur
  |
  v
Interface Streamlit
  |
  v
Agent LangChain / LangGraph
  |
  +---- Ollama / Qwen (compréhension du langage)
  |
  v
Tools LangChain
  |
  v
Services Python
  |
  v
Résultats d'analyse
  |
  v
Réponse de l'agent + Visualisations
```

Le LLM interprète les demandes et choisit les outils. Les services Python réalisent les traitements et les calculs numériques.

## 2. Description des dossiers

### `.github/workflows/`

Contient les automatisations GitHub Actions.

- `tests.yml` : exécute les vérifications automatiques lors des modifications du dépôt ou des Pull Requests.

Ce dossier est indépendant des workflows LangGraph.

### `.venv/`

Environnement virtuel Python local contenant l'interpréteur et les dépendances installées.

Il doit être exclu de Git.

### `src/ts_agent/`

Package Python principal contenant la logique applicative et l'agent IA.

- `__init__.py` : initialise le package Python.
- `config.py` : centralise les paramètres, notamment l'adresse Ollama et le modèle utilisé.

### `src/ts_agent/agent/`

Contient la construction et l'orchestration de l'agent.

- `__init__.py` : initialise le package.
- `builder.py` : initialise le LLM, enregistre les tools et construit l'agent.
- `graph.py` : définit les workflows LangGraph personnalisés et la gestion des états si nécessaire.
- `prompts.py` : contient les prompts système et les instructions de l'agent.

### `src/ts_agent/tools/`

Contient les interfaces accessibles au LLM.

- `__init__.py` : initialise le package.
- `cleaning_tools.py` : outils de chargement et d'inspection des données.
- `analysis_tools.py` : outils statistiques, de détection d'anomalies et de prévision.
- `visualisation_tools.py` : outils de génération de visualisations.

Les tools valident les paramètres, appellent les services appropriés et renvoient les résultats à l'agent.

### `src/ts_agent/services/`

Contient les fonctions Python indépendantes du framework d'agent.

- `__init__.py` : initialise le package.
- `data_loader.py` : charge et valide les fichiers CSV.
- `preprocessing.py` : nettoie les données, traite les valeurs manquantes et prépare les dates.
- `statistics.py` : calcule les statistiques descriptives.
- `anomalies.py` : implémente les algorithmes de détection d'anomalies.
- `forecasting.py` : implémente les méthodes de prévision.
- `plotting.py` : génère les graphiques avec Plotly.

### `app/`

Contient l'interface utilisateur.

- `streamlit_app.py` : gère l'importation des fichiers, les questions, l'affichage de la conversation et les visualisations.

### `data/`

Contient les jeux de données de développement et de démonstration.

- `sample/` : contient les fichiers csv de données

Les données privées, volumineuses ou générées ne doivent pas être publiées.

### `tests/`

Contient les tests automatisés.

- `test_services.py` : vérifie les calculs et traitements des données.
- `test_tools.py` : valide les paramètres et résultats des tools.
- `test_agent.py` : teste l'intégration de l'agent et la sélection des outils.

### `docs/`

Contient la documentation du projet.

- `architecture.md` : documentation de l'architecture en anglais.
- `architecture.fr.md` : documentation de l'architecture en français.

### `experiments/`

Contient les expérimentations et comparaisons.

- `model_comparison.ipynb` : compare différents modèles, prompts ou méthodes d'analyse.

Le code expérimental ne doit pas être nécessaire au fonctionnement de l'application principale.

## 3. Fichiers à la racine

| Fichier | Utilité |
|---|---|
| `README.md` | Documentation principale en anglais |
| `pyproject.toml` | Dépendances, métadonnées et configuration Python |
| `.env.example` | Exemple des variables d'environnement |
| `.gitignore` | Exclut les fichiers locaux du dépôt Git |
| `Dockerfile` | Définit la construction du conteneur Python |
| `.dockerignore` | Exclut certains fichiers de la construction Docker |

## 4. Principes de conception

**Séparation des responsabilités :** l'agent orchestre, les tools exposent les fonctionnalités et les services réalisent les calculs.

**Modularité :** chaque composant peut être développé et testé indépendamment.

**Réutilisabilité :** les services Python fonctionnent sans LangChain ni Ollama.

**Configurabilité :** l'adresse Ollama et le modèle utilisé sont définis dans la configuration.

**Collaboration :** les branches Git, Pull Requests et tests facilitent le travail parallèle.

**Développement progressif :** commencer par le chargement CSV et les statistiques avant les workflows avancés, les prévisions et Docker.

## 5. Remarque sur LangGraph

`builder.py` peut utiliser directement l'implémentation des agents LangChain, qui repose déjà sur LangGraph. Le fichier `graph.py` devient utile lorsqu'un workflow personnalisé nécessite des états, des transitions ou une orchestration particulière.

Les modules décrivent l'architecture cible et pourront être développés progressivement.

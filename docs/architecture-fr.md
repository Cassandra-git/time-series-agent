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

### `src/`

Code Python principal. `pyproject.toml` installe ses trois packages (`agent`, `tools`, `services`) : on les importe directement (`from services.statistics import ...`), sans préfixe `src.`.

### `src/agent/`

Contient la construction et l'orchestration de l'agent.

- `config.py` : centralise les paramètres lus dans l'environnement (`OLLAMA_HOST`, `OLLAMA_MODEL`, dossiers de sortie, nombre maximal d'appels d'outils).
- `builder.py` : initialise le LLM (`ChatOllama`), enregistre les tools, construit l'agent ReAct LangGraph (`create_react_agent`) avec un état et une mémoire de conversation, et fournit `AgentSession`, utilisée par la CLI et plus tard par Streamlit.
- `graph.py` : workflow LangGraph de nettoyage guidé (audit, questions métier, validation humaine, export Parquet). Il est lancé par le tool `clean_dataset` et peut aussi s'exécuter seul dans le terminal.
- `prompts.py` : contient le prompt système et le contexte du dataset actif.

### `src/tools/`

Contient les interfaces accessibles au LLM.

- `agent_tools.py` : catalogue des tools donnés à l'agent. Ils lisent le dataset actif dans l'état de l'agent, donc le LLM n'a jamais à écrire de chemin de fichier.
- `cleaning_tools.py` : outils d'inspection et de nettoyage des données.
- `analysis_tools.py` : outils statistiques.
- `visualisation_tools.py` : outils de génération de visualisations.

Les tools valident les paramètres, appellent les services appropriés et renvoient les résultats à l'agent.

### `src/services/`

Contient les fonctions Python indépendantes du framework d'agent.

- `data_loader.py` : charge, valide et convertit en Parquet les fichiers CSV, Excel et Parquet.
- `preprocessing.py` : nettoie les données, traite les valeurs manquantes et prépare les dates.
- `statistics.py` : calcule les statistiques descriptives.
- `anomalies.py` : implémente les algorithmes de détection d'anomalies (à venir).
- `forecasting.py` : implémente les méthodes de prévision (à venir).
- `plotting.py` : génère les graphiques avec Plotly.

### `app/`

Contient l'interface utilisateur.

- `streamlit_app.py` : gère l'importation des fichiers, les questions, l'affichage de la conversation et les visualisations.

### `datasets/`

Contient les jeux de données de développement et de démonstration.

- `sample/` : contient les fichiers CSV de données.
- `processed/` : fichiers Parquet convertis ou nettoyés par l'agent (non versionnés).

Les données privées, volumineuses ou générées ne doivent pas être publiées.

### `tests/`

Contient les tests automatisés.

- `test_services.py` : vérifie les calculs et traitements des données.
- `test_tools.py` : valide les paramètres et résultats des tools.
- `test_agent.py` : teste l'intégration de l'agent et la sélection des outils.

### `docs/`

Contient la documentation du projet.

- `architecture-en.md` : documentation de l'architecture en anglais.
- `architecture-fr.md` : documentation de l'architecture en français.

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

## 5. Fonctionnement de l'agent

`builder.py` utilise l'agent ReAct de LangGraph : le LLM choisit un tool, LangGraph l'exécute, puis renvoie le résultat au LLM, jusqu'à ce que celui-ci réponde sans appeler de tool.

- **État :** en plus des messages, l'état contient le fichier source et le Parquet actif (`dataset_path`). `load_dataset` et `clean_dataset` mettent à jour cet état ; les autres tools le lisent.
- **Mémoire :** un checkpointer conserve l'état par `thread_id`. Une `AgentSession` correspond à une conversation multi-tours.
- **Validation humaine :** `clean_dataset` lance le workflow de `graph.py`. Chaque question de ce workflow passe par `interrupt()` : l'agent se met en pause, l'interface affiche la question, puis `AgentSession.resume(réponse)` le relance.
- **Erreurs :** une erreur de paramètre dans un tool (colonne inconnue, date invalide, etc.) est renvoyée au LLM, qui peut corriger son appel.
- **Tests :** `tests/test_agent.py` remplace Qwen par un modèle scripté, ce qui permet de tester l'assemblage sans Ollama.

Les modules d'anomalies et de prévision seront ajoutés progressivement.

#!/usr/bin/env bash
# Start Docker, prepare Qwen and open the persistent chat from one command.
set -euo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"

if ! command -v docker > /dev/null 2>&1; then
    printf 'Docker est introuvable dans ce terminal. Installe Docker avec Compose.\n' >&2
    exit 1
fi
if ! docker compose version > /dev/null 2>&1; then
    printf 'Docker Compose est introuvable. Vérifie la commande docker compose version.\n' >&2
    exit 1
fi
if [[ ! -f Dockerfile || ! -f src/agent/cli.py ]]; then
    printf 'Place compose.memory.yaml et scripts/chat.sh dans le dépôt contenant Dockerfile et src/agent/cli.py.\n' >&2
    exit 1
fi

export CHAT_UID="$(id -u)"
export CHAT_GID="$(id -g)"
compose=(docker compose --project-name ict2-memory --file compose.memory.yaml)
"${compose[@]}" config --quiet

if ! docker info > /dev/null 2>&1; then
    printf 'Démarrage du moteur Docker…\n'
    if docker desktop version > /dev/null 2>&1; then
        docker desktop start
    elif command -v systemctl > /dev/null 2>&1 && systemctl --user cat docker.service > /dev/null 2>&1; then
        systemctl --user start docker
    elif command -v systemctl > /dev/null 2>&1 && systemctl cat docker.service > /dev/null 2>&1; then
        if [[ "$EUID" -eq 0 ]]; then
            systemctl start docker
        else
            sudo systemctl start docker
        fi
    else
        printf 'Démarrage automatique du moteur indisponible. Ouvre Docker Desktop et active son accès à Ubuntu, puis relance ce script.\n' >&2
        exit 1
    fi

    for attempt in {1..60}; do
        if docker info > /dev/null 2>&1; then
            break
        fi
        sleep 2
    done
    if ! docker info > /dev/null 2>&1; then
        printf 'Docker reste inaccessible. Le détail de docker info suit.\n' >&2
        docker info
        exit 1
    fi
fi

mkdir -p runtime
printf 'Démarrage et vérification du serveur Ollama…\n'
"${compose[@]}" up --detach --wait ollama

printf 'Vérification du modèle %s…\n' "${OLLAMA_MODEL:-qwen3:8b}"
"${compose[@]}" run --rm --no-deps model-init

printf 'Construction de l’application et ouverture du chat…\n'
"${compose[@]}" run --rm --no-deps --build chat

printf '\nDiscussion fermée. Pour arrêter les services Docker de cette fonctionnalité :\n'
printf 'docker compose --project-name ict2-memory --file compose.memory.yaml down\n'

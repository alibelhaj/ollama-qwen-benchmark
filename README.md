# Inférence LLM locale : Qwen2.5 via Ollama


## Fichiers
| Fichier | Rôle |
|---|---|
| `llm_bench.py` | Requêtes structurées (JSON Schema imposé via `format`), validation (schéma, enums, anti-hallucination, retry borné), mesures (latence, TTFT, tok/s, chargement à froid, pic de RSS via `/proc`), rapport JSON dans `results/` |
| `ablation_format.py` | Coût du décodage contraint : texte libre vs `json` vs schéma |
| `concurrency.py` | Débit agrégé selon le niveau de concurrence |

## Installation

### Prérequis
- Git
- Docker et Docker Compose
- Environ 2 Go de RAM libre et 1 Go de disque pour le modèle `qwen2.5:1.5b`

### Récupérer le projet
```bash
git clone https://github.com/alibelhaj/ollama-qwen-benchmark.git
cd ollama-qwen-benchmark
```

## installation
Trois services dans `docker-compose.yml` : `ollama` (le moteur, modèles stockés dans un volume), `model-pull` (télécharge le modèle une seule fois) et `app` (les scripts Python).

```bash
docker compose up -d ollama                     # démarre le moteur (+ téléchargement du modèle au 1er lancement)
docker compose run --rm app demo.py             # démo interactive : coller un ticket
docker stats                                    # mémoire réelle du conteneur Ollama
docker compose down                             # arrête tout (le volume garde le modèle)
```

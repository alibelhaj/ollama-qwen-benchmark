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

## Note technique

### 1. Résultats obtenus sur ma machine
i7-1355U, 15 Go de RAM, sans GPU, sur secteur · `qwen2.5:1.5b` (Q4_K_M) via Docker · 12 requêtes.

| Mesure | Valeur |
|---|---|
| Temps d'inférence p50 / p95 | 4,5 s / 5,4 s (13,7 s pour la 1re requête, chargement compris) |
| Débit | 13,1 requêtes/min · 16,8 tokens/s |
| Mémoire | 2,4 Go pour le conteneur Ollama (`docker stats`), dont 1,4 Go pour le modèle chargé |
| Conformité au 1er passage | 100 % |


### 2. Passer sur un GPU NVIDIA via Docker/CUDA
L'image Ollama contient déjà CUDA, donc le code ne change pas. Seule la couche d'infrastructure évolue :
- Sur l'hôte : le pilote NVIDIA et le NVIDIA Container Toolkit, pour que Docker puisse exposer le GPU au conteneur.
- Dans `docker-compose.yml`, réserver le GPU pour le service `ollama` :
  ```yaml
  deploy:
    resources:
      reservations:
        devices: [{driver: nvidia, count: all, capabilities: [gpu]}]
  ```

### 3. Reprendre un pipeline existant sans régression
Le principe : on ne peut pas affirmer qu'il n'y a pas de régression sans une référence mesurée.
1. Avant de modifier quoi que ce soit, constituer un jeu de référence à partir de données réelles et mesurer l'état actuel : exactitude, conformité, latence.
2. Changer une seule chose à la fois (modèle, prompt, schéma ou paramètres), pour savoir quelle modification cause quel effet. La CI bloque toute baisse par rapport à la référence.
3. Déployer progressivement : d'abord en parallèle de l'existant pour comparer sur du trafic réel, puis sur une partie des utilisateurs, avec un retour arrière possible.

### 4. JSON valide mais contenant une information absente du document
Le schéma garantit la forme de la réponse, pas sa vérité. Il faut donc un contrôle supplémentaire :
- Vérifier que chaque valeur extraite apparaît bien dans le document source. C'est déjà fait pour le numéro de commande dans `validate()`.
- Si ce n'est pas le cas, rejeter la réponse et relancer en indiquant l'erreur au modèle.
- Si l'erreur persiste, mettre le champ à `null` ou envoyer le cas en revue humaine. Une valeur manquante est visible, alors qu'une valeur inventée se propage sans qu'on la voie.
- Prévenir en amont : autoriser `null` dans le schéma. Un modèle obligé de remplir un champ finit par inventer.

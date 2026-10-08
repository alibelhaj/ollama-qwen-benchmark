# Inférence LLM locale : Qwen2.5 via Ollama

## Environnement
- CPU i7-1355U (hybride : 2 P-cores + 8 E-cores, 12 threads, 15 W), 15 Go RAM, pas de GPU, WSL2
- **Le portable était sur batterie pendant toutes les mesures** (`Discharging`)
- Ollama 0.6.6, modèle `qwen2.5:1.5b` (Q4_K_M, 986 Mo), backend `libggml-cpu-alderlake` (AVX2)

## Fichiers
| Fichier | Rôle |
|---|---|
| `llm_bench.py` | Requêtes structurées (JSON Schema imposé via `format`), validation (schéma, enums, anti-hallucination, retry borné), mesures (latence, TTFT, tok/s, chargement à froid, pic de RSS via `/proc`), rapport JSON dans `results/` |
| `ablation_format.py` | Coût du décodage contraint : texte libre vs `json` vs schéma |
| `concurrency.py` | Débit agrégé selon le niveau de concurrence |

```bash
ollama pull qwen2.5:1.5b
python3 llm_bench.py --runs 2 --num-thread 2 --prompt v2
python3 ablation_format.py --num-thread 2
python3 concurrency.py --levels 1 4 --num-thread 4
```

## Avec Docker (aucune installation à part Docker)
Trois services dans `docker-compose.yml` : `ollama` (le moteur, modèles stockés dans un volume), `model-pull` (télécharge le modèle une seule fois) et `app` (les scripts Python).

```bash
docker compose up -d ollama                     # démarre le moteur (+ téléchargement du modèle au 1er lancement)
docker compose run --rm app demo.py             # démo interactive : coller un ticket
docker compose run --rm app demo.py --exemples  # rejoue les 6 tickets de test
docker compose run --rm app llm_bench.py --runs 2 --num-thread 2   # benchmark → ./results
docker stats                                    # mémoire réelle du conteneur Ollama
docker compose down                             # arrête tout (le volume garde le modèle)
```

Variables : `MODEL=qwen2.5:3b docker compose …` pour changer de modèle, `NUM_THREAD=2` pour fixer les threads.
Dans Docker, `rss_peak_mb` vaut `null` (Ollama tourne dans un autre conteneur) : la mémoire se lit avec `docker stats` (~1,9 Go mesuré).

**Machine sans internet** : sur une machine connectée, `docker save ollama/ollama:0.6.6 test-technique-app | gzip > images.tar.gz`, et copier le volume des modèles ; sur la cible, `docker load < images.tar.gz`.

**Avec GPU NVIDIA** : ajouter au service `ollama` (nécessite le NVIDIA Container Toolkit) :
```yaml
    deploy:
      resources:
        reservations:
          devices: [{driver: nvidia, count: all, capabilities: [gpu]}]
```

## Résultats mesurés (sur batterie)
| Mesure | Valeur |
|---|---|
| Démarrage à froid (1re requête) | 24,9 s (runner : 2,3 à 17 s selon la pression mémoire) |
| Latence à chaud p50 / p95 | 9,7 s / 15,3 s (~60 tokens générés) |
| TTFT p50 | 1,1 s |
| Génération | 4 à 9 tok/s · évaluation du prompt ~120 à 1 600 tok/s |
| RSS du runner (pic) | ~1,4 à 1,5 Go (poids 935 Mo + KV 224 Mo + calcul 303 Mo) |
| JSON valide / validé dès le 1er essai | 100 % / 100 % |
| Exactitude des catégories | prompt v1 : **50 %** (biais vers « technique ») · prompt v2 : **100 %** · numéro de commande : 100 % |

**Nombre de threads** (`num_thread`, génération en tok/s) : 2 → 6,5 · 4 → 5,9 · 6 → 3,7 · 8 → 2,2 (p95 = 98 s). Le test à 12 threads a été interrompu.

**Ablation du format** : libre 5,4 · json 4,7 · schéma 5,5 tok/s. La grammaire ne coûte rien de mesurable.

**Prompt v1 vs v2** (même session, 2 threads, machine moins chargée) :

| | v1 | v2 |
|---|---|---|
| Bonne catégorie | 3/6 (50 %) | **6/6 (100 %)** |
| Latence p50 | 7,9 s | 7,4 s |
| Génération | 9,8 tok/s | 10,0 tok/s |

La v2 corrige les erreurs sans coût mesurable. Attention : 6 exemples, c'est trop peu pour conclure en production ; il faudrait un jeu d'évaluation d'au moins 100 tickets.

**Concurrence (8 requêtes)** : c=1 → 5,7 req/min · c=4 → 7,3 req/min (+28 %), mais la latence p50 passe de 9,7 s à 41 s.

## Diagnostic : la démarche
1. **Mesurer avant d'optimiser, en décomposant la latence** : chargement / évaluation du prompt (TTFT) / génération. Ollama renvoie `load_duration`, `prompt_eval_*` et `eval_*` ; le client mesure en plus le temps total et le TTFT réel. On sépare toujours froid et chaud, et on regarde p50 **et** p95 (ici la variance était le premier symptôme).
2. **Comparer au plafond théorique.** En génération sur CPU, le débit est borné par la bande passante mémoire : débit ≈ bande passante / taille des poids. Avec ~1 Go et 30 à 50 Go/s, on attend plus de 20 tok/s. 5 tok/s indique donc un problème d'environnement, pas de modèle.
3. **Une hypothèse, une ablation** :
   - Les threads → plus de threads donne *moins* de débit : contention (E-cores, hyperthreading, vCPU WSL partagés avec Windows).
   - La grammaire JSON → réfutée par l'ablation.
   - Le backend → le bon (`alderlake`, AVX2), confirmé dans `journalctl -u ollama`.
   - L'alimentation → **sur batterie : cause principale probable** (bridage de la fréquence, variance).
   - La mémoire → `n_ctx` réel = 8192 et non 2048, car `--parallel 4` réserve 4 slots de KV ; avec du swap actif et `--no-mmap`, le chargement varie de 2 à 17 s.
4. **Tester la qualité séparément de la vitesse** : un jeu de référence montre que 100 % de JSON valide ne garantit pas des réponses justes (50 % de catégories correctes).

## Optimisations (de la plus rentable à la moins)
- **Environnement** : secteur + mode performance Windows ; fermer les processus concurrents ; dans `.wslconfig`, donner plus de RAM ou moins de swap ; épingler sur les P-cores (`taskset`) avec `num_thread` = nombre de cœurs physiques rapides (2 à 4 ici).
- **Configuration Ollama** :
  - `OLLAMA_NUM_PARALLEL=1` pour une latence unitaire et moins de mémoire, ou ≥ 4 pour le débit (batching : +28 % mesuré).
  - `OLLAMA_KEEP_ALIVE` long pour ne jamais repayer le démarrage à froid.
  - Plafonner `num_ctx` et `num_predict` au besoin réel.
  - `OLLAMA_FLASH_ATTENTION=1` et un cache KV `q8_0` pour réduire la mémoire.
- **Prompt et schéma** :
  - Un prompt système stable et identique d'une requête à l'autre réutilise le cache de préfixe : le TTFT passe de 1,1 s à 0,15 s quand le préfixe est en cache (1 671 tok/s observés).
  - Des sorties courtes : la génération domine la latence.
  - Le prompt v2 (définitions des catégories + `resume` placé avant `categorie`) fait passer l'exactitude de 50 % à 100 % sur le jeu de test, sans surcoût.
- **Modèle et quantification** : Q4_K_M est un bon compromis sur CPU. Corriger le prompt avant de changer de modèle (cf. v2) ; si cela ne suffit pas, passer à `qwen2.5:3b`/`7b` ou ajouter quelques exemples (few-shot).
- **Passage à l'échelle** : avec un GPU, vLLM (batching continu, PagedAttention, décodage guidé). En production, exporter ces métriques vers Prometheus et alerter sur le p95 et le taux d'échec de validation.

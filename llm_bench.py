#!/usr/bin/env python3
"""
Pipeline d'inférence locale (Ollama) avec sorties structurées + benchmark.

- Soumet des requêtes structurées (JSON Schema imposé via le paramètre `format` d'Ollama).
- Contrôle les réponses : parsing JSON, validation de schéma, règles métier, retry borné.
- Mesure : latence client, TTFT, débit (tokens/s), temps de chargement, RSS du runner
  Ollama (pic, échantillonné), mémoire système disponible.
- Écrit un rapport JSON détaillé + affiche un résumé.

Dépendances : stdlib + requests.
Usage :
    python3 llm_bench.py --model qwen2.5:1.5b --runs 2
    python3 llm_bench.py --model qwen2.5:1.5b --num-thread 6 --num-ctx 1024
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import requests

OLLAMA_URL = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

# --------------------------------------------------------------------------- #
# 1. Contrat de sortie : JSON Schema (envoyé au modèle ET utilisé pour valider)
# --------------------------------------------------------------------------- #
CATEGORIES = ["facturation", "technique", "livraison", "compte", "autre"]
PRIORITIES = ["basse", "moyenne", "haute", "critique"]
SENTIMENTS = ["negatif", "neutre", "positif"]

TICKET_SCHEMA = {
    "type": "object",
    "properties": {
        "categorie": {"type": "string", "enum": CATEGORIES},
        "priorite": {"type": "string", "enum": PRIORITIES},
        "sentiment": {"type": "string", "enum": SENTIMENTS},
        "resume": {"type": "string"},
        "numero_commande": {"type": ["string", "null"]},
    },
    "required": ["categorie", "priorite", "sentiment", "resume", "numero_commande"],
}

SYSTEM_PROMPT = (
    "Tu es un classifieur de tickets de support client. "
    "Réponds UNIQUEMENT avec un objet JSON conforme au schéma. "
    f"categorie ∈ {CATEGORIES}; priorite ∈ {PRIORITIES}; sentiment ∈ {SENTIMENTS}. "
    "resume : une phrase de 20 mots maximum, en français. "
    "numero_commande : le numéro de commande cité (ex: 'CMD-1234'), sinon null."
)

# v2 : définitions explicites des catégories + `resume` généré AVANT `categorie`
# (le décodage contraint suit l'ordre des propriétés : le modèle "lit" avant de classer).
TICKET_SCHEMA_V2 = {
    "type": "object",
    "properties": {
        "resume": {"type": "string"},
        "numero_commande": {"type": ["string", "null"]},
        "categorie": {"type": "string", "enum": CATEGORIES},
        "priorite": {"type": "string", "enum": PRIORITIES},
        "sentiment": {"type": "string", "enum": SENTIMENTS},
    },
    "required": ["resume", "numero_commande", "categorie", "priorite", "sentiment"],
}

SYSTEM_PROMPT_V2 = (
    "Tu es un classifieur de tickets de support client. Réponds UNIQUEMENT en JSON.\n"
    "resume : une phrase de 20 mots maximum, en français.\n"
    "numero_commande : le numéro cité (format 'CMD-1234'), sinon null.\n"
    "categorie (choisis selon le SUJET du ticket, pas son ton) :\n"
    "- facturation : paiement, prélèvement, remboursement, facture, prix\n"
    "- technique : bug, plantage, erreur d'application ou du site\n"
    "- livraison : colis, expédition, suivi, retard ou réception d'une commande\n"
    "- compte : connexion, mot de passe, profil, inscription\n"
    "- autre : tout le reste\n"
    f"priorite ∈ {PRIORITIES} ; sentiment ∈ {SENTIMENTS}."
)

PROMPTS = {"v1": (SYSTEM_PROMPT, TICKET_SCHEMA), "v2": (SYSTEM_PROMPT_V2, TICKET_SCHEMA_V2)}

# Jeu d'évaluation avec vérité terrain (pour mesurer l'exactitude, pas seulement la vitesse)
DATASET = [
    {"text": "Bonjour, j'ai été débité deux fois pour la commande CMD-4821. Merci de me rembourser rapidement.",
     "expected": {"categorie": "facturation", "numero_commande": "CMD-4821"}},
    {"text": "L'application plante à chaque ouverture depuis la mise à jour d'hier, impossible de travailler. Toute l'équipe est bloquée !",
     "expected": {"categorie": "technique", "numero_commande": None}},
    {"text": "Mon colis CMD-1177 devait arriver lundi, on est jeudi et le suivi n'a pas bougé.",
     "expected": {"categorie": "livraison", "numero_commande": "CMD-1177"}},
    {"text": "Je n'arrive plus à me connecter, le lien de réinitialisation du mot de passe ne fonctionne pas.",
     "expected": {"categorie": "compte", "numero_commande": None}},
    {"text": "Super service, livraison rapide pour CMD-9003, merci à toute l'équipe !",
     "expected": {"categorie": "livraison", "numero_commande": "CMD-9003"}},
    {"text": "Pouvez-vous m'envoyer une facture au nom de ma société pour la commande CMD-3310 ?",
     "expected": {"categorie": "facturation", "numero_commande": "CMD-3310"}},
]


# --------------------------------------------------------------------------- #
# 2. Contrôle des réponses
# --------------------------------------------------------------------------- #
class ValidationError(Exception):
    pass


def _check_type(value, expected) -> bool:
    types = expected if isinstance(expected, list) else [expected]
    mapping = {"string": str, "null": type(None), "object": dict, "integer": int, "number": (int, float)}
    return any(isinstance(value, mapping[t]) for t in types)


def validate(obj, schema=TICKET_SCHEMA, source_text: str = "") -> dict:
    """Validation de schéma (sous-ensemble JSON Schema) + règles métier."""
    if not isinstance(obj, dict):
        raise ValidationError("la racine n'est pas un objet")
    missing = [k for k in schema["required"] if k not in obj]
    if missing:
        raise ValidationError(f"champs manquants: {missing}")
    for key, spec in schema["properties"].items():
        val = obj.get(key)
        if not _check_type(val, spec["type"]):
            raise ValidationError(f"{key}: type invalide ({type(val).__name__})")
        if "enum" in spec and val not in spec["enum"]:
            raise ValidationError(f"{key}: '{val}' hors enum")
    # Règles métier : anti-hallucination sur le numéro de commande, longueur du résumé
    num = obj["numero_commande"]
    if num is not None and num not in source_text:
        raise ValidationError(f"numero_commande '{num}' absent du texte source (hallucination)")
    if len(obj["resume"].split()) > 30:
        raise ValidationError("resume trop long")
    return obj


# --------------------------------------------------------------------------- #
# 3. Mesure mémoire : échantillonnage du RSS des processus Ollama via /proc
# --------------------------------------------------------------------------- #
def _ollama_pids() -> list[int]:
    pids = []
    for d in Path("/proc").iterdir():
        if d.name.isdigit():
            try:
                cmd = (d / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="ignore")
            except OSError:
                continue
            if "ollama" in cmd and ("serve" in cmd or "runner" in cmd):
                pids.append(int(d.name))
    return pids


def _rss_mb(pid: int) -> float:
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    except OSError:
        pass
    return 0.0


def mem_available_mb() -> float:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1024
    return 0.0


class MemorySampler(threading.Thread):
    """Échantillonne le RSS cumulé des processus Ollama toutes les `interval` s."""

    def __init__(self, interval: float = 0.05):
        super().__init__(daemon=True)
        self.interval = interval
        self.samples: list[float] = []
        self._halt = threading.Event()

    def run(self):
        while not self._halt.is_set():
            pids = _ollama_pids()
            if pids:  # vide si Ollama tourne dans un autre conteneur : utiliser `docker stats`
                self.samples.append(sum(_rss_mb(p) for p in pids))
            time.sleep(self.interval)

    def stop(self) -> dict:
        self._halt.set()
        self.join()
        s = self.samples
        if not s:
            return {"rss_peak_mb": None, "rss_mean_mb": None}
        return {"rss_peak_mb": round(max(s), 1), "rss_mean_mb": round(statistics.mean(s), 1)}


# --------------------------------------------------------------------------- #
# 4. Client Ollama (streaming pour mesurer le TTFT côté client)
# --------------------------------------------------------------------------- #
@dataclass
class CallResult:
    ok: bool
    attempts: int
    wall_s: float
    ttft_s: float | None
    load_s: float
    prompt_tokens: int
    prompt_eval_s: float
    gen_tokens: int
    gen_s: float
    gen_tps: float
    prompt_tps: float
    output: dict | None = None
    error: str | None = None
    correct_category: bool | None = None
    correct_order_id: bool | None = None
    mem: dict = field(default_factory=dict)


def chat(model: str, text: str, options: dict, keep_alive: str, prompt: str = "v1", timeout: int = 300):
    system, schema = PROMPTS[prompt]
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": text}],
        "format": schema,  # décodage contraint par grammaire côté llama.cpp
        "stream": True,
        "options": options,
        "keep_alive": keep_alive,
    }
    t0 = time.perf_counter()
    ttft, chunks, final = None, [], {}
    with requests.post(f"{OLLAMA_URL}/api/chat", json=payload, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line:
                continue
            msg = json.loads(line)
            piece = msg.get("message", {}).get("content", "")
            if piece and ttft is None:
                ttft = time.perf_counter() - t0
            chunks.append(piece)
            if msg.get("done"):
                final = msg
    return "".join(chunks), final, time.perf_counter() - t0, ttft


def run_one(model: str, item: dict, options: dict, keep_alive: str, max_retries: int,
            prompt: str = "v1") -> CallResult:
    sampler = MemorySampler()
    sampler.start()
    agg = dict(wall=0.0, load=0.0, ptok=0, pdur=0.0, gtok=0, gdur=0.0)
    ttft, output, error, attempt = None, None, None, 0
    try:
        for attempt in range(1, max_retries + 2):
            content, final, wall, t = chat(model, item["text"], options, keep_alive, prompt)
            ttft = ttft if ttft is not None else t
            agg["wall"] += wall
            agg["load"] += final.get("load_duration", 0) / 1e9
            agg["ptok"] += final.get("prompt_eval_count", 0)
            agg["pdur"] += final.get("prompt_eval_duration", 0) / 1e9
            agg["gtok"] += final.get("eval_count", 0)
            agg["gdur"] += final.get("eval_duration", 0) / 1e9
            try:
                output = validate(json.loads(content), PROMPTS[prompt][1], source_text=item["text"])
                error = None
                break
            except (json.JSONDecodeError, ValidationError) as e:
                error = f"{type(e).__name__}: {e}"
                options = {**options, "seed": options.get("seed", 0) + attempt}  # varier au retry
    except requests.RequestException as e:
        error = f"HTTP: {e}"
    mem = sampler.stop()

    exp = item["expected"]
    return CallResult(
        ok=output is not None,
        attempts=attempt,
        wall_s=round(agg["wall"], 3),
        ttft_s=round(ttft, 3) if ttft else None,
        load_s=round(agg["load"], 3),
        prompt_tokens=agg["ptok"],
        prompt_eval_s=round(agg["pdur"], 3),
        gen_tokens=agg["gtok"],
        gen_s=round(agg["gdur"], 3),
        gen_tps=round(agg["gtok"] / agg["gdur"], 1) if agg["gdur"] else 0.0,
        prompt_tps=round(agg["ptok"] / agg["pdur"], 1) if agg["pdur"] else 0.0,
        output=output,
        error=error,
        correct_category=(output["categorie"] == exp["categorie"]) if output else None,
        correct_order_id=(output["numero_commande"] == exp["numero_commande"]) if output else None,
        mem=mem,
    )


def unload(model: str):
    """Décharge le modèle pour mesurer un vrai démarrage à froid."""
    requests.post(f"{OLLAMA_URL}/api/generate", json={"model": model, "keep_alive": 0}, timeout=60)
    time.sleep(1)


def pct(values, p):
    v = sorted(values)
    k = (len(v) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="qwen2.5:1.5b")
    ap.add_argument("--runs", type=int, default=2, help="passes sur le dataset (à chaud)")
    ap.add_argument("--num-ctx", type=int, default=2048)
    ap.add_argument("--num-thread", type=int, default=None)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-retries", type=int, default=2)
    ap.add_argument("--prompt", choices=list(PROMPTS), default="v2")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    options = {"temperature": args.temperature, "num_ctx": args.num_ctx, "num_predict": 200, "seed": 42}
    if args.num_thread:
        options["num_thread"] = args.num_thread

    print(f"Modèle: {args.model} | prompt: {args.prompt} | options: {options}")
    mem_before = mem_available_mb()

    # --- Démarrage à froid ---
    unload(args.model)
    cold = run_one(args.model, DATASET[0], options, "10m", args.max_retries, args.prompt)
    print(f"[froid] wall={cold.wall_s}s load={cold.load_s}s ttft={cold.ttft_s}s "
          f"rss_peak={cold.mem['rss_peak_mb']}MB ok={cold.ok}")

    ps = requests.get(f"{OLLAMA_URL}/api/ps", timeout=10).json().get("models", [])
    model_info = next((m for m in ps if m["name"] == args.model), {})

    # --- À chaud ---
    warm: list[CallResult] = []
    for r in range(args.runs):
        for i, item in enumerate(DATASET):
            res = run_one(args.model, item, options, "10m", args.max_retries, args.prompt)
            warm.append(res)
            flag = "OK " if res.ok else "ERR"
            print(f"[run {r+1} #{i}] {flag} wall={res.wall_s:6.2f}s ttft={res.ttft_s}s "
                  f"gen={res.gen_tokens}tok @{res.gen_tps} tok/s prompt@{res.prompt_tps} tok/s "
                  f"try={res.attempts} cat={res.output and res.output['categorie']}"
                  + (f" | {res.error}" if res.error else ""))

    walls = [w.wall_s for w in warm]
    ttfts = [w.ttft_s for w in warm if w.ttft_s]
    valid = [w for w in warm if w.ok]
    summary = {
        "model": args.model,
        "options": options,
        "prompt": args.prompt,
        "model_loaded": {k: model_info.get(k) for k in ("size", "size_vram", "details")},
        "cold_start": {"wall_s": cold.wall_s, "load_s": cold.load_s, "ttft_s": cold.ttft_s, **cold.mem},
        "warm": {
            "n_requests": len(warm),
            "latency_p50_s": round(pct(walls, 50), 3),
            "latency_p95_s": round(pct(walls, 95), 3),
            "ttft_p50_s": round(pct(ttfts, 50), 3) if ttfts else None,
            "gen_tps_mean": round(statistics.mean(w.gen_tps for w in warm), 1),
            "prompt_tps_mean": round(statistics.mean(w.prompt_tps for w in warm), 1),
            "throughput_req_per_min": round(60 * len(warm) / sum(walls), 1),
            "rss_peak_mb": max((w.mem["rss_peak_mb"] for w in warm if w.mem["rss_peak_mb"]), default=None),
        },
        "quality": {
            "valid_json_rate": round(len(valid) / len(warm), 3),
            "first_try_rate": round(sum(1 for w in warm if w.ok and w.attempts == 1) / len(warm), 3),
            "category_accuracy": round(sum(w.correct_category for w in valid) / max(len(valid), 1), 3),
            "order_id_accuracy": round(sum(w.correct_order_id for w in valid) / max(len(valid), 1), 3),
        },
        "system": {"mem_available_before_mb": round(mem_before), "mem_available_after_mb": round(mem_available_mb()),
                   "cpu_count": os.cpu_count()},
    }

    out = Path(args.out)
    out.mkdir(exist_ok=True)
    tag = f"{args.model.replace(':', '_')}_ctx{args.num_ctx}_t{args.num_thread or 'auto'}_{args.prompt}"
    path = out / f"bench_{tag}.json"
    path.write_text(json.dumps({"summary": summary, "cold": asdict(cold),
                                "warm": [asdict(w) for w in warm]}, ensure_ascii=False, indent=2))
    print("\n=== RÉSUMÉ ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nRapport détaillé : {path}")


if __name__ == "__main__":
    main()

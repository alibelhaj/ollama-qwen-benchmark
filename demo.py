#!/usr/bin/env python3
"""
Démo client : classification de tickets de support par un LLM 100 % local (Ollama).

    python3 demo.py              # mode interactif : collez un ticket, Entrée
    python3 demo.py --exemples   # rejoue les tickets du jeu de test

Montre en direct : la réponse structurée (JSON), le contrôle automatique,
et les métriques (temps, vitesse, mémoire). Aucune donnée ne quitte la machine.
"""
import argparse
import json
import os
import sys

import requests

from llm_bench import DATASET, OLLAMA_URL, PROMPTS, ValidationError, chat, mem_available_mb, validate

B, G, R, Y, C, D, X = "\033[1m", "\033[32m", "\033[31m", "\033[33m", "\033[36m", "\033[2m", "\033[0m"
MODEL = os.environ.get("MODEL", "qwen2.5:1.5b")
OPTIONS = {"temperature": 0, "num_ctx": 2048, "num_predict": 200, "seed": 42}
if os.environ.get("NUM_THREAD"):  # vide = Ollama choisit ; 2 était optimal sur le portable de test
    OPTIONS["num_thread"] = int(os.environ["NUM_THREAD"])


def traiter(texte: str, attendu: dict | None = None):
    print(f"\n{D}Analyse en cours…{X}", end="", flush=True)
    content, final, wall, ttft = chat(MODEL, texte, OPTIONS, "30m", "v2")
    print("\r" + " " * 20 + "\r", end="")
    try:
        obj = validate(json.loads(content), PROMPTS["v2"][1], source_text=texte)
        print(f"{G}{B}✔ Réponse conforme au schéma{X}")
    except (json.JSONDecodeError, ValidationError) as e:
        obj = None
        print(f"{R}{B}✘ Réponse rejetée par le contrôle : {e}{X}")

    print(f"{C}{json.dumps(obj or content, ensure_ascii=False, indent=2)}{X}")
    if obj and attendu:
        ok = obj["categorie"] == attendu["categorie"]
        print(f"Catégorie attendue : {attendu['categorie']}  →  {G + '✔ correct' if ok else R + '✘ erreur'}{X}")

    n, d = final.get("eval_count", 0), final.get("eval_duration", 1) / 1e9
    print(f"{Y}⏱  {wall:.1f}s au total · premier token {ttft:.2f}s · {n / d:.1f} tokens/s · "
          f"RAM disponible {mem_available_mb() / 1024:.1f} Go{X}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exemples", action="store_true", help="rejouer le jeu de test")
    args = ap.parse_args()

    print(f"{B}=== Démo : tri automatique de tickets support — LLM local ({MODEL}) ==={X}")
    try:
        requests.get(f"{OLLAMA_URL}/api/tags", timeout=5).raise_for_status()
    except requests.RequestException:
        sys.exit(f"{R}Ollama ne répond pas sur {OLLAMA_URL}. Lancez : ollama serve{X}")

    print(f"{D}Préchargement du modèle (une seule fois)…{X}")
    chat(MODEL, "bonjour", OPTIONS, "30m", "v2")

    if args.exemples:
        for item in DATASET:
            print(f"\n{B}Ticket :{X} {item['text']}")
            traiter(item["text"], item["expected"])
        return

    print("Collez un ticket client puis Entrée (ligne vide ou Ctrl+C pour quitter).")
    while True:
        try:
            texte = input(f"\n{B}Ticket > {X}").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not texte:
            break
        traiter(texte)


if __name__ == "__main__":
    main()

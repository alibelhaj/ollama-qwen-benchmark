#!/usr/bin/env python3
"""
Ablation : isole le coût du décodage contraint.
Même prompt, mêmes options, trois modes : texte libre / format="json" / JSON Schema.
Compare le débit de génération (eval tokens/s) rapporté par Ollama.

Usage : python3 ablation_format.py --model qwen2.5:1.5b --reps 3 --num-thread 4
"""
import argparse
import statistics

import requests

from llm_bench import DATASET, OLLAMA_URL, SYSTEM_PROMPT, TICKET_SCHEMA

MODES = {"libre": None, "json": "json", "schema": TICKET_SCHEMA}


def one(model, fmt, text, options):
    payload = {"model": model, "stream": False, "options": options, "keep_alive": "10m",
               "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": text}]}
    if fmt is not None:
        payload["format"] = fmt
    r = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=300).json()
    return r["eval_count"], r["eval_duration"] / 1e9, r["total_duration"] / 1e9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5:1.5b")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--num-thread", type=int, default=None)
    args = ap.parse_args()
    options = {"temperature": 0, "num_ctx": 2048, "num_predict": 80, "seed": 42}
    if args.num_thread:
        options["num_thread"] = args.num_thread

    one(args.model, None, "ping", options)  # warm-up
    for name, fmt in MODES.items():
        tps, tot = [], []
        for _ in range(args.reps):
            for item in DATASET[:3]:
                n, d, t = one(args.model, fmt, item["text"], options)
                tps.append(n / d)
                tot.append(t)
        print(f"{name:7s} gen={statistics.median(tps):6.1f} tok/s (min {min(tps):.1f}, max {max(tps):.1f})"
              f"  total_p50={statistics.median(tot):.2f}s")


if __name__ == "__main__":
    main()

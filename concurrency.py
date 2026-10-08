#!/usr/bin/env python3
"""
Débit agrégé vs concurrence : Ollama sert jusqu'à OLLAMA_NUM_PARALLEL requêtes en batch
dans le même runner. On compare N requêtes séquentielles vs N en parallèle.

Usage : python3 concurrency.py --levels 1 4 --num-thread 4
"""
import argparse
import time
from concurrent.futures import ThreadPoolExecutor

from llm_bench import DATASET, chat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5:1.5b")
    ap.add_argument("--levels", type=int, nargs="+", default=[1, 4])
    ap.add_argument("--num-thread", type=int, default=None)
    ap.add_argument("--n", type=int, default=8, help="nombre total de requêtes par niveau")
    args = ap.parse_args()
    options = {"temperature": 0, "num_ctx": 2048, "num_predict": 200, "seed": 42}
    if args.num_thread:
        options["num_thread"] = args.num_thread

    chat(args.model, "ping", options, "10m", "v2")  # warm-up
    items = [DATASET[i % len(DATASET)]["text"] for i in range(args.n)]
    for c in args.levels:
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=c) as ex:
            res = list(ex.map(lambda t: chat(args.model, t, options, "10m", "v2"), items))
        dt = time.perf_counter() - t0
        tokens = sum(r[1].get("eval_count", 0) for r in res)
        lat = sorted(r[2] for r in res)
        print(f"concurrence={c}: {args.n} req en {dt:.1f}s -> {60*args.n/dt:.1f} req/min, "
              f"{tokens/dt:.1f} tok/s agrégés, latence p50={lat[len(lat)//2]:.1f}s max={lat[-1]:.1f}s")


if __name__ == "__main__":
    main()

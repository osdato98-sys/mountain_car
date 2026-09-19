"""Entrena un agente por bloques, evalúa (greedy) tras cada bloque y guarda
historial + mejor checkpoint.

Uso:
    python scripts/run_experiment.py qlearning --episodes 20000 --chunk 500
    python scripts/run_experiment.py dqn --episodes 2500 --chunk 50
"""
import argparse
import json
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from mountain_car.agents import DQNAgent, QLearningAgent

ENV_ID = "MountainCar-v0"
RESULTS = Path("results")
SAVES = Path("saves")


def evaluate(agent, n=50, seed=1000):
    env = gym.make(ENV_ID)
    rets, reached = [], 0
    for i in range(n):
        obs, _ = env.reset(seed=seed + i)
        total, term = 0.0, False
        while True:
            a, _ = agent.predict(obs, deterministic=True)
            obs, r, term, trunc, _ = env.step(int(a))
            total += r
            if term or trunc:
                break
        rets.append(total)
        reached += int(term)
    env.close()
    return float(np.mean(rets)), float(np.std(rets)), reached / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("agent", choices=["qlearning", "dqn"])
    ap.add_argument("--episodes", type=int, required=True)
    ap.add_argument("--chunk", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import random
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.set_num_threads(1)

    agent = QLearningAgent(ENV_ID) if args.agent == "qlearning" else DQNAgent(ENV_ID)
    ext = "pkl" if args.agent == "qlearning" else "pt"
    best_path = SAVES / f"{args.agent}_best.{ext}"
    SAVES.mkdir(exist_ok=True); RESULTS.mkdir(exist_ok=True)

    train_rewards, evals, best = [], [], -1e9
    t0 = time.time()
    done = 0
    while done < args.episodes:
        n = min(args.chunk, args.episodes - done)
        train_rewards += agent.train(total_episodes=n, log_interval=n)
        done += n
        m, s, frac = evaluate(agent)
        evals.append({"episode": done, "eval_mean": m, "eval_std": s, "success": frac,
                      "epsilon": agent.epsilon, "time_s": time.time() - t0})
        print(f"[eval] ep={done} mean={m:.1f}±{s:.1f} success={frac:.0%} t={time.time()-t0:.0f}s", flush=True)
        if m > best:
            best = m
            agent.save(best_path)
        json.dump({"agent": args.agent, "train_rewards": train_rewards, "evals": evals,
                   "best_eval_mean": best},
                  open(RESULTS / f"{args.agent}_history.json", "w"))
    agent.save(SAVES / f"{args.agent}_final.{ext}")
    print("BEST", best)


if __name__ == "__main__":
    main()

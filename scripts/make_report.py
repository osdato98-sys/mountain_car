"""Genera figuras de comparación y evidencia a partir de results/*_history.json."""
import json
from pathlib import Path

import gymnasium as gym
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mountain_car.agents import DQNAgent, QLearningAgent

R = Path("results")
q = json.load(open(R / "qlearning_history.json"))
d = json.load(open(R / "dqn_history.json"))

def smooth(x, w):
    x = np.asarray(x, float)
    return np.convolve(x, np.ones(w) / w, mode="valid")

# 1) Curvas de entrenamiento (recompensa por episodio, media móvil)
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
for a, h, name, w in [(ax[0], q, "Q-Learning tabular", 200), (ax[1], d, "DQN", 50)]:
    a.plot(h["train_rewards"], alpha=.2, color="tab:blue")
    a.plot(np.arange(w - 1, len(h["train_rewards"])), smooth(h["train_rewards"], w), color="tab:blue", label=f"media móvil ({w})")
    a.set(title=name, xlabel="Episodio de entrenamiento", ylabel="Recompensa"); a.axhline(-110, ls="--", c="g", label="umbral -110"); a.legend()
plt.tight_layout(); plt.savefig(R / "training_curves.png", dpi=130); plt.close()

# 2) Evaluación greedy vs episodios (comparación directa)
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
for h, name, c in [(q, "Q-Learning", "tab:orange"), (d, "DQN", "tab:blue")]:
    e = h["evals"]; x = [i["episode"] for i in e]; m = np.array([i["eval_mean"] for i in e]); s = np.array([i["eval_std"] for i in e])
    ax[0].plot(x, m, label=name, color=c); ax[0].fill_between(x, m - s, m + s, alpha=.15, color=c)
    ax[1].plot(x, [i["success"] * 100 for i in e], label=name, color=c)
ax[0].axhline(-110, ls="--", c="g"); ax[0].set(title="Evaluación greedy (50 episodios)", xlabel="Episodios de entrenamiento", ylabel="Recompensa media"); ax[0].set_xscale("log"); ax[0].legend()
ax[1].set(title="% de episodios que alcanzan la bandera", xlabel="Episodios de entrenamiento", ylabel="%"); ax[1].set_xscale("log"); ax[1].legend()
plt.tight_layout(); plt.savefig(R / "comparison_eval.png", dpi=130); plt.close()

# 3) Evidencia del mejor modelo: 100 episodios + política/valor + trayectoria
def rollout(agent, seed):
    env = gym.make("MountainCar-v0"); obs, _ = env.reset(seed=seed); tr = [obs.copy()]; tot = 0; term = False
    while True:
        a, _ = agent.predict(obs, deterministic=True); obs, r, term, trunc, _ = env.step(int(a)); tot += r; tr.append(obs.copy())
        if term or trunc: break
    return tot, term, np.array(tr)

qa = QLearningAgent.load(Path("saves/qlearning_best.pkl")); da = DQNAgent.load(Path("saves/dqn_best.pt"))
summary = {}
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
for a, ag, name, c in [(ax[0], qa, "Q-Learning", "tab:orange"), (ax[1], da, "DQN", "tab:blue")]:
    res = [rollout(ag, 5000 + i) for i in range(100)]
    rets = np.array([r[0] for r in res]); succ = np.mean([r[1] for r in res])
    summary[name] = {"mean": float(rets.mean()), "std": float(rets.std()), "best": float(rets.max()), "worst": float(rets.min()), "success": float(succ)}
    for _, _, tr in res[:10]: a.plot(tr[:, 0], tr[:, 1], alpha=.5, color=c)
    a.axvline(.5, c="g", ls="--"); a.set(title=f"{name}: 10 trayectorias (posición vs velocidad)", xlabel="posición", ylabel="velocidad")
plt.tight_layout(); plt.savefig(R / "best_trajectories.png", dpi=130); plt.close()

# Política aprendida (acción greedy) sobre el plano posición-velocidad
P, V = np.meshgrid(np.linspace(-1.2, .6, 200), np.linspace(-.07, .07, 200))
grid = np.stack([P.ravel(), V.ravel()], 1).astype(np.float32)
pol_q = np.array([qa.predict(s)[0] for s in grid]).reshape(P.shape)
pol_d = np.array([da.predict(s)[0] for s in grid]).reshape(P.shape)
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
for a, pol, name in [(ax[0], pol_q, "Q-Learning (tabla 20x20)"), (ax[1], pol_d, "DQN (red MLP)")]:
    im = a.pcolormesh(P, V, pol, cmap="coolwarm", shading="auto", vmin=0, vmax=2); a.set(title=f"Política greedy: {name}", xlabel="posición", ylabel="velocidad")
cb = fig.colorbar(im, ax=ax, ticks=[0, 1, 2]); cb.ax.set_yticklabels(["izq", "nada", "der"])
plt.savefig(R / "learned_policies.png", dpi=130, bbox_inches="tight"); plt.close()

# Métricas de velocidad de aprendizaje: primer episodio con éxito >= 90% y media >= umbral
def first(h, cond):
    for e in h["evals"]:
        if cond(e): return e["episode"], e["time_s"]
    return None, None
for name, h in [("Q-Learning", q), ("DQN", d)]:
    summary[name]["first_success90"] = first(h, lambda e: e["success"] >= .9)
    summary[name]["first_mean_ge_-130"] = first(h, lambda e: e["eval_mean"] >= -130)
    summary[name]["total_time_s"] = h["evals"][-1]["time_s"]
    last = h["evals"][-10:]
    summary[name]["eval_std_last10_checkpoints"] = float(np.std([e["eval_mean"] for e in last]))
json.dump(summary, open(R / "summary.json", "w"), indent=2)
print(json.dumps(summary, indent=2))

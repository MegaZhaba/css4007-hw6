"""HW6 Part B: random vs fixed policy on Gymnasium FrozenLake-v1 (4x4).

Run:  python src/frozenlake_sim.py
Writes outputs/partB_episodes.csv, outputs/partB_summary.md, outputs/partB_traces.txt
"""
from __future__ import annotations

import csv
import statistics
from pathlib import Path

import gymnasium as gym
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"

MAX_STEPS = 100
BATCH_SEEDS = [11, 22, 33]
EPISODES = 50
LEFT, DOWN, RIGHT, UP = 0, 1, 2, 3
NAMES = {LEFT: "Left", DOWN: "Down", RIGHT: "Right", UP: "Up"}

# Map (state = row*4 + col):
#   S  F  F  F      0  1  2  3
#   F  H  F  H      4  5  6  7
#   F  F  F  H      8  9 10 11
#   H  F  F  G     12 13 14 15
GOAL, HOLES = 15, {5, 7, 11, 12}

# Fixed policy, written from the map BEFORE running any experiment.
# Main route: 0 v 4 v 8 > 9 v 13 > 14 > 15 (6 steps, never steps next to a hole
# in the direction of travel). Other states route back onto that path.
FIXED_POLICY = {
    0: DOWN,    # start -> (1,0); going right would run along the top above hole 5
    1: LEFT,    # back to the start column, away from hole 5 below
    2: DOWN,    # -> 6 -> 10 -> 14, a parallel safe column
    3: LEFT,    # down would be hole 7
    4: DOWN,    # right would be hole 5
    6: DOWN,    # left = hole 5, right = hole 7
    8: RIGHT,   # down would be hole 12
    9: DOWN,    # -> 13 on the bottom row
    10: DOWN,   # right would be hole 11
    13: RIGHT,  # -> 14
    14: RIGHT,  # -> goal
}


def policy_fixed(state, rng):
    return FIXED_POLICY[state]


def policy_random(state, rng):
    return int(rng.integers(4))       # uniform over the 4 actions, seeded generator


POLICIES = {"random": policy_random, "fixed": policy_fixed}


def run_episode(env, policy, seed, render=False):
    """reset()/step() loop; stops on terminated or truncated, hard cap of 100 steps."""
    rng = np.random.default_rng(seed)       # action RNG seeded per episode
    state, _ = env.reset(seed=seed)
    total, steps, frames = 0.0, 0, []
    if render:
        frames.append(env.render())
    terminated = truncated = False
    trace = [state]
    while not (terminated or truncated) and steps < MAX_STEPS:
        action = policy(state, rng)
        state, reward, terminated, truncated, _ = env.step(action)
        total += reward
        steps += 1
        trace.append((NAMES[action], state))
        if render:
            frames.append(env.render())
    # terminal outcome wins even if it happened on the very last allowed step
    if terminated:
        outcome = "goal" if state == GOAL else "hole"
    else:
        outcome = "time_limit"
    return dict(steps=steps, total_reward=total, outcome=outcome, trace=trace, frames=frames)


def fmt_trace(trace):
    return f"{trace[0]} " + " ".join(f"-{a}-> {s}" for a, s in trace[1:])


def main():
    OUT.mkdir(exist_ok=True)
    assert all(s in FIXED_POLICY for s in range(16) if s not in HOLES | {GOAL})
    rows, summary, examples = [], [], {}

    for slippery in (False, True):
        env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=slippery,
                       max_episode_steps=MAX_STEPS, render_mode="ansi")
        for pname, pol in POLICIES.items():
            cond = f"slippery={slippery}/{pname}"
            for b in BATCH_SEEDS:
                eps = []
                for i in range(EPISODES):
                    seed = 1000 * b + i
                    r = run_episode(env, pol, seed)
                    eps.append(r)
                    rows.append(dict(condition=cond, slippery=slippery, batch_seed=b,
                                     episode_seed=seed, policy=pname, steps=r["steps"],
                                     total_reward=r["total_reward"], outcome=r["outcome"]))
                    examples.setdefault((cond, r["outcome"]), (seed, r["trace"]))
                g = sum(e["outcome"] == "goal" for e in eps)
                summary.append((slippery, pname, b, g,
                                sum(e["outcome"] == "hole" for e in eps),
                                sum(e["outcome"] == "time_limit" for e in eps),
                                100 * g / EPISODES,
                                statistics.mean(e["steps"] for e in eps)))
        env.close()

    with open(OUT / "partB_episodes.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    table = ["| Slippery? | Policy | Batch seed | Goals /50 | Holes /50 | Time limits /50 "
             "| Success % | Mean steps |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for sl, p, b, g, h, t, sp, ms in summary:
        table.append(f"| {sl} | {p} | {b} | {g} | {h} | {t} | {sp:.0f}% | {ms:.2f} |")
    (OUT / "partB_summary.md").write_text("\n".join(table) + "\n")
    print("\n".join(table))

    # Text-rendered episodes + example traces
    lines = []
    for slippery in (False, True):
        env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=slippery,
                       max_episode_steps=MAX_STEPS, render_mode="ansi")
        r = run_episode(env, policy_fixed, 11000, render=True)
        lines.append(f"=== Rendered episode: slippery={slippery}, fixed policy, seed 11000 -> "
                     f"{r['outcome']} in {r['steps']} steps (agent shown as [x]) ===")
        # ansi render highlights the agent with a red background; make it plain text
        lines += [fr.replace("\x1b[41m", "[").replace("\x1b[0m", "]") for fr in r["frames"]]
        env.close()
    lines.append("\n=== Example traces (state -action-> next state) ===")
    for cond in sorted({k[0] for k in examples}):
        for outcome in ("goal", "hole", "time_limit"):
            if (cond, outcome) in examples:
                seed, tr = examples[(cond, outcome)]
                lines.append(f"[{cond}] {outcome} (seed {seed}, {len(tr) - 1} steps): {fmt_trace(tr)}")
            else:
                lines.append(f"[{cond}] {outcome}: never observed in 150 episodes")
    (OUT / "partB_traces.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[-12:]))


if __name__ == "__main__":
    main()

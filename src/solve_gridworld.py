"""HW6 Part C: value iteration on the Part A map and the extended map + simulations.

Run:  python src/solve_gridworld.py
Writes outputs/partC_*.txt / .csv
"""
from __future__ import annotations

import csv
import statistics
from pathlib import Path

from gridworld import (ACTIONS, ARROWS, GridMDP, ROOT, load_config, load_floor_rules,
                       one_based)

OUT = ROOT / "outputs"


def solve_map(name, rows, cfg, rules, log):
    sim = cfg["simulation"]
    solver = cfg["solver"]
    mdp = GridMDP(rows, rules, cfg["discount"])
    log(f"\n{'=' * 70}\nMAP: {name}\n" + "\n".join(rows))

    V, iters, delta, converged, hist = mdp.value_iteration(solver["tolerance"],
                                                           solver["max_iterations"])

    log("\nV1:")
    log(mdp.format_values(hist[1], 3))
    log("\nV2 at the requested positions (with all four Q-values from V1):")
    for r, c in cfg["paper_v2_positions"]:
        s = (r - 1, c - 1)
        qs = {a: mdp.q_value(hist[1], s, a) for a in ACTIONS}
        q_txt = ", ".join(f"{a}={q:.4f}" for a, q in qs.items())
        log(f"  ({r},{c}) tile {mdp.tile(s)}: {q_txt}  ->  V2 = {hist[2][s]:.4f}")

    status = "CONVERGED" if converged else "DID NOT CONVERGE (safeguard reached)"
    log(f"\nValue iteration: {status} after {iters} iterations, "
        f"final max|ΔV| = {delta:.3e} (tolerance {solver['tolerance']})")
    log("\nFinal values V*:")
    log(mdp.format_values(V, 3))

    policy, ties = mdp.greedy_policy(V)
    log("\nGreedy policy (ties broken in order up > right > down > left; "
        "a cell with several arrows = exact tie):")
    log(mdp.format_policy(policy, ties))
    log("\nQ-values at every nonterminal state (final V):")
    for s in mdp.transitions:
        qs = "  ".join(f"{ARROWS[a]}{mdp.q_value(V, s, a):8.3f}" for a in ACTIONS)
        log(f"  {one_based(s)} {mdp.tile(s)}: {qs}   best={ARROWS[policy[s]]}"
            + (f"  TIES={ties[s]}" if len(ties[s]) > 1 else ""))
    log(f"\nV*(start {one_based(mdp.start)}) = {V[mdp.start]:.4f}")

    # --- simulation ---------------------------------------------------------
    rows_out, summary = [], []
    example = {}
    for b in sim["batch_seeds"]:
        eps = []
        for i in range(sim["episodes_per_batch"]):
            seed = 1000 * b + i
            res = mdp.run_episode(policy, seed, sim["max_steps"])
            eps.append(res)
            rows_out.append(dict(map=name, batch_seed=b, episode_seed=seed,
                                 steps=res["steps"], total_reward=res["total_reward"],
                                 discounted_return=round(res["discounted_return"], 6),
                                 outcome=res["outcome"]))
            example.setdefault(res["outcome"], (seed, res))
        summary.append(dict(
            map=name, batch_seed=b,
            goals=sum(e["outcome"] == "goal" for e in eps),
            holes=sum(e["outcome"] == "hole" for e in eps),
            time_limits=sum(e["outcome"] == "time_limit" for e in eps),
            mean_steps=statistics.mean(e["steps"] for e in eps),
            mean_disc=statistics.mean(e["discounted_return"] for e in eps)))

    log("\nSimulation of the greedy policy (3 x 50 episodes, 100-step limit):")
    log("| Map | Batch seed | Goals /50 | Holes /50 | Time limits /50 | Mean steps | Mean discounted return |")
    log("| --- | --- | --- | --- | --- | --- | --- |")
    for s in summary:
        log(f"| {s['map']} | {s['batch_seed']} | {s['goals']} | {s['holes']} | "
            f"{s['time_limits']} | {s['mean_steps']:.2f} | {s['mean_disc']:.3f} |")
    allr = [r["discounted_return"] for r in rows_out]
    log(f"All 150 episodes: mean discounted return = {statistics.mean(allr):.4f} "
        f"(sd {statistics.stdev(allr):.3f})  vs  V*(start) = {V[mdp.start]:.4f}")
    for outcome, (seed, res) in example.items():
        path = " -> ".join(f"{one_based(s)}{ARROWS[a]}" for s, a, _, _ in res["trace"])
        log(f"Example {outcome} episode (seed {seed}): {path} -> {one_based(res['trace'][-1][2])}")
    for missing in {"goal", "hole", "time_limit"} - example.keys():
        log(f"Outcome '{missing}' never occurred in 150 episodes.")

    with open(OUT / f"partC_episodes_{name}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows_out[0].keys())
        w.writeheader()
        w.writerows(rows_out)
    return mdp, V, policy


def main():
    OUT.mkdir(exist_ok=True)
    cfg = load_config()
    rules = load_floor_rules(cfg)
    lines = []

    def log(msg=""):
        print(msg)
        lines.append(msg)

    paper = solve_map("paper", cfg["paper_map"], cfg, rules, log)
    ext = solve_map("extended", cfg["extended_map"], cfg, rules, log)

    log(f"\n{'=' * 70}\nCOMPARISON paper -> extended (same coordinates)")
    (m1, V1, p1), (m2, V2, p2) = paper, ext
    for s in m1.transitions:
        if s in m2.transitions:
            mark = "  <-- action changed" if p1[s] != p2[s] else ""
            log(f"  {one_based(s)} {m1.tile(s)}->{m2.tile(s)}: V {V1[s]:8.3f} -> {V2[s]:8.3f} "
                f"(Δ {V2[s] - V1[s]:+.3f}), action {ARROWS[p1[s]]} -> {ARROWS[p2[s]]}{mark}")

    (OUT / "partC_solver_output.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()

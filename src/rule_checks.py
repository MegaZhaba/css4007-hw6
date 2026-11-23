"""Sanity checks of the Grid World model before trusting the solver.

Run:  python src/rule_checks.py      (exit code 1 if any check fails)
Writes outputs/partC_rule_checks.txt
"""
import sys

from gridworld import ACTIONS, GridMDP, ROOT, load_config, load_floor_rules, one_based

cfg = load_config()
rules = load_floor_rules(cfg)
paper = GridMDP(cfg["paper_map"], rules, cfg["discount"])
ext = GridMDP(cfg["extended_map"], rules, cfg["discount"])
lines, failures = [], 0


def check(name, cond, detail=""):
    global failures
    failures += not cond
    lines.append(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))


def outs(mdp, rc, a):
    s = (rc[0] - 1, rc[1] - 1)
    return [(round(p, 10), one_based(n), r, d) for p, n, r, d in mdp.transitions[s][a]]


# 1. Wall and boundary behaviour
check("Boundary: (1,1) up stays at (1,1), reward -1",
      outs(paper, (1, 1), "up") == [(1.0, (1, 1), -1, False)], outs(paper, (1, 1), "up"))
check("Boundary: (3,1) left stays at (3,1)",
      outs(paper, (3, 1), "left") == [(1.0, (3, 1), -1, False)], outs(paper, (3, 1), "left"))
check("Wall: (2,1) right bumps into # and stays",
      outs(paper, (2, 1), "right") == [(1.0, (2, 1), -1, False)], outs(paper, (2, 1), "right"))
check("Wall in mud: (1,2) down stays in M and costs -3",
      outs(paper, (1, 2), "down") == [(1.0, (1, 2), -3, False)], outs(paper, (1, 2), "down"))
check("Wall is not a state", (1, 1) not in paper.states, "(2,2) 0-based=(1,1)")
ice_left = outs(paper, (2, 3), "left")
check("Ice left: 0.8 bump into wall (stay) + 0.1 up + 0.1 down",
      sorted(ice_left) == sorted([(0.8, (2, 3), -1, False), (0.1, (1, 3), -1, False),
                                  (0.1, (3, 3), -1, False)]), ice_left)
ice_up = outs(paper, (2, 3), "up")
check("Ice up: each slip resolved separately (left slip hits wall, right slip -> hole)",
      sorted(ice_up) == sorted([(0.8, (1, 3), -1, False), (0.1, (2, 3), -1, False),
                                (0.1, (2, 4), -10, True)]), ice_up)

# 2. Probabilities sum to 1 everywhere (ice in particular)
for mdp, nm in ((paper, "paper"), (ext, "extended")):
    sums = [sum(p for p, *_ in mdp.transitions[s][a]) for s in mdp.transitions for a in ACTIONS]
    check(f"{nm}: all P(s'|s,a) sum to 1", all(abs(x - 1) < 1e-12 for x in sums),
          f"min={min(sums)}, max={max(sums)}")
for a in ACTIONS:
    o = outs(paper, (2, 3), a)
    check(f"Ice {a}: probabilities sum to 1", abs(sum(p for p, *_ in o) - 1) < 1e-12, o)
check("Deterministic tiles give a single outcome with p=1",
      all(len(paper.transitions[s][a]) == 1 for s in paper.transitions
          if paper.tile(s) != "I" for a in ACTIONS))

# 3. Floor rewards
check("Entering mud M costs -3: (1,1) right", outs(paper, (1, 1), "right")[0][2] == -3)
check("Entering sand D costs -2: extended (3,1) right",
      outs(ext, (3, 1), "right") == [(1.0, (3, 2), -2, False)], outs(ext, (3, 1), "right"))
check("Entering water W costs -5: extended (3,3) right",
      outs(ext, (3, 3), "right") == [(1.0, (3, 4), -5, False)], outs(ext, (3, 3), "right"))
check("Bumping inside water costs -5 again: extended (3,4) down",
      outs(ext, (3, 4), "down") == [(1.0, (3, 4), -5, False)], outs(ext, (3, 4), "down"))
check("Sand and water are deterministic",
      all(len(ext.transitions[(2, c)][a]) == 1 for c in (1, 3) for a in ACTIONS))
check("Entering G gives exactly +10 (no extra floor cost)",
      outs(paper, (1, 3), "right") == [(1.0, (1, 4), 10, True)], outs(paper, (1, 3), "right"))
check("Entering H gives -10", outs(paper, (3, 4), "up") == [(1.0, (2, 4), -10, True)])

# 4. Terminal handling
V, *_ = paper.value_iteration(1e-6, 10_000)
check("Terminal states have no actions/transitions",
      (0, 3) not in paper.transitions and (1, 3) not in paper.transitions)
check("V(G) = V(H) = 0 after convergence", V[(0, 3)] == 0 and V[(1, 3)] == 0)
pol = {s: "right" for s in paper.transitions}
pol[(2, 0)] = "up"; pol[(1, 0)] = "up"; pol[(0, 0)] = "right"; pol[(0, 1)] = "right"
ep = paper.run_episode(pol, seed=0)
check("Simulator stops at goal; last reward +10, nothing after",
      ep["outcome"] == "goal" and ep["trace"][-1][3] == 10 and ep["steps"] == len(ep["trace"]),
      f"steps={ep['steps']}, rewards={[t[3] for t in ep['trace']]}")
# all "right" leads S->(3,4); at (3,4) we force "up" into the hole
ep_hole = paper.run_episode({s: "right" for s in paper.transitions} | {(2, 3): "up"}, seed=0)
check("Simulator stops at hole; last reward -10",
      ep_hole["outcome"] == "hole" and ep_hole["trace"][-1][3] == -10,
      f"rewards={[t[3] for t in ep_hole['trace']]}")

# 5. Unknown floor letter is rejected (floors are data, not code)
try:
    GridMDP(["SX"], rules, 0.9)
    check("Unknown tile letter is rejected", False)
except ValueError as e:
    check("Unknown tile letter is rejected", True, str(e))

text = "\n".join(lines) + f"\n\n{len(lines) - failures}/{len(lines)} checks passed\n"
print(text)
(ROOT / "outputs").mkdir(exist_ok=True)
(ROOT / "outputs" / "partC_rule_checks.txt").write_text(text)
sys.exit(1 if failures else 0)

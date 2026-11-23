# HW6 submission

## AI-use declaration

I used **Claude Code (Anthropic)** as an assistant for this homework:

- generating and checking the Part A hand calculations (I re-did them by hand on paper; the photo is mine);
- writing the Python code in `src/` (value iteration, simulators, rule checks, FrozenLake runner);
- drafting the explanations in this file, which I reviewed against the printed outputs.

All numbers in this file come from the programs' own outputs in `outputs/`.

## Setup and reproduction

Versions actually used: **Python 3.13.18, gymnasium 1.4.0, numpy 2.5.5** (full list in `requirements.txt`).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

```bash
python src/rule_checks.py        # Part C model checks  -> outputs/partC_rule_checks.txt
python src/solve_gridworld.py    # Part C solver + sims -> outputs/partC_solver_output.txt, partC_episodes_*.csv
python src/frozenlake_sim.py     # Part B               -> outputs/partB_episodes.csv, partB_summary.md, partB_traces.txt
```

Code layout:

| File | Purpose |
| --- | --- |
| `data/maps.json` | maps **and floor rules** (reward, terminal, slippery, traversable) |
| `src/gridworld.py` | generic `GridMDP`: model built from floor rules, value iteration, greedy policy, simulator |
| `src/solve_gridworld.py` | Part C runs on both maps, comparison |
| `src/rule_checks.py` | 26 checks of walls, slip probabilities, rewards, termination |
| `src/frozenlake_sim.py` | Part B, Gymnasium FrozenLake runner |

## Part A — Paper calculations

Scan: [`part_a/part_a_scan.jpg`](part_a/part_a_scan.jpg). Typed copy with every step: [`part_a/PART_A.md`](part_a/PART_A.md).
Part A was committed (commit "Part A: paper calculations") before any code.

**MDP.** States = 9 nonterminal tiles + terminal G, H (the wall is not a state). Actions = up/right/down/left.
Transitions deterministic except when *leaving* ice (0.8 intended, 0.1 each perpendicular); moves into
a wall/edge stay in place. Reward = reward of the tile where the move ends (−1 floor, −3 mud, +10 goal, −10 hole).
The agent maximizes expected discounted return `Σ 0.9^t r_t`.

**V1** (best one-step expected reward):

```text
|  -1  |  -1  |  10  |  G  |
|  -1  |  ##  |  -1  |  H  |
|  -1  |  -1  |  -1  |  -1 |
```

At the ice tile: up −1.9, right −8.2, down −1.9, **left −1.0** (bump into the wall, no hole risk).

**V2** (from V1):

| Position | up | right | down | left | V2 | arrow |
| --- | --- | --- | --- | --- | --- | --- |
| (1,2) M | −3+0.9·(−1) = −3.9 | −1+0.9·10 = **8.0** | −3.9 | −1+0.9·(−1) = −1.9 | **8.0** | → |
| (1,3) | −1+0.9·10 = 8.0 | 10+0 = **10.0** | −1.9 | −3.9 | **10.0** | → |
| (2,3) I | 0.8·8 + 0.1·(−1.9) + 0.1·(−10) = **5.21** | 0.8·(−10)+0.1·8+0.1·(−1.9) = −7.39 | 0.8·(−1.9)+0.1·(−1.9)+0.1·(−10) = −2.71 | 0.8·(−1.9)+0.1·8+0.1·(−1.9) = −0.91 | **5.21** | ↑ |

No ties at these three positions. Two rounds are not enough: V2 is the optimum for a 2-step horizon only —
the +10 has propagated just two tiles, the bottom row still "doesn't see" the goal, and values keep
changing in later rounds (they converge only in the limit), so the greedy policy can still change.

## Part B — FrozenLake

`FrozenLake-v1`, `map_name="4x4"`, default rewards (+1 for goal, 0 otherwise), `max_episode_steps=100`.
Episode `i` of batch `b` uses `env.reset(seed=1000·b+i)`; the random policy's action generator is
`numpy.random.default_rng(1000·b+i)`, so all four conditions use the same seed lists.
Outcome: `terminated` and final state 15 → goal; `terminated` elsewhere → hole; `truncated` → time limit
(a termination on step 100 is recorded as its terminal outcome).

### Fixed policy, written before the experiments

```text
 S  F  F  F        ↓  ←  ↓  ←         states  0  1  2  3
 F  H  F  H        ↓  H  ↓  H                 4  5  6  7
 F  F  F  H        →  ↓  ↓  H                 8  9 10 11
 H  F  F  G        H  →  →  G                12 13 14 15
```

Main route `0↓4↓8→9↓13→14→15` (6 moves). Each state's action is chosen so that the *intended* move never
enters a hole (e.g. 4: down, because right = hole 5; 8: right, because down = hole 12; 10: down, because
right = hole 11; 3: left, because down = hole 7). Off-route states (1, 2, 3, 6, 10) lead back onto a safe column.
The table is in `FIXED_POLICY` in `src/frozenlake_sim.py` and is identical for all experiments.

### Results

| Slippery? | Policy | Batch seed | Goals /50 | Holes /50 | Time limits /50 | Success % | Mean steps |
| --- | --- | --- | --- | --- | --- | --- | --- |
| False | random | 11 | 0 | 50 | 0 | 0% | 7.98 |
| False | random | 22 | 0 | 50 | 0 | 0% | 6.42 |
| False | random | 33 | 1 | 49 | 0 | 2% | 6.66 |
| False | fixed | 11 | 50 | 0 | 0 | 100% | 6.00 |
| False | fixed | 22 | 50 | 0 | 0 | 100% | 6.00 |
| False | fixed | 33 | 50 | 0 | 0 | 100% | 6.00 |
| True | random | 11 | 0 | 50 | 0 | 0% | 8.30 |
| True | random | 22 | 2 | 48 | 0 | 4% | 7.76 |
| True | random | 33 | 0 | 50 | 0 | 0% | 6.74 |
| True | fixed | 11 | 1 | 49 | 0 | 2% | 5.22 |
| True | fixed | 22 | 2 | 48 | 0 | 4% | 6.00 |
| True | fixed | 33 | 1 | 49 | 0 | 2% | 5.14 |

Files: [`outputs/partB_episodes.csv`](outputs/partB_episodes.csv) (600 rows, one per episode),
[`outputs/partB_summary.md`](outputs/partB_summary.md),
[`outputs/partB_traces.txt`](outputs/partB_traces.txt) (two text-rendered episodes + example traces).

Example traces (state –action→ next state):

- success, slippery fixed, seed 11020 (14 steps): `0 -Down-> 4 -Down-> 4 -Down-> 8 -Right-> 9 -Down-> 13 -Right-> 13 -Right-> 9 -Down-> 10 -Down-> 9 -Down-> 10 -Down-> 14 -Right-> 10 -Down-> 14 -Right-> 15`
- failure, slippery fixed, seed 11000 (8 steps): `0 -Down-> 1 -Left-> 1 -Left-> 1 -Left-> 1 -Left-> 0 -Down-> 4 -Down-> 4 -Down-> 5` (slipped right into hole 5)
- success, non-slippery random, seed 33006: `0 -Left-> 0 -Down-> 4 -Down-> 8 -Right-> 9 -Down-> 13 -Right-> 14 -Right-> 15`
- A **time limit never occurred** in any of the 600 episodes; a **hole never occurred** for the fixed policy without slipping.

**What slipping changed.** Without slipping the environment is deterministic: the fixed policy follows the
same 6-step path from every seed (150/150 goals, exactly 6.00 steps), the seed only matters for the random
policy. With `is_slippery=True` Gymnasium moves in the intended direction with only 1/3 probability and
in each perpendicular direction with 1/3. My policy was designed so that the *intended* move is safe, but
many of its states have a hole *perpendicular* to the move — e.g. state 4 "down" slips right into hole 5
with probability 1/3 (`env.P[4][1]`). Success collapsed from 100% to 4/150 ≈ 2.7%. From the model I
computed that this policy's exact success probability within 100 steps is ≈ 4.2%, consistent with the sample.
A policy that is good under slipping must look different (e.g. pressing *against* walls so that slips are
harmless) — which is exactly what value iteration finds in Part C for our ice tile.
The random policy is bad in both cases (3/300 goals overall): most random walks hit a hole within ~7 steps.

**Consistency across batches.** Deterministic conditions are identical across batches; the stochastic ones
vary (fixed-slippery 1, 2, 1 goals; random 0–2 goals). With 50 episodes and p ≈ 0.03, the standard error of
the success rate is about √(0.03·0.97/50) ≈ 2.4 percentage points — the same size as the estimate itself, so
batches of 50 can only tell "almost never" from "often", not 2% from 4%.

**Why one success is not evidence.** One success shows that the goal is *reachable*, not that the policy is
*reliable*: the random policy also reached the goal (seed 33006) while its success rate is ≈ 1%. Performance
is an expectation over the environment's randomness, so it must be estimated from many seeded episodes.

**Why this is not training.** Both policies are fixed tables/rules written before running. Nothing in the
runner reads the rewards to change the policy; episodes only *evaluate* it (Monte-Carlo policy evaluation).
Training (Q-learning, policy gradient …) would update action choices from experienced rewards.

## Part C — Custom solver

The model is built from `data/maps.json`: each floor has `reward`, `terminal`, `traversable`, and for
slippery floors `intended` / `perpendicular_each`. `GridMDP` precomputes `P(s'|s,a)` from those rules, so
adding sand `D` and water `W` required **no** solver change — only their JSON entries (an unknown
letter is rejected with an error). Full output: [`outputs/partC_solver_output.txt`](outputs/partC_solver_output.txt).

### Compare code with paper

```text
V1:
| -1.000 | -1.000 | 10.000 |  G 0   |
| -1.000 |  ####  | -1.000 |  H 0   |
| -1.000 | -1.000 | -1.000 | -1.000 |

V2 at the requested positions (with all four Q-values from V1):
  (1,2) tile M: up=-3.9000, right=8.0000, down=-3.9000, left=-1.9000  ->  V2 = 8.0000
  (1,3) tile .: up=8.0000, right=10.0000, down=-1.9000, left=-3.9000  ->  V2 = 10.0000
  (2,3) tile I: up=5.2100, right=-7.3900, down=-2.7100, left=-0.9100  ->  V2 = 5.2100
```

All values match the paper exactly, including every individual Q-value — no discrepancies to correct.
(Points to watch that I checked by hand: a bump inside mud costs −3 not −1; at the ice tile the left
slip of "up"/"down" hits the wall and stays *on the ice*.)

### Values, policies and convergence

Synchronous value iteration (each sweep uses only the previous round's V), stop when `max|ΔV| < 1e-6`,
safeguard 10 000 iterations.

| Map | Iterations | Final max\|ΔV\| | Converged? | V*(start) |
| --- | --- | --- | --- | --- |
| paper `.M.G/.#IH/S...` | 12 | 2.165e-07 | yes | 1.5358 |
| extended `.M.G/.#IH/SD.W` | 11 | 2.406e-07 | yes | 1.5020 |

```text
PAPER — V*                                   PAPER — policy
| 4.200 | 8.000 | 10.000 |  G 0  |           |  →  | M→ |  →  | G |
| 2.780 | ####  | 5.824  |  H 0  |           |  ↑  | ## | I↑  | H |
| 1.536 | 2.818 | 4.242  | 2.818 |           |  →  |  → |  ↑  | ← |

EXTENDED — V*                                EXTENDED — policy
| 4.200 | 8.000 | 10.000 |  G 0  |           |  →  | M→ |  →  | G |
| 2.780 | ####  | 5.824  |  H 0  |           |  ↑  | ## | I↑  | H |
| 1.502 | 2.818 | 4.242  | 2.818 |           |  ↑  | D→ |  ↑  | W← |
```

**Tie-breaking:** the policy takes the arg-max of `Q(s,a)` computed from the final V; actions within 1e-9 of
the best count as tied and the first in the order **up > right > down > left** is chosen. All tied actions
are printed; with the converged values there are **no exact ties** on either map (the closest call is the
start on the paper map: → 1.536 vs ↑ 1.502).

### Simulation results

Custom simulator (`GridMDP.run_episode`), episode seed `1000·b+i` → `numpy.random.default_rng(seed)` for the ice
slips, start at S, 100-step limit, discounted return `r0 + 0.9 r1 + 0.9² r2 + …`.

| Map | Batch seed | Goals /50 | Holes /50 | Time limits /50 | Mean steps | Mean discounted return |
| --- | --- | --- | --- | --- | --- | --- |
| paper | 11 | 47 | 3 | 0 | 5.04 | 2.230 |
| paper | 22 | 44 | 6 | 0 | 4.96 | 1.442 |
| paper | 33 | 42 | 8 | 0 | 4.88 | 0.970 |
| extended | 11 | 50 | 0 | 0 | 5.00 | 1.502 |
| extended | 22 | 50 | 0 | 0 | 5.00 | 1.502 |
| extended | 33 | 50 | 0 | 0 | 5.00 | 1.502 |

Per-episode rows: [`outputs/partC_episodes_paper.csv`](outputs/partC_episodes_paper.csv),
[`outputs/partC_episodes_extended.csv`](outputs/partC_episodes_extended.csv).
Example paper-map hole (seed 11019): `(3,1)→ (3,2)→ (3,3)↑ (2,3)↑ slip-left-stays (2,3)↑ slip-right → H`.

### Explain your results

**What changed with sand and water.** Only one action changed: the start (3,1) goes **→ on the paper map but
↑ on the extended map**. On the paper map the bottom route `S→.→.→ice↑→.→G` (V = 1.536) beats the left/top
route `S↑ .↑ .→ M→ .→ G` (V = 1.502) by only 0.034 — it is shorter in cost but passes over the ice and risks
the hole. Sand at (3,2) makes the first step of the bottom route cost −2 instead of −1, so it drops to
`Q(S,→) = 0.536`, and the safe route through mud wins: V(S) 1.536 → 1.502. Every other value is unchanged
because the optimal paths from those states never enter D or W: e.g. (3,2) is now sand but its *value*
only depends on where you go *from* it, and (3,4) is now water but the best move leaves it to the left.
Water mattered only as a cost *of entering*: `Q((3,3),→)` fell from 1.536 to −2.464, and staying in water
costs −5 per bump (`Q(W,↓)= −2.464`).

**Cost and risk.** The two start routes trade cost against risk: the top route pays mud (−3) but is deterministic;
the bottom route is cheaper per step but crosses the ice with a 10% hole risk per attempt. On the paper map the
risk is (barely) worth it; adding one extra cost unit of sand flips the decision to the safe route. On the ice the
policy goes ↑ (5.824), not ← (4.475): the 0.1 chance of the right-slip into H is accepted because the 0.8
chance of reaching (1,3) is worth more. The left-slip into the wall merely keeps the agent on the ice, so
repeating ↑ is fine. "Right" from ice is terrible (−6.918). From (3,4) the policy never presses ↑ (→ H, −10).

**Simulation vs. values.** V*(S) is an *expectation over infinitely many* infinite-horizon episodes.
The simulation is a finite sample, truncated at 100 steps (here irrelevant — the longest episode had 6 steps,
and 0.9^100 ≈ 3e-5 bounds the truncation error anyway). For the extended map the policy never touches ice,
so every episode is identical and the sample mean equals V*(S) = 1.502 exactly (sd 0).
For the paper map an episode returns one of only three values — 3.122 (goal in 5 steps, 123 episodes), 1.810 (goal after one
slip-into-wall, 6 steps, 10 episodes) or −10.0 (hole, 17 episodes) — so a batch mean of 50 is noisy:
2.230, 1.442, 0.970 per batch, but the mean over all 150 episodes is **1.547 vs V*(S)=1.536**. The sample agrees with
the value only on average, with standard error ≈ 4.16/√150 ≈ 0.34. Note also that the paper map's policy
has *higher* expected return but a 6–16% hole rate per batch, while the extended policy has a slightly lower
expectation and zero variance — value iteration optimizes expectation, not risk.

### Rule checks

[`outputs/partC_rule_checks.txt`](outputs/partC_rule_checks.txt) — **26/26 PASS**, including:

- boundary: `(1,1) up → stay (1,1), −1`; wall: `(2,1) right → stay, −1`; bump inside mud: `(1,2) down → stay, −3`;
- ice "up": `0.8 → (1,3)`, `0.1 → left slip hits wall, stays on ice`, `0.1 → H (−10, terminal)`; all ice actions sum to 1; every `P(·|s,a)` on both maps sums to 1;
- rewards: enter M −3, enter D −2, enter W −5 (and bump in W −5), enter G exactly +10, enter H −10;
- termination: G/H have no actions, V(G)=V(H)=0, the simulator stops on G/H with the +10/−10 as the last reward;
- an unknown floor letter is rejected (floors are data, not code).

# Part A — calculations on paper (typed copy)

> The handwritten photo/scan goes next to this file as `part_a/part_a_scan.jpg`.
> This typed copy contains the same calculations.

Coordinates are `(row, column)`, starting at 1. Discount γ = 0.9, V0 = 0.

```text
             Column
          1   2   3   4
Row 1   | . | M | . | G |
Row 2   | . | # | I | H |
Row 3   | S | . | . | . |
```

## 1. The MDP in my own words

- **States**: every non-wall tile the agent can stand on. Nine of them are nonterminal:
  (1,1), (1,2)=M, (1,3), (2,1), (2,3)=I, (3,1)=S, (3,2), (3,3), (3,4).
  Two are terminal: (1,4)=G and (2,4)=H; their value is fixed at 0. The wall (2,2) is not a state.
- **Actions**: up, right, down, left in every nonterminal state.
- **Transitions** `P(s'|s,a)`: from S, `.` and M the chosen move happens with probability 1.
  From I the chosen move happens with 0.8 and each perpendicular move with 0.1.
  A move into the wall or off the map leaves the agent where it is (checked separately for every slip direction).
- **Rewards** `R(s,a,s')`: the reward of the tile where the move *ends*: `.`/S/I = −1, M = −3, G = +10, H = −10.
  A bumped move earns the reward of the tile you stayed on (e.g. bumping inside mud costs −3).
- **Goal of the agent**: maximize the expected discounted sum of rewards
  `E[r0 + 0.9·r1 + 0.9²·r2 + …]` — i.e. reach G (+10) quickly, avoid H (−10), avoid mud.

## 2. V1 for every nonterminal tile

With V0 = 0 everywhere, `V1(s) = max_a Σ P(s'|s,a)·R(s,a,s')` — the best expected immediate reward.

| State | up | right | down | left | **V1** |
| --- | --- | --- | --- | --- | --- |
| (1,1) `.` | stay −1 | M −3 | (2,1) −1 | stay −1 | **−1** |
| (1,2) M | stay in M −3 | (1,3) −1 | wall, stay in M −3 | (1,1) −1 | **−1** |
| (1,3) `.` | stay −1 | G +10 | ice −1 | M −3 | **+10** |
| (2,1) `.` | (1,1) −1 | wall, stay −1 | S −1 | stay −1 | **−1** |
| (3,1) S | (2,1) −1 | (3,2) −1 | stay −1 | stay −1 | **−1** |
| (3,2) `.` | wall, stay −1 | (3,3) −1 | stay −1 | S −1 | **−1** |
| (3,3) `.` | ice −1 | (3,4) −1 | stay −1 | (3,2) −1 | **−1** |
| (3,4) `.` | H −10 | stay −1 | stay −1 | (3,3) −1 | **−1** |

Ice (2,3) — probability-weighted:

| Action | 0.8 intended | 0.1 perpendicular | 0.1 perpendicular | Sum |
| --- | --- | --- | --- | --- |
| up | (1,3): 0.8·(−1) | left→wall, stay on I: 0.1·(−1) | right→H: 0.1·(−10) | **−1.9** |
| right | H: 0.8·(−10) | up (1,3): 0.1·(−1) | down (3,3): 0.1·(−1) | **−8.2** |
| down | (3,3): 0.8·(−1) | left→wall, stay: 0.1·(−1) | right→H: 0.1·(−10) | **−1.9** |
| left | wall, stay: 0.8·(−1) | up (1,3): 0.1·(−1) | down (3,3): 0.1·(−1) | **−1.0** |

V1(2,3) = **−1** (left: bump into the wall on purpose; it is the only action without a hole risk).

**V1 map**

```text
+------+------+------+------+
|  -1  |  -1  |  10  |  G 0 |
+------+------+------+------+
|  -1  |  ##  |  -1  |  H 0 |
+------+------+------+------+
|  -1  |  -1  |  -1  |  -1  |
+------+------+------+------+
```

## 3. V2 at (1,2), (1,3), (2,3) using only V1

`Q2(s,a) = Σ P(s'|s,a) · [R + 0.9·V1(s')]`

**(1,2) — mud**

| a | result | calculation | Q |
| --- | --- | --- | --- |
| up | wall/boundary, stay in M | −3 + 0.9·(−1) | −3.9 |
| right | (1,3) | −1 + 0.9·10 | **8.0** |
| down | wall, stay in M | −3 + 0.9·(−1) | −3.9 |
| left | (1,1) | −1 + 0.9·(−1) | −1.9 |

V2(1,2) = **8.0**, best action **→ right**.

**(1,3)**

| a | result | calculation | Q |
| --- | --- | --- | --- |
| up | boundary, stay | −1 + 0.9·10 | 8.0 |
| right | G (terminal) | +10 + 0.9·0 | **10.0** |
| down | I (2,3) | −1 + 0.9·(−1) | −1.9 |
| left | M (1,2) | −3 + 0.9·(−1) | −3.9 |

V2(1,3) = **10.0**, best action **→ right**.

**(2,3) — ice (probability-weighted)**

| a | 0.8 × intended | 0.1 × perpendicular | 0.1 × perpendicular | Q |
| --- | --- | --- | --- | --- |
| up | (1,3): 0.8·(−1+0.9·10) = 6.4 | left→wall, stay I: 0.1·(−1+0.9·(−1)) = −0.19 | right→H: 0.1·(−10+0) = −1.0 | **5.21** |
| right | H: 0.8·(−10) = −8.0 | up (1,3): 0.1·8 = 0.8 | down (3,3): 0.1·(−1.9) = −0.19 | −7.39 |
| down | (3,3): 0.8·(−1.9) = −1.52 | left→wall, stay: −0.19 | right→H: −1.0 | −2.71 |
| left | wall, stay: 0.8·(−1.9) = −1.52 | up (1,3): 0.8 | down (3,3): −0.19 | −0.91 |

V2(2,3) = **5.21**, best action **↑ up**.

## 4. Best-action arrows after two rounds

```text
+------+------+------+------+
|      |  →   |  →   |  G   |
+------+------+------+------+
|      |  ##  |  ↑   |  H   |
+------+------+------+------+
|      |      |      |      |
+------+------+------+------+
```

No ties at these three positions (for V1 there were many ties, e.g. all four actions at S give −1).

**Why two rounds do not prove optimality.** `Vk` is the optimal value for a horizon of only *k* steps.
After two rounds the +10 of the goal has propagated only two tiles away; states like S, (3,2), (3,3)
still "do not know" the goal exists (their V2 is still about −1.9), so their greedy actions are arbitrary ties.
Even at the three positions above, values keep changing (e.g. the ice tile's slip into (3,3) is
still valued at −1.9 instead of its true value). Value iteration converges to V* only in the limit,
so we have to iterate until the change is tiny (Part C) and only then extract the policy.

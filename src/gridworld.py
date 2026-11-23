"""Grid World MDP for HW6 Part C.

The floor rules (reward, terminal, slippery, traversable) live in data/maps.json.
This module only *reads* them, so adding a new floor letter = adding one JSON entry;
the value-iteration code below never mentions concrete letters.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "data" / "maps.json"

ACTIONS = ["up", "right", "down", "left"]          # also the tie-breaking order
ARROWS = {"up": "↑", "right": "→", "down": "↓", "left": "←"}
DELTA = {"up": (-1, 0), "right": (0, 1), "down": (1, 0), "left": (0, -1)}
PERPENDICULAR = {"up": ("left", "right"), "down": ("left", "right"),
                 "left": ("up", "down"), "right": ("up", "down")}


def load_config(path: Path = CONFIG_PATH) -> dict:
    return json.loads(path.read_text())


@dataclass(frozen=True)
class FloorRule:
    symbol: str
    meaning: str
    reward: float | None
    terminal: bool
    traversable: bool
    intended: float          # probability of moving in the chosen direction
    perpendicular_each: float

    @staticmethod
    def from_json(symbol: str, d: dict) -> "FloorRule":
        slippery = d.get("movement") == "slippery"
        return FloorRule(
            symbol=symbol,
            meaning=d["meaning"],
            reward=d.get("reward"),
            terminal=d.get("terminal", False),
            traversable=d.get("traversable", True),
            intended=d.get("intended", 1.0) if slippery else 1.0,
            perpendicular_each=d.get("perpendicular_each", 0.0) if slippery else 0.0,
        )


def load_floor_rules(config: dict) -> dict[str, FloorRule]:
    return {sym: FloorRule.from_json(sym, d) for sym, d in config["tiles"].items()}


class GridMDP:
    """A finite MDP built from a list of row strings + floor rules.

    State = (row, col) 0-based internally; printed 1-based (as in the assignment).
    """

    def __init__(self, rows: list[str], rules: dict[str, FloorRule], gamma: float):
        self.grid = [list(r) for r in rows]
        self.n_rows, self.n_cols = len(rows), len(rows[0])
        self.rules = rules
        self.gamma = gamma
        unknown = {ch for r in rows for ch in r} - rules.keys()
        if unknown:
            raise ValueError(f"no floor rule for tiles {unknown}")
        self.states = [(r, c) for r in range(self.n_rows) for c in range(self.n_cols)
                       if rules[self.grid[r][c]].traversable]
        self.start = next(s for s in self.states if self.tile(s) == "S")
        # Pre-compute the full model: transitions[s][a] = [(prob, s', reward, done), ...]
        self.transitions = {s: {a: self._outcomes(s, a) for a in ACTIONS}
                            for s in self.states if not self.is_terminal(s)}

    # --- tile helpers -------------------------------------------------------
    def tile(self, s) -> str:
        return self.grid[s[0]][s[1]]

    def rule(self, s) -> FloorRule:
        return self.rules[self.tile(s)]

    def is_terminal(self, s) -> bool:
        return self.rule(s).terminal

    # --- model --------------------------------------------------------------
    def move(self, s, direction):
        """Deterministic result of one direction: walls/boundaries => stay."""
        dr, dc = DELTA[direction]
        r, c = s[0] + dr, s[1] + dc
        if 0 <= r < self.n_rows and 0 <= c < self.n_cols \
                and self.rules[self.grid[r][c]].traversable:
            return (r, c)
        return s

    def _outcomes(self, s, a):
        rule = self.rule(s)                         # slipping depends on the tile we LEAVE
        dist = [(rule.intended, a)]
        if rule.perpendicular_each > 0:
            dist += [(rule.perpendicular_each, p) for p in PERPENDICULAR[a]]
        merged: dict[tuple, float] = {}
        for p, d in dist:                           # resolve EACH slip direction separately
            nxt = self.move(s, d)
            merged[nxt] = merged.get(nxt, 0.0) + p
        return [(p, nxt, self.rule(nxt).reward, self.is_terminal(nxt))
                for nxt, p in merged.items()]

    def q_value(self, V: dict, s, a) -> float:
        return sum(p * (r + self.gamma * (0.0 if done else V[nxt]))
                   for p, nxt, r, done in self.transitions[s][a])

    # --- value iteration ----------------------------------------------------
    def bellman_update(self, V: dict) -> dict:
        """One synchronous sweep: uses ONLY the previous round's V."""
        new = dict(V)
        for s in self.transitions:
            new[s] = max(self.q_value(V, s, a) for a in ACTIONS)
        return new

    def value_iteration(self, tol=1e-6, max_iter=10_000):
        V = {s: 0.0 for s in self.states}
        history = [V]
        delta = float("inf")
        for k in range(1, max_iter + 1):
            new = self.bellman_update(V)
            delta = max(abs(new[s] - V[s]) for s in self.states)
            V = new
            history.append(V)
            if delta < tol:
                return V, k, delta, True, history
        return V, max_iter, delta, False, history

    def greedy_policy(self, V: dict, tie_eps=1e-9):
        """Best action per state; ties broken by ACTIONS order (up, right, down, left)."""
        policy, ties = {}, {}
        for s in self.transitions:
            qs = {a: self.q_value(V, s, a) for a in ACTIONS}
            best = max(qs.values())
            tied = [a for a in ACTIONS if qs[a] >= best - tie_eps]
            policy[s], ties[s] = tied[0], tied
        return policy, ties

    # --- simulation ---------------------------------------------------------
    def run_episode(self, policy: dict, seed: int, max_steps=100):
        rng = np.random.default_rng(seed)
        s, total, disc_return, trace = self.start, 0.0, 0.0, []
        for t in range(max_steps):
            a = policy[s]
            outs = self.transitions[s][a]
            i = rng.choice(len(outs), p=[o[0] for o in outs])
            _, nxt, r, done = outs[i]
            total += r
            disc_return += (self.gamma ** t) * r
            trace.append((s, a, nxt, r))
            s = nxt
            if done:
                outcome = "goal" if self.tile(s) == "G" else "hole"
                return dict(steps=t + 1, total_reward=total, discounted_return=disc_return,
                            outcome=outcome, trace=trace)
        return dict(steps=max_steps, total_reward=total, discounted_return=disc_return,
                    outcome="time_limit", trace=trace)

    # --- pretty printing (1-based coordinates) -------------------------------
    def _table(self, cell):
        sep = "+" + "+".join(["--------"] * self.n_cols) + "+"
        lines = [sep]
        for r in range(self.n_rows):
            lines.append("|" + "|".join(f"{cell((r, c)):^8}" for c in range(self.n_cols)) + "|")
            lines.append(sep)
        return "\n".join(lines)

    def format_values(self, V, digits=3):
        def cell(s):
            t = self.grid[s[0]][s[1]]
            if not self.rules[t].traversable:
                return "####"
            if self.rules[t].terminal:
                return f"{t} 0"
            return f"{V[s]:.{digits}f}"
        return self._table(cell)

    def format_policy(self, policy, ties=None):
        def cell(s):
            t = self.grid[s[0]][s[1]]
            if not self.rules[t].traversable:
                return "####"
            if self.rules[t].terminal:
                return t
            if ties and len(ties[s]) > 1:
                return "".join(ARROWS[a] for a in ties[s])
            return f"{t}{ARROWS[policy[s]]}" if t not in ".S" else ARROWS[policy[s]]
        return self._table(cell)


def one_based(s):
    return (s[0] + 1, s[1] + 1)

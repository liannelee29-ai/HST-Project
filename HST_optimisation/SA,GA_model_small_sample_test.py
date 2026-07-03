"""
benchmark_performance.py
========================
Section 6 Performance Benchmark — HST Project

Runs a head-to-head comparison of:
    1. Gurobi exact solver  (small instance, proves optimality)
    2. Genetic Algorithm    (same instance, multiple epsilon values)
    3. Simulated Annealing  (same instance, multiple epsilon values)

Also runs the SA operator comparison experiment required for Section 5.5.

Usage:
    python benchmark_performance.py

All parameters are defined in the CONFIG block below.
Tweak them without touching the logic.
"""

import math
import random
import statistics
import sys
import time
import warnings
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")          # non-interactive backend — safe for all environments
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── local imports ─────────────────────────────────────────────────────────────
# Adjust these paths if your folder structure differs
sys.path.insert(0, str(Path(__file__).resolve().parent))

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG  — tweak here, don't touch the logic below
# ══════════════════════════════════════════════════════════════════════════════
CFG = dict(
    # ── synthetic instance scenarios (shared, fixed seeds) ────────────────
    # Compare methods on both easy and harder instances.
    SCENARIOS = [
        {
            "name": "small",
            "num_stops": 10,
            "num_services": 3,
            "max_stops_per_svc": 4,
            "data_seed": 7,
        },
        {
            "name": "large",
            "num_stops": 40,
            "num_services": 6,
            "max_stops_per_svc": 8,
            "data_seed": 7,
        },
    ],
    MAX_DIST_PER_SVC    = None,     # set a float (km) to cap route length, else None

    # ── epsilon-constraint sweep ─────────────────────────────────────────────
    # Fractions of total demand used as demand-loss upper bounds.
    # Gurobi, GA and SA are all run at each of these points.
    EPSILON_FRACTIONS   = [0.60, 0.45, 0.30, 0.15, 0.0],

    # ── Gurobi ──────────────────────────────────────────────────────────────
    GUROBI_TIME_LIMIT   = 120,      # seconds per epsilon solve

    # ── SA ──────────────────────────────────────────────────────────────────
    SA_T_INIT           = 500.0,
    SA_T_FINAL          = 1e-3,
    SA_COOLING          = 0.995,
    SA_ITERS_PER_TEMP   = 200,
    SA_PENALTY_WEIGHT   = 10_000.0,
    # Same fixed random seeds for SA and GA so comparisons are reproducible.
    METHOD_SEEDS        = [42, 43, 44, 45, 46],

    # ── GA ───────────────────────────────────────────────────────────────────
    GA_POP_SIZE         = 80,
    GA_ELITE            = 6,
    GA_TOURNAMENT       = 4,
    GA_CROSSOVER_RATE   = 0.85,
    GA_MUTATION_RATE    = 0.35,
    GA_MAX_GENERATIONS  = 300,
    GA_TIME_LIMIT       = 30.0,     # seconds per epsilon solve
    GA_PENALTY_WEIGHT   = 10_000.0,
    # ── operator comparison (SA only) ────────────────────────────────────────
    OP_EPSILON_FRACTION = 0.30,     # single epsilon used for operator comparison
    OP_REPEATS          = 5,        # independent runs per operator set

    # ── output ───────────────────────────────────────────────────────────────
    OUTPUT_DIR          = Path("benchmark_output"),
    FIGURE_DPI          = 150,
)
# ══════════════════════════════════════════════════════════════════════════════


# ── helpers ───────────────────────────────────────────────────────────────────

def generate_data(num_stops: int, num_services: int, seed: int):
    """Synthetic benchmark instance — identical structure to Gurobi generate_data."""
    rng = random.Random(seed)
    depot = 0
    stops = list(range(1, num_stops + 1))
    nodes = [depot] + stops
    services = list(range(num_services))

    coords: Dict[int, Tuple[float, float]] = {depot: (5.0, 5.0)}
    for s in stops:
        coords[s] = (rng.uniform(0, 10), rng.uniform(0, 10))

    distance: Dict[Tuple[int, int], float] = {}
    for i in nodes:
        for j in nodes:
            if i != j:
                xi, yi = coords[i]
                xj, yj = coords[j]
                distance[i, j] = math.sqrt((xi - xj) ** 2 + (yi - yj) ** 2)

    demand: Dict[int, int] = {s: rng.randint(80, 500) for s in stops}
    return nodes, stops, depot, services, coords, distance, demand


def route_distance(route: List[int], depot: int,
                   distance: Dict[Tuple[int, int], float]) -> float:
    if not route:
        return 0.0
    total = distance[depot, route[0]]
    for a, b in zip(route[:-1], route[1:]):
        total += distance[a, b]
    total += distance[route[-1], depot]
    return total


def evaluate(
    routes: List[List[int]],
    depot: int,
    demand: Dict[int, int],
    distance: Dict[Tuple[int, int], float],
    K: int,
    epsilon: float,
    penalty_weight: float,
    exclusive: bool = True,
) -> Tuple[float, float, float, float]:
    """
    Returns (penalised_obj, total_distance, demand_loss, penalty).
    exclusive=True enforces no stop in more than one route.
    """
    total_dist = sum(route_distance(r, depot, distance) for r in routes)
    served: Dict[int, int] = {}   # stop → first service index
    penalty = 0.0

    for z_idx, route in enumerate(routes):
        if len(route) > K:
            penalty += penalty_weight * (len(route) - K)
        for stop in route:
            if stop in served:
                if exclusive:
                    penalty += penalty_weight   # hard: each duplicate is penalised
            else:
                served[stop] = z_idx

    served_demand = sum(demand[s] for s in served)
    total_demand  = sum(demand.values())
    demand_loss   = total_demand - served_demand

    if demand_loss > epsilon:
        penalty += penalty_weight * (demand_loss - epsilon)

    return total_dist + penalty, total_dist, demand_loss, penalty


def repair_exclusive(routes: List[List[int]], K: int) -> List[List[int]]:
    """
    Remove duplicate stop appearances across services and trim to K.
    Keeps the stop in the first service that claims it.
    This aligns the metaheuristics with Gurobi's exclusive-assignment constraint.
    """
    seen: set = set()
    repaired = []
    for route in routes:
        new_route = []
        for stop in route:
            if stop not in seen and len(new_route) < K:
                new_route.append(stop)
                seen.add(stop)
        repaired.append(new_route)
    return repaired


# ── Gurobi wrapper ────────────────────────────────────────────────────────────

def run_gurobi(nodes, stops, depot, services, distance, demand,
               K, epsilon, time_limit, max_dist=None):
    """
    Thin wrapper — calls your existing solve_bus_route_gurobi.
    Returns (distance, demand_loss, runtime, mip_gap) or None if infeasible.
    """
    try:
        from gurobi_model_small_sample_test import solve_bus_route_gurobi
    except ImportError:
        # Fallback path: try loading from same directory
        try:
            import importlib.util, os
            spec = importlib.util.spec_from_file_location(
                "gurobi_model",
                Path(__file__).parent / "gurobi_model_small_sample_test.py"
            )
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            solve_bus_route_gurobi = mod.solve_bus_route_gurobi
        except Exception as e:
            print(f"  [Gurobi] Could not import solver: {e}")
            return None

    result = solve_bus_route_gurobi(
        nodes=nodes,
        stops=stops,
        depot=depot,
        services=services,
        distance=distance,
        demand=demand,
        max_stops_per_service=K,
        max_distance_per_service=max_dist,
        demand_loss_limit=epsilon if epsilon < sum(demand.values()) else None,
        time_limit=time_limit,
        verbose=False,
        plot_progress=False,
        allow_stop_sharing=False,
    )

    if result is None:
        return None

    return {
        "distance":    result["objective_distance"],
        "demand_loss": result["demand_loss"],
        "runtime":     result["runtime"],
        "mip_gap":     result["mip_gap"],
        "routes":      result["routes"],
    }


# ── Simulated Annealing (self-contained, aligned with exclusive constraint) ───

class _SA:
    """
    Minimal SA that enforces exclusive assignment and matches Gurobi's model.
    Uses the same 4 operators described in your SA code/report.
    """

    OPERATORS = ["relocate", "swap", "two_opt", "insert_unserved"]

    def __init__(self, nodes, stops, depot, services, distance, demand,
                 K, epsilon, penalty_weight, T_init, T_final, cooling,
                 iters_per_temp, seed, operators=None):
        self.nodes    = nodes
        self.stops    = stops
        self.depot    = depot
        self.services = services
        self.distance = distance
        self.demand   = demand
        self.K        = K
        self.epsilon  = epsilon
        self.pw       = penalty_weight
        self.T_init   = T_init
        self.T_final  = T_final
        self.cooling  = cooling
        self.ipt      = iters_per_temp
        self.rng      = random.Random(seed)
        self.ops      = operators or self.OPERATORS
        self.total_demand = float(sum(demand.values()))

    def _eval(self, routes):
        return evaluate(routes, self.depot, self.demand, self.distance,
                        self.K, self.epsilon, self.pw, exclusive=True)

    def _copy(self, routes):
        return [list(r) for r in routes]

    # ── greedy initial solution ──────────────────────────────────────────────
    def _greedy_init(self):
        routes = [[] for _ in self.services]
        for stop in sorted(self.stops, key=lambda s: self.demand[s], reverse=True):
            best_delta, best_z, best_pos = float("inf"), None, None
            for z, route in enumerate(routes):
                if len(route) >= self.K:
                    continue
                for pos in range(len(route) + 1):
                    new_r = route[:pos] + [stop] + route[pos:]
                    delta = (route_distance(new_r, self.depot, self.distance)
                             - route_distance(route, self.depot, self.distance))
                    if delta < best_delta:
                        best_delta, best_z, best_pos = delta, z, pos
            if best_z is not None:
                routes[best_z].insert(best_pos, stop)
        return repair_exclusive(routes, self.K)

    # ── move operators ───────────────────────────────────────────────────────
    def _op_relocate(self, routes):
        new = self._copy(routes)
        non_empty = [i for i, r in enumerate(new) if r]
        if not non_empty:
            return new
        src = self.rng.choice(non_empty)
        pos = self.rng.randrange(len(new[src]))
        stop = new[src].pop(pos)
        # only insert into a route that doesn't already have this stop (exclusive)
        candidates = [i for i, r in enumerate(new)
                      if len(r) < self.K and stop not in r]
        if not candidates:
            new[src].insert(pos, stop)
            return new
        dst = self.rng.choice(candidates)
        ins = self.rng.randrange(len(new[dst]) + 1)
        new[dst].insert(ins, stop)
        return new

    def _op_swap(self, routes):
        new = self._copy(routes)
        non_empty = [i for i, r in enumerate(new) if r]
        if len(non_empty) < 2:
            return new
        r1, r2 = self.rng.sample(non_empty, 2)
        if not new[r1] or not new[r2]:
            return new
        p1 = self.rng.randrange(len(new[r1]))
        p2 = self.rng.randrange(len(new[r2]))
        s1, s2 = new[r1][p1], new[r2][p2]
        # only swap if neither stop already exists in the other route
        r1_others = set(new[r1]) - {s1}
        r2_others = set(new[r2]) - {s2}
        if s2 not in r1_others and s1 not in r2_others:
            new[r1][p1], new[r2][p2] = s2, s1
        return new

    def _op_two_opt(self, routes):
        new = self._copy(routes)
        candidates = [i for i, r in enumerate(new) if len(r) >= 4]
        if not candidates:
            return new
        ridx = self.rng.choice(candidates)
        i, j = sorted(self.rng.sample(range(len(new[ridx])), 2))
        if j - i >= 2:
            new[ridx][i:j + 1] = reversed(new[ridx][i:j + 1])
        return new

    def _op_insert_unserved(self, routes):
        new = self._copy(routes)
        served = set(s for r in new for s in r)
        unserved = [s for s in self.stops if s not in served]
        if not unserved:
            return new
        stop = max(unserved, key=lambda s: self.demand[s])
        candidates = [i for i, r in enumerate(new) if len(r) < self.K]
        if not candidates:
            return new
        ridx = self.rng.choice(candidates)
        best_pos, best_cost = 0, float("inf")
        for p in range(len(new[ridx]) + 1):
            trial = new[ridx][:p] + [stop] + new[ridx][p:]
            c = route_distance(trial, self.depot, self.distance)
            if c < best_cost:
                best_cost, best_pos = c, p
        new[ridx].insert(best_pos, stop)
        return new

    def _apply(self, routes, op):
        if op == "relocate":        return self._op_relocate(routes)
        if op == "swap":            return self._op_swap(routes)
        if op == "two_opt":         return self._op_two_opt(routes)
        if op == "insert_unserved": return self._op_insert_unserved(routes)
        raise ValueError(op)

    # ── main solve ───────────────────────────────────────────────────────────
    def solve(self):
        t0 = time.perf_counter()
        current = self._greedy_init()
        cur_obj, cur_dist, cur_loss, cur_pen = self._eval(current)

        best = self._copy(current)
        best_obj, best_dist, best_loss = cur_obj, cur_dist, cur_loss

        T = self.T_init
        history = []          # (elapsed, best_dist, best_loss)
        accepted = iters = 0

        while T > self.T_final:
            for _ in range(self.ipt):
                op        = self.rng.choice(self.ops)
                candidate = self._apply(current, op)
                n_obj, n_dist, n_loss, n_pen = self._eval(candidate)
                delta = n_obj - cur_obj

                if delta <= 0 or self.rng.random() < math.exp(-delta / max(T, 1e-12)):
                    current = candidate
                    cur_obj, cur_dist, cur_loss = n_obj, n_dist, n_loss
                    accepted += 1
                    if cur_obj < best_obj:
                        best = self._copy(current)
                        best_obj, best_dist, best_loss = cur_obj, cur_dist, cur_loss

                iters += 1

            history.append((time.perf_counter() - t0, best_dist, best_loss))
            T *= self.cooling

        runtime = time.perf_counter() - t0
        return {
            "distance":     best_dist,
            "demand_loss":  best_loss,
            "runtime":      runtime,
            "iterations":   iters,
            "accepted":     accepted,
            "history":      history,   # list of (time, dist, loss)
        }


def run_sa(nodes, stops, depot, services, distance, demand,
           K, epsilon, cfg, seed, operators=None):
    sa = _SA(
        nodes=nodes, stops=stops, depot=depot, services=services,
        distance=distance, demand=demand,
        K=K, epsilon=epsilon,
        penalty_weight=cfg["SA_PENALTY_WEIGHT"],
        T_init=cfg["SA_T_INIT"], T_final=cfg["SA_T_FINAL"],
        cooling=cfg["SA_COOLING"], iters_per_temp=cfg["SA_ITERS_PER_TEMP"],
        seed=seed, operators=operators,
    )
    return sa.solve()


# ── Genetic Algorithm (self-contained, aligned with exclusive constraint) ─────

class _GA:
    def __init__(self, nodes, stops, depot, services, distance, demand,
                 K, epsilon, penalty_weight, pop_size, elite, tournament,
                 crossover_rate, mutation_rate, max_gen, time_limit, seed):
        self.nodes         = nodes
        self.stops         = stops
        self.depot         = depot
        self.services      = services
        self.distance      = distance
        self.demand        = demand
        self.K             = K
        self.epsilon       = epsilon
        self.pw            = penalty_weight
        self.pop_size      = pop_size
        self.elite         = elite
        self.tournament    = tournament
        self.cx_rate       = crossover_rate
        self.mut_rate      = mutation_rate
        self.max_gen       = max_gen
        self.time_limit    = time_limit
        self.rng           = random.Random(seed)
        self.total_demand  = float(sum(demand.values()))

    def _eval(self, routes):
        return evaluate(routes, self.depot, self.demand, self.distance,
                        self.K, self.epsilon, self.pw, exclusive=True)

    def _greedy_seed(self):
        routes = [[] for _ in self.services]
        for stop in sorted(self.stops, key=lambda s: self.demand[s], reverse=True):
            best_delta, best_z, best_pos = float("inf"), None, None
            for z, route in enumerate(routes):
                if len(route) >= self.K:
                    continue
                for pos in range(len(route) + 1):
                    new_r = route[:pos] + [stop] + route[pos:]
                    delta = (route_distance(new_r, self.depot, self.distance)
                             - route_distance(route, self.depot, self.distance))
                    if delta < best_delta:
                        best_delta, best_z, best_pos = delta, z, pos
            if best_z is not None:
                routes[best_z].insert(best_pos, stop)
        return repair_exclusive(routes, self.K)

    def _random_individual(self):
        stops_shuffled = self.stops[:]
        self.rng.shuffle(stops_shuffled)
        routes = [[] for _ in self.services]
        for stop in stops_shuffled:
            z = self.rng.randrange(len(self.services))
            if len(routes[z]) < self.K:
                routes[z].append(stop)
        return repair_exclusive(routes, self.K)

    def _tournament_select(self, pop_scored):
        contestants = self.rng.sample(pop_scored, min(self.tournament, len(pop_scored)))
        return min(contestants, key=lambda x: x[0])[1]

    def _crossover(self, p1, p2):
        child = []
        used: set = set()
        for z in range(len(self.services)):
            route = []
            # take first half from p1 route, fill remainder from p2 route
            half = len(p1[z]) // 2
            for stop in p1[z][:half]:
                if stop not in used and len(route) < self.K:
                    route.append(stop)
                    used.add(stop)
            for stop in p2[z]:
                if stop not in used and len(route) < self.K:
                    route.append(stop)
                    used.add(stop)
            child.append(route)
        return repair_exclusive(child, self.K)

    def _mutate(self, routes):
        new = [list(r) for r in routes]
        op = self.rng.choice(["relocate", "swap", "two_opt", "insert_unserved"])

        if op == "relocate":
            non_empty = [i for i, r in enumerate(new) if r]
            if non_empty:
                src = self.rng.choice(non_empty)
                pos = self.rng.randrange(len(new[src]))
                stop = new[src].pop(pos)
                candidates = [i for i, r in enumerate(new)
                              if len(r) < self.K and stop not in r]
                if candidates:
                    dst = self.rng.choice(candidates)
                    new[dst].insert(self.rng.randrange(len(new[dst]) + 1), stop)
                else:
                    new[src].insert(pos, stop)

        elif op == "swap":
            non_empty = [i for i, r in enumerate(new) if r]
            if len(non_empty) >= 2:
                r1, r2 = self.rng.sample(non_empty, 2)
                if new[r1] and new[r2]:
                    p1 = self.rng.randrange(len(new[r1]))
                    p2 = self.rng.randrange(len(new[r2]))
                    s1, s2 = new[r1][p1], new[r2][p2]
                    if s2 not in set(new[r1]) - {s1} and s1 not in set(new[r2]) - {s2}:
                        new[r1][p1], new[r2][p2] = s2, s1

        elif op == "two_opt":
            candidates = [i for i, r in enumerate(new) if len(r) >= 4]
            if candidates:
                ridx = self.rng.choice(candidates)
                i, j = sorted(self.rng.sample(range(len(new[ridx])), 2))
                if j - i >= 2:
                    new[ridx][i:j + 1] = reversed(new[ridx][i:j + 1])

        elif op == "insert_unserved":
            served = set(s for r in new for s in r)
            unserved = [s for s in self.stops if s not in served]
            if unserved:
                stop = max(unserved, key=lambda s: self.demand[s])
                candidates = [i for i, r in enumerate(new) if len(r) < self.K]
                if candidates:
                    ridx = self.rng.choice(candidates)
                    new[ridx].insert(self.rng.randrange(len(new[ridx]) + 1), stop)

        return repair_exclusive(new, self.K)

    def solve(self):
        t0 = time.perf_counter()
        population = [self._greedy_seed()]
        while len(population) < self.pop_size:
            population.append(self._random_individual())

        best_routes = population[0]
        best_obj, best_dist, best_loss, _ = self._eval(best_routes)
        history = []

        for gen in range(self.max_gen):
            if time.perf_counter() - t0 > self.time_limit:
                break

            scored = [(self._eval(ind)[0], ind) for ind in population]
            scored.sort(key=lambda x: x[0])

            gen_best_obj = scored[0][0]
            _, gen_dist, gen_loss, _ = self._eval(scored[0][1])
            if gen_best_obj < best_obj:
                best_obj   = gen_best_obj
                best_dist  = gen_dist
                best_loss  = gen_loss
                best_routes = [list(r) for r in scored[0][1]]

            history.append((time.perf_counter() - t0, best_dist, best_loss))

            next_pop = [list(ind) for _, ind in scored[:self.elite]]
            while len(next_pop) < self.pop_size:
                p1 = self._tournament_select(scored)
                p2 = self._tournament_select(scored)
                child = self._crossover(p1, p2) if self.rng.random() < self.cx_rate \
                        else [list(r) for r in p1]
                if self.rng.random() < self.mut_rate:
                    child = self._mutate(child)
                next_pop.append(child)
            population = next_pop

        runtime = time.perf_counter() - t0
        return {
            "distance":    best_dist,
            "demand_loss": best_loss,
            "runtime":     runtime,
            "generations": gen + 1,
            "history":     history,
        }


def run_ga(nodes, stops, depot, services, distance, demand,
           K, epsilon, cfg, seed):
    ga = _GA(
        nodes=nodes, stops=stops, depot=depot, services=services,
        distance=distance, demand=demand,
        K=K, epsilon=epsilon,
        penalty_weight=cfg["GA_PENALTY_WEIGHT"],
        pop_size=cfg["GA_POP_SIZE"],
        elite=cfg["GA_ELITE"],
        tournament=cfg["GA_TOURNAMENT"],
        crossover_rate=cfg["GA_CROSSOVER_RATE"],
        mutation_rate=cfg["GA_MUTATION_RATE"],
        max_gen=cfg["GA_MAX_GENERATIONS"],
        time_limit=cfg["GA_TIME_LIMIT"],
        seed=seed,
    )
    return ga.solve()


# ── plotting helpers ──────────────────────────────────────────────────────────

def _savefig(fig, name, cfg):
    path = cfg["OUTPUT_DIR"] / f"{name}.png"
    fig.savefig(path, dpi=cfg["FIGURE_DPI"], bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


def plot_pareto(gurobi_pts, sa_pts, ga_pts, cfg):
    """Figure 1 — Pareto trade-off curves for all three methods."""
    fig, ax = plt.subplots(figsize=(8, 5))

    if gurobi_pts:
        xs = [p["distance"] for p in gurobi_pts]
        ys = [p["demand_loss"] for p in gurobi_pts]
        ax.plot(xs, ys, "o-", color="#1f77b4", label="Gurobi (exact)", zorder=5)
        for i, p in enumerate(gurobi_pts, 1):
            ax.annotate(str(i), (p["distance"], p["demand_loss"]),
                        textcoords="offset points", xytext=(4, 4), fontsize=8)

    if sa_pts:
        xs = [p["distance"] for p in sa_pts]
        ys = [p["demand_loss"] for p in sa_pts]
        ax.plot(xs, ys, "s--", color="#ff7f0e", label="SA (best run)", zorder=4)

    if ga_pts:
        xs = [p["distance"] for p in ga_pts]
        ys = [p["demand_loss"] for p in ga_pts]
        ax.plot(xs, ys, "^--", color="#2ca02c", label="GA (best run)", zorder=4)

    ax.set_xlabel("Total Travel Distance (km)", fontsize=11)
    ax.set_ylabel("Demand Loss (passengers)", fontsize=11)
    ax.set_title("Pareto Trade-off: Distance vs Demand Loss\n"
                 "Gurobi exact vs SA vs GA (small synthetic instance)", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _savefig(fig, "fig1_pareto_comparison", cfg)


def plot_gap_bar(comparison_rows, cfg):
    """Figure 2 — Gap (%) vs Gurobi optimum per epsilon."""
    eps_labels = [f"ε={r['epsilon_frac']:.0%}" for r in comparison_rows
                  if r["gurobi_dist"] is not None]
    sa_gaps = [r["sa_gap_pct"] for r in comparison_rows if r["gurobi_dist"] is not None]
    ga_gaps = [r["ga_gap_pct"] for r in comparison_rows if r["gurobi_dist"] is not None]

    if not eps_labels:
        return

    x = np.arange(len(eps_labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars_sa = ax.bar(x - width / 2, sa_gaps, width, label="SA", color="#ff7f0e", alpha=0.85)
    bars_ga = ax.bar(x + width / 2, ga_gaps, width, label="GA", color="#2ca02c", alpha=0.85)

    ax.bar_label(bars_sa, fmt="%.1f%%", padding=3, fontsize=8)
    ax.bar_label(bars_ga, fmt="%.1f%%", padding=3, fontsize=8)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Epsilon (demand-loss limit as % of total demand)", fontsize=11)
    ax.set_ylabel("Gap vs Gurobi Optimum (%)", fontsize=11)
    ax.set_title("Optimality Gap: SA and GA vs Gurobi Exact Solution", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(eps_labels)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    _savefig(fig, "fig2_optimality_gap", cfg)


def plot_runtime(comparison_rows, cfg):
    """Figure 3 — Runtime comparison (log scale)."""
    valid = [r for r in comparison_rows if r["gurobi_dist"] is not None]
    if not valid:
        return

    eps_labels = [f"ε={r['epsilon_frac']:.0%}" for r in valid]
    g_rt  = [r["gurobi_rt"]  for r in valid]
    sa_rt = [r["sa_rt_mean"] for r in valid]
    ga_rt = [r["ga_rt_mean"] for r in valid]

    x = np.arange(len(eps_labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width, g_rt,  width, label="Gurobi", color="#1f77b4", alpha=0.85)
    ax.bar(x,         sa_rt, width, label="SA (mean)", color="#ff7f0e", alpha=0.85)
    ax.bar(x + width, ga_rt, width, label="GA (mean)", color="#2ca02c", alpha=0.85)

    ax.set_yscale("log")
    ax.set_xlabel("Epsilon (demand-loss limit)", fontsize=11)
    ax.set_ylabel("Runtime (seconds, log scale)", fontsize=11)
    ax.set_title("Runtime Comparison: Gurobi vs SA vs GA", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(eps_labels)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    _savefig(fig, "fig3_runtime_comparison", cfg)


def plot_sa_convergence(sa_histories_by_eps, cfg):
    """Figure 4 — SA convergence curve (distance over iterations) per epsilon."""
    n = len(sa_histories_by_eps)
    if n == 0:
        return

    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, (eps_label, histories) in zip(axes, sa_histories_by_eps.items()):
        # histories is a list of (time, dist, loss) lists — one per seed
        for h in histories:
            iters = list(range(len(h)))
            dists = [pt[1] for pt in h]
            ax.plot(iters, dists, alpha=0.5, linewidth=0.8)
        ax.set_title(f"SA convergence\n{eps_label}", fontsize=9)
        ax.set_xlabel("Temperature steps", fontsize=8)
        ax.set_ylabel("Best distance (km)", fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("SA Convergence Across Seeds (5 runs per ε)", fontsize=11)
    fig.tight_layout()
    _savefig(fig, "fig4_sa_convergence", cfg)


def plot_operator_comparison(op_summary, cfg):
    """Figure 5 — SA operator comparison bar chart."""
    labels = [r["operator"] for r in op_summary]
    means  = [r["mean_dist"] for r in op_summary]
    stds   = [r["std_dist"]  for r in op_summary]
    losses = [r["mean_loss"] for r in op_summary]

    x = np.arange(len(labels))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    bars = ax1.bar(x, means, yerr=stds, capsize=4,
                   color=["#ff7f0e" if l != "all_operators" else "#1f77b4" for l in labels],
                   alpha=0.85)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax1.set_ylabel("Mean Distance (km)", fontsize=10)
    ax1.set_title("SA: Mean Route Distance by Operator Set\n(lower is better)", fontsize=10)
    ax1.grid(True, alpha=0.3, axis="y")

    ax2.bar(x, losses,
            color=["#ff7f0e" if l != "all_operators" else "#1f77b4" for l in labels],
            alpha=0.85)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax2.set_ylabel("Mean Demand Loss (passengers)", fontsize=10)
    ax2.set_title("SA: Mean Demand Loss by Operator Set\n(lower is better)", fontsize=10)
    ax2.grid(True, alpha=0.3, axis="y")

    # legend
    handles = [
        mpatches.Patch(color="#1f77b4", label="All operators (combined)"),
        mpatches.Patch(color="#ff7f0e", label="Single operator"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=9,
               bbox_to_anchor=(0.5, -0.02))

    fig.suptitle("SA Operator Comparison (5 runs, ε = 30% demand loss limit)", fontsize=11)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    _savefig(fig, "fig5_operator_comparison", cfg)


# ── main benchmark orchestrator ───────────────────────────────────────────────

def run_benchmark(cfg, scenario):
    scenario_name = scenario["name"]
    out_dir = Path(cfg["OUTPUT_DIR"]) / scenario_name
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg_local = dict(cfg)
    cfg_local["OUTPUT_DIR"] = out_dir

    # ── generate shared synthetic instance ──────────────────────────────────
    print("\n" + "=" * 70)
    print(f"GENERATING SYNTHETIC INSTANCE ({scenario_name.upper()})")
    print("=" * 70)
    nodes, stops, depot, services, coords, distance, demand = generate_data(
        scenario["num_stops"], scenario["num_services"], scenario["data_seed"]
    )
    total_demand = sum(demand.values())
    K = scenario["max_stops_per_svc"]

    print(f"  Scenario: {scenario_name}")
    print(f"  Stops: {scenario['num_stops']}  |  Services: {scenario['num_services']}"
          f"  |  K: {K}  |  Data seed: {scenario['data_seed']}  |  Total demand: {total_demand}")
    print(f"  Method seeds (SA/GA): {cfg['METHOD_SEEDS']}")
    print(f"  Stop demands: { {s: demand[s] for s in stops} }")

    epsilon_values = [
        int(total_demand * f) for f in cfg["EPSILON_FRACTIONS"]
    ]
    epsilon_values = sorted(set(epsilon_values), reverse=True)
    epsilon_fracs  = {eps: eps / total_demand for eps in epsilon_values}

    # ── run Gurobi ───────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("GUROBI EXACT SOLVER")
    print("=" * 70)
    gurobi_results = {}
    for eps in epsilon_values:
        print(f"  ε = {eps} ({epsilon_fracs[eps]:.0%} of demand)  ...", end=" ", flush=True)
        t0 = time.perf_counter()
        res = run_gurobi(nodes, stops, depot, services, distance, demand,
                         K, eps, cfg["GUROBI_TIME_LIMIT"], cfg["MAX_DIST_PER_SVC"])
        elapsed = time.perf_counter() - t0
        if res:
            gurobi_results[eps] = res
            print(f"dist={res['distance']:.3f}  loss={res['demand_loss']:.0f}"
                  f"  gap={res['mip_gap']*100:.2f}%  t={res['runtime']:.2f}s")
        else:
            print("INFEASIBLE / no solution")

    # ── run SA ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SIMULATED ANNEALING")
    print("=" * 70)
    sa_results = {}
    sa_histories = {}
    for eps in epsilon_values:
        runs = []
        histories = []
        for seed in cfg["METHOD_SEEDS"]:
            r = run_sa(nodes, stops, depot, services, distance, demand,
                       K, eps, cfg, seed)
            runs.append(r)
            histories.append(r["history"])
        sa_results[eps]  = runs
        sa_histories[f"ε={epsilon_fracs[eps]:.0%}"] = histories
        mean_d = statistics.mean(r["distance"] for r in runs)
        std_d  = statistics.stdev(r["distance"] for r in runs) if len(runs) > 1 else 0
        mean_l = statistics.mean(r["demand_loss"] for r in runs)
        print(f"  ε = {eps} ({epsilon_fracs[eps]:.0%})  "
              f"dist={mean_d:.3f}±{std_d:.3f}  loss={mean_l:.0f}  "
              f"t={statistics.mean(r['runtime'] for r in runs):.2f}s")

    # ── run GA ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("GENETIC ALGORITHM")
    print("=" * 70)
    ga_results = {}
    for eps in epsilon_values:
        runs = []
        for seed in cfg["METHOD_SEEDS"]:
            r = run_ga(nodes, stops, depot, services, distance, demand,
                       K, eps, cfg, seed)
            runs.append(r)
        ga_results[eps] = runs
        mean_d = statistics.mean(r["distance"] for r in runs)
        std_d  = statistics.stdev(r["distance"] for r in runs) if len(runs) > 1 else 0
        mean_l = statistics.mean(r["demand_loss"] for r in runs)
        print(f"  ε = {eps} ({epsilon_fracs[eps]:.0%})  "
              f"dist={mean_d:.3f}±{std_d:.3f}  loss={mean_l:.0f}  "
              f"t={statistics.mean(r['runtime'] for r in runs):.2f}s")

    # ── operator comparison experiment ────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SA OPERATOR COMPARISON EXPERIMENT")
    print("=" * 70)
    op_eps = int(total_demand * cfg["OP_EPSILON_FRACTION"])
    operator_sets = {
        "relocate_only":        ["relocate"],
        "swap_only":            ["swap"],
        "two_opt_only":         ["two_opt"],
        "insert_unserved_only": ["insert_unserved"],
        "all_operators":        ["relocate", "swap", "two_opt", "insert_unserved"],
    }
    op_summary = []
    for label, ops in operator_sets.items():
        runs = [
            run_sa(nodes, stops, depot, services, distance, demand,
                   K, op_eps, cfg, seed=100 + i * 17, operators=ops)
            for i in range(cfg["OP_REPEATS"])
        ]
        dists  = [r["distance"]    for r in runs]
        losses = [r["demand_loss"] for r in runs]
        op_summary.append({
            "operator":  label,
            "mean_dist": statistics.mean(dists),
            "std_dist":  statistics.stdev(dists) if len(dists) > 1 else 0,
            "best_dist": min(dists),
            "mean_loss": statistics.mean(losses),
            "mean_rt":   statistics.mean(r["runtime"] for r in runs),
        })
        print(f"  {label:<26}  dist={op_summary[-1]['mean_dist']:.3f}"
              f"±{op_summary[-1]['std_dist']:.3f}"
              f"  loss={op_summary[-1]['mean_loss']:.0f}")

    # ── build comparison table ────────────────────────────────────────────────
    comparison_rows = []
    for eps in epsilon_values:
        g = gurobi_results.get(eps)
        sa_runs = sa_results.get(eps, [])
        ga_runs = ga_results.get(eps, [])

        sa_best = min(sa_runs, key=lambda r: r["distance"]) if sa_runs else None
        ga_best = min(ga_runs, key=lambda r: r["distance"]) if ga_runs else None

        def gap(method_dist, gurobi_dist):
            if gurobi_dist and gurobi_dist > 1e-9:
                return 100.0 * (method_dist - gurobi_dist) / gurobi_dist
            return float("nan")

        comparison_rows.append({
            "epsilon":         eps,
            "epsilon_frac":    epsilon_fracs[eps],
            "gurobi_dist":     g["distance"]    if g else None,
            "gurobi_loss":     g["demand_loss"] if g else None,
            "gurobi_rt":       g["runtime"]     if g else None,
            "gurobi_gap":      g["mip_gap"]     if g else None,
            "sa_dist_mean":    statistics.mean(r["distance"]    for r in sa_runs) if sa_runs else None,
            "sa_dist_best":    min(r["distance"]    for r in sa_runs) if sa_runs else None,
            "sa_loss_mean":    statistics.mean(r["demand_loss"] for r in sa_runs) if sa_runs else None,
            "sa_rt_mean":      statistics.mean(r["runtime"]     for r in sa_runs) if sa_runs else None,
            "sa_gap_pct":      gap(sa_best["distance"], g["distance"] if g else None) if sa_best and g else float("nan"),
            "ga_dist_mean":    statistics.mean(r["distance"]    for r in ga_runs) if ga_runs else None,
            "ga_dist_best":    min(r["distance"]    for r in ga_runs) if ga_runs else None,
            "ga_loss_mean":    statistics.mean(r["demand_loss"] for r in ga_runs) if ga_runs else None,
            "ga_rt_mean":      statistics.mean(r["runtime"]     for r in ga_runs) if ga_runs else None,
            "ga_gap_pct":      gap(ga_best["distance"], g["distance"] if g else None) if ga_best and g else float("nan"),
        })

    # ── print summary table ───────────────────────────────────────────────────
    print("\n" + "=" * 100)
    print("PERFORMANCE BENCHMARK SUMMARY TABLE")
    print("=" * 100)
    hdr = (f"{'ε':>6}  {'Gurobi dist':>12}  {'Gurobi t':>9}  {'MIP gap':>7}  "
           f"{'SA best':>9}  {'SA gap%':>7}  {'SA t':>6}  "
           f"{'GA best':>9}  {'GA gap%':>7}  {'GA t':>6}")
    print(hdr)
    print("-" * 100)
    for r in comparison_rows:
        g_d  = f"{r['gurobi_dist']:.3f}"  if r["gurobi_dist"]  is not None else "—"
        g_t  = f"{r['gurobi_rt']:.2f}s"   if r["gurobi_rt"]    is not None else "—"
        g_g  = f"{r['gurobi_gap']*100:.2f}%" if r["gurobi_gap"] is not None else "—"
        sa_d = f"{r['sa_dist_best']:.3f}" if r["sa_dist_best"] is not None else "—"
        sa_g = f"{r['sa_gap_pct']:.1f}%"  if not math.isnan(r["sa_gap_pct"]) else "—"
        sa_t = f"{r['sa_rt_mean']:.2f}s"  if r["sa_rt_mean"]   is not None else "—"
        ga_d = f"{r['ga_dist_best']:.3f}" if r["ga_dist_best"] is not None else "—"
        ga_g = f"{r['ga_gap_pct']:.1f}%"  if not math.isnan(r["ga_gap_pct"]) else "—"
        ga_t = f"{r['ga_rt_mean']:.2f}s"  if r["ga_rt_mean"]   is not None else "—"
        print(f"{r['epsilon_frac']:>6.0%}  {g_d:>12}  {g_t:>9}  {g_g:>7}  "
              f"{sa_d:>9}  {sa_g:>7}  {sa_t:>6}  "
              f"{ga_d:>9}  {ga_g:>7}  {ga_t:>6}")

    # ── operator table ────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print(f"SA OPERATOR COMPARISON  (ε = {cfg['OP_EPSILON_FRACTION']:.0%} demand loss)")
    print("=" * 80)
    print(f"{'Operator':<26}  {'Mean dist':>10}  {'Std':>8}  {'Best':>8}  "
          f"{'Mean loss':>10}  {'Mean t':>7}")
    print("-" * 80)
    for row in sorted(op_summary, key=lambda r: r["mean_dist"]):
        print(f"{row['operator']:<26}  {row['mean_dist']:>10.3f}  "
              f"{row['std_dist']:>8.3f}  {row['best_dist']:>8.3f}  "
              f"{row['mean_loss']:>10.0f}  {row['mean_rt']:>7.2f}s")

    # ── plots ─────────────────────────────────────────────────────────────────
    print("\nGenerating figures ...")

    gurobi_pts = [{"distance": r["gurobi_dist"], "demand_loss": r["gurobi_loss"]}
                  for r in comparison_rows if r["gurobi_dist"] is not None]
    sa_pts     = [{"distance": r["sa_dist_best"], "demand_loss": r["sa_loss_mean"]}
                  for r in comparison_rows if r["sa_dist_best"] is not None]
    ga_pts     = [{"distance": r["ga_dist_best"], "demand_loss": r["ga_loss_mean"]}
                  for r in comparison_rows if r["ga_dist_best"] is not None]

    plot_pareto(gurobi_pts, sa_pts, ga_pts, cfg_local)
    plot_gap_bar(comparison_rows, cfg_local)
    plot_runtime(comparison_rows, cfg_local)
    plot_sa_convergence(sa_histories, cfg_local)
    plot_operator_comparison(op_summary, cfg_local)

    print(f"\nAll done for scenario '{scenario_name}'. Results saved to:", out_dir.resolve())
    return comparison_rows, op_summary


def run_all_scenarios(cfg):
    all_results = {}

    for scenario in cfg["SCENARIOS"]:
        comparison_rows, op_summary = run_benchmark(cfg, scenario)
        all_results[scenario["name"]] = {
            "comparison_rows": comparison_rows,
            "op_summary": op_summary,
        }

    print("\n" + "=" * 100)
    print("CROSS-SCENARIO QUICK SUMMARY (best distance at strictest epsilon)")
    print("=" * 100)
    for scenario in cfg["SCENARIOS"]:
        name = scenario["name"]
        rows = all_results[name]["comparison_rows"]
        strictest = min(rows, key=lambda r: r["epsilon"])

        g = strictest["gurobi_dist"]
        s = strictest["sa_dist_best"]
        a = strictest["ga_dist_best"]

        g_txt = f"{g:.3f}" if g is not None else "—"
        s_txt = f"{s:.3f}" if s is not None else "—"
        a_txt = f"{a:.3f}" if a is not None else "—"

        print(
            f"{name:<10} ε={strictest['epsilon_frac']:.0%}  "
            f"Gurobi={g_txt:<10} SA={s_txt:<10} GA={a_txt:<10}"
        )

    return all_results


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_all_scenarios(CFG)
"""
Genetic algorithm baseline for the prize-collecting multi-service bus routing problem.

This module is designed to benchmark against the notebook's Gurobi exact solver.
It uses the same synthetic data shape:
- node 0 is the interchange/depot,
- stops are 1..n,
- services are parallel routes starting/ending at the depot.

Problem interpretation:
- Minimise route distance.
- Stops may be skipped; skipped demand is demand loss.
- Multiple services may share the same stop.
- A stop's demand is counted as served once if at least one service visits it.
"""

from __future__ import annotations

from dataclasses import dataclass
import argparse
import math
import random
import statistics
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple


Stop = int
Node = int
Service = int
Route = List[Stop]
Routes = List[Route]


@dataclass
class GeneticAlgorithmConfig:
    population_size: int = 80
    elite_size: int = 6
    tournament_size: int = 4
    crossover_rate: float = 0.85
    mutation_rate: float = 0.35
    penalty_weight: float = 10_000.0
    max_generations: int = 400
    time_limit_seconds: Optional[float] = 30.0
    random_seed: int = 42


@dataclass
class GARunResult:
    best_routes: Routes
    best_distance: float
    demand_loss: float
    served_demand: float
    total_demand: float
    penalized_objective: float
    infeasible_penalty: float
    runtime_seconds: float
    generations: int
    evaluations: int
    history: List[Dict[str, float]]


def generate_data(
    num_stops: int = 10,
    num_services: int = 3,
    seed: int = 42,
) -> Tuple[List[Node], List[Stop], int, List[Service], Dict[int, Tuple[float, float]], Dict[Tuple[int, int], float], Dict[Stop, int]]:
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
            if i == j:
                continue
            xi, yi = coords[i]
            xj, yj = coords[j]
            distance[i, j] = math.sqrt((xi - xj) ** 2 + (yi - yj) ** 2)

    demand: Dict[Stop, int] = {s: rng.randint(80, 500) for s in stops}
    return nodes, stops, depot, services, coords, distance, demand


class BusRouteGeneticAlgorithm:
    """
    GA for a prize-collecting multi-service TSP approximation.

    Chromosome: list of service routes. Each service route is an ordered list of stops.
    Multiple services may include the same stop, but a stop cannot repeat inside the same route.
    """

    def __init__(
        self,
        nodes: Sequence[Node],
        stops: Sequence[Stop],
        depot: int,
        services: Sequence[Service],
        distance: Dict[Tuple[int, int], float],
        demand: Dict[Stop, int],
        max_stops_per_service: int = 4,
        max_distance_per_service: Optional[float] = None,
        demand_loss_limit: Optional[float] = None,
        config: Optional[GeneticAlgorithmConfig] = None,
    ) -> None:
        self.nodes = list(nodes)
        self.stops = list(stops)
        self.depot = depot
        self.services = list(services)
        self.distance = distance
        self.demand = demand
        self.K = max_stops_per_service
        self.max_distance_per_service = max_distance_per_service
        self.demand_loss_limit = demand_loss_limit
        self.config = config or GeneticAlgorithmConfig()
        self.rng = random.Random(self.config.random_seed)
        self.total_demand = float(sum(self.demand[s] for s in self.stops))
        self.evaluations = 0

    def _route_distance(self, route: Route) -> float:
        if not route:
            return 0.0
        total = 0.0
        current = self.depot
        for stop in route:
            total += self.distance[current, stop]
            current = stop
        total += self.distance[current, self.depot]
        return total

    def _total_distance(self, routes: Routes) -> float:
        return sum(self._route_distance(route) for route in routes)

    def _served_set(self, routes: Routes) -> set[Stop]:
        served: set[Stop] = set()
        for route in routes:
            served.update(route)
        return served

    def _repair_route(self, route: Route) -> Route:
        repaired: Route = []
        seen: set[Stop] = set()
        for stop in route:
            if stop in self.demand and stop not in seen:
                repaired.append(stop)
                seen.add(stop)
            if len(repaired) >= self.K:
                break
        return repaired

    def _repair(self, routes: Routes) -> Routes:
        repaired = [self._repair_route(route) for route in routes[: len(self.services)]]
        while len(repaired) < len(self.services):
            repaired.append([])

        if self.max_distance_per_service is not None:
            for route in repaired:
                while route and self._route_distance(route) > self.max_distance_per_service:
                    route.remove(min(route, key=lambda s: self.demand[s]))
        return repaired

    def _evaluate(self, routes: Routes) -> Tuple[float, float, float, float, float]:
        self.evaluations += 1
        routes = self._repair(routes)
        distance_value = self._total_distance(routes)
        served = self._served_set(routes)
        served_demand = float(sum(self.demand[s] for s in served))
        demand_loss = self.total_demand - served_demand

        penalty = 0.0
        if self.demand_loss_limit is not None and demand_loss > self.demand_loss_limit:
            penalty += self.config.penalty_weight * (demand_loss - self.demand_loss_limit)

        if self.max_distance_per_service is not None:
            for route in routes:
                excess = self._route_distance(route) - self.max_distance_per_service
                if excess > 0:
                    penalty += self.config.penalty_weight * excess

        return distance_value + penalty, distance_value, demand_loss, served_demand, penalty

    def _random_route(self) -> Route:
        route_len = self.rng.randint(0, self.K)
        if route_len == 0:
            return []
        return self.rng.sample(self.stops, k=min(route_len, len(self.stops)))

    def _greedy_seed(self) -> Routes:
        routes: Routes = [[] for _ in self.services]
        ordered = sorted(self.stops, key=lambda s: self.demand[s], reverse=True)

        for stop in ordered:
            # Multiple services can share stops, but the seed avoids duplicates until capacity is filled.
            best: Optional[Tuple[float, int, int]] = None
            for ridx, route in enumerate(routes):
                if len(route) >= self.K:
                    continue
                for pos in range(len(route) + 1):
                    candidate = route[:pos] + [stop] + route[pos:]
                    delta = self._route_distance(candidate) - self._route_distance(route)
                    if best is None or delta < best[0]:
                        best = (delta, ridx, pos)
            if best is not None:
                _, ridx, pos = best
                routes[ridx].insert(pos, stop)
        return routes

    def _initial_population(self) -> List[Routes]:
        population = [self._greedy_seed()]
        while len(population) < self.config.population_size:
            individual = [self._random_route() for _ in self.services]

            # Bias toward feasible demand coverage if an epsilon constraint is provided.
            if self.demand_loss_limit is not None and self.rng.random() < 0.65:
                individual = self._repair(individual)
                served = self._served_set(individual)
                unserved = sorted(
                    [s for s in self.stops if s not in served],
                    key=lambda s: self.demand[s],
                    reverse=True,
                )
                for stop in unserved:
                    if self.total_demand - sum(self.demand[s] for s in self._served_set(individual)) <= self.demand_loss_limit:
                        break
                    candidates = [i for i, r in enumerate(individual) if len(r) < self.K]
                    if not candidates:
                        break
                    ridx = self.rng.choice(candidates)
                    individual[ridx].append(stop)

            population.append(self._repair(individual))
        return population

    def _tournament_select(self, population: List[Routes], scores: List[Tuple[float, float, float, float, float]]) -> Routes:
        idxs = self.rng.sample(range(len(population)), k=min(self.config.tournament_size, len(population)))
        winner = min(idxs, key=lambda idx: scores[idx][0])
        return [list(route) for route in population[winner]]

    def _ordered_mix(self, left: Route, right: Route) -> Route:
        if not left and not right:
            return []
        child: Route = []
        for stop in left:
            if self.rng.random() < 0.55 and stop not in child:
                child.append(stop)
        for stop in right:
            if stop not in child and len(child) < self.K:
                child.append(stop)
        return self._repair_route(child)

    def _crossover(self, parent_a: Routes, parent_b: Routes) -> Routes:
        child: Routes = []
        for ridx in range(len(self.services)):
            if self.rng.random() < 0.5:
                child.append(self._ordered_mix(parent_a[ridx], parent_b[ridx]))
            else:
                child.append(self._ordered_mix(parent_b[ridx], parent_a[ridx]))
        return self._repair(child)

    def _mutate_relocate(self, routes: Routes) -> None:
        non_empty = [i for i, r in enumerate(routes) if r]
        if not non_empty:
            return
        src = self.rng.choice(non_empty)
        pos = self.rng.randrange(len(routes[src]))
        stop = routes[src].pop(pos)
        dst_candidates = [i for i, r in enumerate(routes) if len(r) < self.K]
        if not dst_candidates:
            routes[src].insert(pos, stop)
            return
        dst = self.rng.choice(dst_candidates)
        insert_pos = self.rng.randrange(len(routes[dst]) + 1)
        if stop not in routes[dst]:
            routes[dst].insert(insert_pos, stop)

    def _mutate_swap(self, routes: Routes) -> None:
        non_empty = [i for i, r in enumerate(routes) if r]
        if len(non_empty) < 2:
            return
        r1, r2 = self.rng.sample(non_empty, 2)
        p1 = self.rng.randrange(len(routes[r1]))
        p2 = self.rng.randrange(len(routes[r2]))
        routes[r1][p1], routes[r2][p2] = routes[r2][p2], routes[r1][p1]

    def _mutate_two_opt(self, routes: Routes) -> None:
        candidates = [i for i, r in enumerate(routes) if len(r) >= 4]
        if not candidates:
            return
        ridx = self.rng.choice(candidates)
        i, j = sorted(self.rng.sample(range(len(routes[ridx])), 2))
        routes[ridx][i : j + 1] = reversed(routes[ridx][i : j + 1])

    def _mutate_insert_prize(self, routes: Routes) -> None:
        served = self._served_set(routes)
        unserved = [s for s in self.stops if s not in served]
        if not unserved:
            return
        stop = max(unserved, key=lambda s: self.demand[s])
        candidates = [i for i, r in enumerate(routes) if len(r) < self.K]
        if not candidates:
            return
        ridx = self.rng.choice(candidates)
        pos = self.rng.randrange(len(routes[ridx]) + 1)
        routes[ridx].insert(pos, stop)

    def _mutate_remove_low_prize(self, routes: Routes) -> None:
        non_empty = [i for i, r in enumerate(routes) if r]
        if not non_empty:
            return
        ridx = self.rng.choice(non_empty)
        stop = min(routes[ridx], key=lambda s: self.demand[s])
        routes[ridx].remove(stop)

    def _mutate_duplicate_to_service(self, routes: Routes) -> None:
        # This operator explicitly permits shared stops across services.
        served = list(self._served_set(routes))
        if not served:
            return
        stop = self.rng.choice(served)
        candidates = [i for i, r in enumerate(routes) if len(r) < self.K and stop not in r]
        if not candidates:
            return
        ridx = self.rng.choice(candidates)
        pos = self.rng.randrange(len(routes[ridx]) + 1)
        routes[ridx].insert(pos, stop)

    def _mutate(self, routes: Routes) -> Routes:
        mutated = [list(route) for route in routes]
        operator = self.rng.choice(
            [
                self._mutate_relocate,
                self._mutate_swap,
                self._mutate_two_opt,
                self._mutate_insert_prize,
                self._mutate_remove_low_prize,
                self._mutate_duplicate_to_service,
            ]
        )
        operator(mutated)
        return self._repair(mutated)

    def solve(self) -> GARunResult:
        start = time.perf_counter()
        population = self._initial_population()
        history: List[Dict[str, float]] = []

        best_routes: Optional[Routes] = None
        best_eval: Optional[Tuple[float, float, float, float, float]] = None
        generation = 0

        while generation < self.config.max_generations:
            elapsed = time.perf_counter() - start
            if self.config.time_limit_seconds is not None and elapsed >= self.config.time_limit_seconds:
                break

            scores = [self._evaluate(individual) for individual in population]
            ranked = sorted(range(len(population)), key=lambda idx: scores[idx][0])
            gen_best_idx = ranked[0]
            gen_best_eval = scores[gen_best_idx]

            if best_eval is None or gen_best_eval[0] < best_eval[0]:
                best_eval = gen_best_eval
                best_routes = [list(route) for route in population[gen_best_idx]]

            feasible_scores = [score for score in scores if score[4] == 0]
            history.append(
                {
                    "generation": float(generation),
                    "elapsed_seconds": elapsed,
                    "best_penalized_objective": float(gen_best_eval[0]),
                    "best_distance": float(gen_best_eval[1]),
                    "best_demand_loss": float(gen_best_eval[2]),
                    "best_penalty": float(gen_best_eval[4]),
                    "avg_penalized_objective": float(statistics.mean(score[0] for score in scores)),
                    "feasible_count": float(len(feasible_scores)),
                }
            )

            next_population = [[list(route) for route in population[idx]] for idx in ranked[: self.config.elite_size]]
            while len(next_population) < self.config.population_size:
                p1 = self._tournament_select(population, scores)
                p2 = self._tournament_select(population, scores)
                if self.rng.random() < self.config.crossover_rate:
                    child = self._crossover(p1, p2)
                else:
                    child = [list(route) for route in p1]
                if self.rng.random() < self.config.mutation_rate:
                    child = self._mutate(child)
                next_population.append(self._repair(child))

            population = next_population
            generation += 1

        if best_routes is None or best_eval is None:
            scores = [self._evaluate(individual) for individual in population]
            idx = min(range(len(population)), key=lambda i: scores[i][0])
            best_routes = [list(route) for route in population[idx]]
            best_eval = scores[idx]

        runtime = time.perf_counter() - start
        penalized, dist, loss, served, penalty = best_eval
        return GARunResult(
            best_routes=best_routes,
            best_distance=dist,
            demand_loss=loss,
            served_demand=served,
            total_demand=self.total_demand,
            penalized_objective=penalized,
            infeasible_penalty=penalty,
            runtime_seconds=runtime,
            generations=generation,
            evaluations=self.evaluations,
            history=history,
        )


def routes_to_depot_paths(routes: Routes, depot: int = 0) -> Dict[int, List[int]]:
    return {idx: [depot] + list(route) + [depot] for idx, route in enumerate(routes)}


def benchmark_against_exact_solver(
    exact_solver: Callable[..., Optional[Dict[str, object]]],
    nodes: Sequence[Node],
    stops: Sequence[Stop],
    depot: int,
    services: Sequence[Service],
    distance: Dict[Tuple[int, int], float],
    demand: Dict[Stop, int],
    max_stops_per_service: int = 4,
    max_distance_per_service: Optional[float] = None,
    demand_loss_limit: Optional[float] = None,
    ga_config: Optional[GeneticAlgorithmConfig] = None,
    exact_time_limit: int = 60,
) -> Dict[str, object]:
    ga = BusRouteGeneticAlgorithm(
        nodes=nodes,
        stops=stops,
        depot=depot,
        services=services,
        distance=distance,
        demand=demand,
        max_stops_per_service=max_stops_per_service,
        max_distance_per_service=max_distance_per_service,
        demand_loss_limit=demand_loss_limit,
        config=ga_config or GeneticAlgorithmConfig(),
    )
    ga_result = ga.solve()

    exact_result = exact_solver(
        nodes=nodes,
        stops=stops,
        depot=depot,
        services=services,
        distance=distance,
        demand=demand,
        max_stops_per_service=max_stops_per_service,
        max_distance_per_service=max_distance_per_service,
        demand_loss_limit=demand_loss_limit,
        time_limit=exact_time_limit,
        verbose=False,
        plot_progress=False,
    )

    exact_distance = None
    exact_runtime = None
    exact_demand_loss = None
    exact_gap = None
    speedup = None

    if exact_result is not None:
        exact_distance_raw = exact_result.get("objective_distance")
        exact_runtime_raw = exact_result.get("runtime")
        exact_loss_raw = exact_result.get("demand_loss")
        if isinstance(exact_distance_raw, (int, float)):
            exact_distance = float(exact_distance_raw)
        if isinstance(exact_runtime_raw, (int, float)):
            exact_runtime = float(exact_runtime_raw)
        if isinstance(exact_loss_raw, (int, float)):
            exact_demand_loss = float(exact_loss_raw)
        if exact_distance is not None and exact_distance > 1e-12:
            exact_gap = 100.0 * (ga_result.best_distance - exact_distance) / exact_distance
        if exact_runtime is not None and ga_result.runtime_seconds > 1e-12:
            speedup = exact_runtime / ga_result.runtime_seconds

    return {
        "ga": {
            "distance": ga_result.best_distance,
            "runtime_s": ga_result.runtime_seconds,
            "demand_loss": ga_result.demand_loss,
            "served_demand": ga_result.served_demand,
            "penalty": ga_result.infeasible_penalty,
            "generations": ga_result.generations,
            "evaluations": ga_result.evaluations,
            "routes": routes_to_depot_paths(ga_result.best_routes, depot=depot),
        },
        "exact": {
            "distance": exact_distance,
            "runtime_s": exact_runtime,
            "demand_loss": exact_demand_loss,
        },
        "metrics": {
            "relative_gap_percent": exact_gap,
            "runtime_speedup_exact_over_ga": speedup,
        },
        "ga_result": ga_result,
        "exact_result": exact_result,
    }


def plot_ga_result(
    result: GARunResult,
    coords: Optional[Dict[int, Tuple[float, float]]] = None,
    depot: int = 0,
    title: str = "Genetic Algorithm Result",
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("Matplotlib not available. Install with: pip install matplotlib")
        return

    fig, axes = plt.subplots(1, 2 if coords else 1, figsize=(15 if coords else 8, 5))
    if not isinstance(axes, (list, tuple)):
        try:
            axes_list = list(axes)
        except TypeError:
            axes_list = [axes]
    else:
        axes_list = list(axes)

    ax = axes_list[0]
    if result.history:
        xs = [h["elapsed_seconds"] for h in result.history]
        ax.plot(xs, [h["best_distance"] for h in result.history], label="Best distance")
        ax.plot(xs, [h["best_penalized_objective"] for h in result.history], label="Best penalized objective", alpha=0.75)
        ax.set_xlabel("CPU time (seconds)")
        ax.set_ylabel("Objective value")
        ax.set_title("GA convergence")
        ax.grid(True, alpha=0.3)
        ax.legend()

    if coords and len(axes_list) > 1:
        ax2 = axes_list[1]
        depot_x, depot_y = coords[depot]
        ax2.scatter([depot_x], [depot_y], marker="s", s=120, label="Depot")
        colors = plt.cm.tab10.colors
        for service_idx, route in enumerate(result.best_routes):
            path = [depot] + route + [depot]
            xs = [coords[node][0] for node in path]
            ys = [coords[node][1] for node in path]
            ax2.plot(xs, ys, marker="o", color=colors[service_idx % len(colors)], label=f"Service {service_idx}")
            for node in route:
                ax2.text(coords[node][0], coords[node][1], str(node), fontsize=8)
        ax2.set_title("Best route layout")
        ax2.set_xlabel("x")
        ax2.set_ylabel("y")
        ax2.grid(True, alpha=0.3)
        ax2.legend()

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def run_demo(
    num_stops: int = 10,
    num_services: int = 3,
    seed: int = 7,
    max_stops_per_service: int = 4,
    demand_loss_fraction: float = 0.15,
    max_generations: int = 300,
    time_limit_seconds: Optional[float] = 30.0,
) -> GARunResult:
    nodes, stops, depot, services, coords, distance, demand = generate_data(
        num_stops=num_stops,
        num_services=num_services,
        seed=seed,
    )
    total_demand = sum(demand.values())
    demand_loss_limit = float(int(total_demand * demand_loss_fraction))

    config = GeneticAlgorithmConfig(
        population_size=80,
        elite_size=6,
        tournament_size=4,
        crossover_rate=0.85,
        mutation_rate=0.35,
        max_generations=max_generations,
        time_limit_seconds=time_limit_seconds,
        random_seed=42,
    )

    solver = BusRouteGeneticAlgorithm(
        nodes=nodes,
        stops=stops,
        depot=depot,
        services=services,
        distance=distance,
        demand=demand,
        max_stops_per_service=max_stops_per_service,
        demand_loss_limit=demand_loss_limit,
        config=config,
    )
    result = solver.solve()

    print("\nGA Result")
    print("-" * 80)
    print(f"Distance: {result.best_distance:.3f}")
    print(f"Demand loss: {result.demand_loss:.1f}")
    print(f"Served demand: {result.served_demand:.1f} / {result.total_demand:.1f}")
    print(f"Penalty: {result.infeasible_penalty:.3f}")
    print(f"Runtime (s): {result.runtime_seconds:.3f}")
    print(f"Generations: {result.generations}")
    print(f"Evaluations: {result.evaluations}")
    for service_id, path in routes_to_depot_paths(result.best_routes, depot).items():
        print(f"Service {service_id}: {path}")

    plot_ga_result(result, coords=coords, depot=depot)
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GA benchmark for bus route redesign.")
    parser.add_argument("--stops", type=int, default=10)
    parser.add_argument("--services", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-stops", type=int, default=4)
    parser.add_argument("--demand-loss-fraction", type=float, default=0.15)
    parser.add_argument("--generations", type=int, default=300)
    parser.add_argument("--time-limit", type=float, default=30.0)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_demo(
        num_stops=args.stops,
        num_services=args.services,
        seed=args.seed,
        max_stops_per_service=args.max_stops,
        demand_loss_fraction=args.demand_loss_fraction,
        max_generations=args.generations,
        time_limit_seconds=args.time_limit,
    )

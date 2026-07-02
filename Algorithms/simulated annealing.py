"""
Simulated annealing baseline for bus service routing redesign.

This module is designed to be compared against the notebook's exact Gurobi solver.
It implements:
1) A route representation for multiple services (multiple tours from one depot).
2) Several neighborhood move operators for simulated annealing.
3) Utilities to benchmark operator performance and compare against an exact solver.
"""

from __future__ import annotations

from dataclasses import dataclass
import argparse
import math
import random
import statistics
import time
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple


Stop = int
Node = int
Service = int
Route = List[Stop]  # stop sequence only; depot is implicit at start/end.
Routes = List[Route]


@dataclass
class SimulatedAnnealingConfig:
	initial_temperature: float = 250.0
	final_temperature: float = 1e-3
	cooling_rate: float = 0.995
	iterations_per_temp: int = 200
	penalty_weight: float = 10_000.0
	random_seed: int = 42


@dataclass
class SARunResult:
	best_routes: Routes
	best_distance: float
	demand_loss: float
	served_demand: float
	total_demand: float
	penalized_objective: float
	runtime_seconds: float
	iterations: int
	accepted_moves: int
	infeasible_penalty: float
	operators_used: Tuple[str, ...]


def generate_data(
	num_stops: int = 10,
	num_services: int = 3,
	seed: int = 42,
) -> Tuple[List[Node], List[Stop], int, List[Service], Dict[int, Tuple[float, float]], Dict[Tuple[int, int], float], Dict[Stop, int]]:
	"""
	Generate synthetic bus routing data compatible with the notebook model.
	"""
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


class BusRouteSimulatedAnnealing:
	"""
	Simulated annealing solver for a multi-service bus routing approximation.

	Constraints mirrored from the exact model as closely as possible:
	- each stop assigned to at most one service,
	- each service can visit at most K stops,
	- optional demand loss upper bound,
	- optional max route distance per service.
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
		config: Optional[SimulatedAnnealingConfig] = None,
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
		self.config = config or SimulatedAnnealingConfig()
		self.rng = random.Random(self.config.random_seed)

		self.total_demand = float(sum(self.demand[s] for s in self.stops))

	def _route_distance(self, route: Route) -> float:
		if not route:
			return 2.0 * self.distance[self.depot, self.depot] if (self.depot, self.depot) in self.distance else 0.0

		total = 0.0
		current = self.depot
		for s in route:
			total += self.distance[current, s]
			current = s
		total += self.distance[current, self.depot]
		return total

	def _total_distance(self, routes: Routes) -> float:
		return sum(self._route_distance(r) for r in routes)

	def _served_stops(self, routes: Routes) -> List[Stop]:
		served: List[Stop] = []
		for r in routes:
			served.extend(r)
		return served

	def _evaluate(self, routes: Routes) -> Tuple[float, float, float, float, float]:
		"""
		Returns:
			(penalized_objective, distance, demand_loss, served_demand, penalty)
		"""
		distance_value = self._total_distance(routes)
		served = self._served_stops(routes)
		served_set = set(served)
		served_demand = float(sum(self.demand[s] for s in served_set))
		demand_loss = self.total_demand - served_demand

		penalty = 0.0

		duplicates = len(served) - len(served_set)
		if duplicates > 0:
			penalty += self.config.penalty_weight * duplicates

		for r in routes:
			if len(r) > self.K:
				penalty += self.config.penalty_weight * (len(r) - self.K)

			if self.max_distance_per_service is not None:
				route_dist = self._route_distance(r)
				if route_dist > self.max_distance_per_service:
					penalty += self.config.penalty_weight * (route_dist - self.max_distance_per_service)

		if self.demand_loss_limit is not None and demand_loss > self.demand_loss_limit:
			penalty += self.config.penalty_weight * (demand_loss - self.demand_loss_limit)

		penalized = distance_value + penalty
		return penalized, distance_value, demand_loss, served_demand, penalty

	def _copy_routes(self, routes: Routes) -> Routes:
		return [list(r) for r in routes]

	def _initial_solution(self) -> Routes:
		"""
		Greedy initialization by demand descending with cheapest insertion.
		"""
		routes: Routes = [[] for _ in self.services]
		ordered_stops = sorted(self.stops, key=lambda s: self.demand[s], reverse=True)

		for stop in ordered_stops:
			best_delta = float("inf")
			best_route_idx = None
			best_pos = None

			for z in range(len(routes)):
				if len(routes[z]) >= self.K:
					continue
				candidate_route = routes[z]
				for pos in range(len(candidate_route) + 1):
					new_route = candidate_route[:pos] + [stop] + candidate_route[pos:]
					delta = self._route_distance(new_route) - self._route_distance(candidate_route)
					if delta < best_delta:
						best_delta = delta
						best_route_idx = z
						best_pos = pos

			if best_route_idx is not None and best_pos is not None:
				routes[best_route_idx].insert(best_pos, stop)

		return routes

	# ---------------------------
	# Move operators
	# ---------------------------

	def _move_relocate(self, routes: Routes) -> Routes:
		new_routes = self._copy_routes(routes)
		non_empty = [idx for idx, r in enumerate(new_routes) if r]
		if not non_empty:
			return new_routes

		src_idx = self.rng.choice(non_empty)
		stop_pos = self.rng.randrange(len(new_routes[src_idx]))
		stop = new_routes[src_idx].pop(stop_pos)

		dst_candidates = [i for i, r in enumerate(new_routes) if len(r) < self.K]
		if not dst_candidates:
			new_routes[src_idx].insert(stop_pos, stop)
			return new_routes

		dst_idx = self.rng.choice(dst_candidates)
		insert_pos = self.rng.randrange(len(new_routes[dst_idx]) + 1)
		new_routes[dst_idx].insert(insert_pos, stop)
		return new_routes

	def _move_swap(self, routes: Routes) -> Routes:
		new_routes = self._copy_routes(routes)
		non_empty = [idx for idx, r in enumerate(new_routes) if r]
		if len(non_empty) < 2:
			return new_routes

		r1, r2 = self.rng.sample(non_empty, 2)
		p1 = self.rng.randrange(len(new_routes[r1]))
		p2 = self.rng.randrange(len(new_routes[r2]))
		new_routes[r1][p1], new_routes[r2][p2] = new_routes[r2][p2], new_routes[r1][p1]
		return new_routes

	def _move_two_opt(self, routes: Routes) -> Routes:
		new_routes = self._copy_routes(routes)
		candidates = [idx for idx, r in enumerate(new_routes) if len(r) >= 4]
		if not candidates:
			return new_routes

		ridx = self.rng.choice(candidates)
		route = new_routes[ridx]
		i, j = sorted(self.rng.sample(range(len(route)), 2))
		if j - i < 2:
			return new_routes
		route[i : j + 1] = reversed(route[i : j + 1])
		return new_routes

	def _move_cross_exchange(self, routes: Routes) -> Routes:
		new_routes = self._copy_routes(routes)
		candidates = [idx for idx, r in enumerate(new_routes) if len(r) >= 2]
		if len(candidates) < 2:
			return new_routes

		r1, r2 = self.rng.sample(candidates, 2)
		route1, route2 = new_routes[r1], new_routes[r2]

		i1 = self.rng.randrange(1, len(route1))
		i2 = self.rng.randrange(1, len(route2))
		new_r1 = route1[:i1] + route2[i2:]
		new_r2 = route2[:i2] + route1[i1:]

		if len(new_r1) <= self.K and len(new_r2) <= self.K:
			new_routes[r1] = new_r1
			new_routes[r2] = new_r2
		return new_routes

	def _move_insert_unserved(self, routes: Routes) -> Routes:
		new_routes = self._copy_routes(routes)
		served = set(self._served_stops(new_routes))
		unserved = [s for s in self.stops if s not in served]
		if not unserved:
			return new_routes

		stop = max(unserved, key=lambda s: self.demand[s])
		candidates = [i for i, r in enumerate(new_routes) if len(r) < self.K]
		if not candidates:
			return new_routes

		ridx = self.rng.choice(candidates)
		route = new_routes[ridx]
		best_pos = 0
		best_delta = float("inf")

		for pos in range(len(route) + 1):
			candidate = route[:pos] + [stop] + route[pos:]
			delta = self._route_distance(candidate) - self._route_distance(route)
			if delta < best_delta:
				best_delta = delta
				best_pos = pos

		route.insert(best_pos, stop)
		return new_routes

	def _apply_move(self, routes: Routes, operator: str) -> Routes:
		if operator == "relocate":
			return self._move_relocate(routes)
		if operator == "swap":
			return self._move_swap(routes)
		if operator == "two_opt":
			return self._move_two_opt(routes)
		if operator == "cross_exchange":
			return self._move_cross_exchange(routes)
		if operator == "insert_unserved":
			return self._move_insert_unserved(routes)
		raise ValueError(f"Unknown operator: {operator}")

	def solve(
		self,
		operators: Optional[Sequence[str]] = None,
	) -> SARunResult:
		ops = tuple(operators) if operators else (
			"relocate",
			"swap",
			"two_opt",
			"cross_exchange",
			"insert_unserved",
		)

		start_time = time.perf_counter()
		current_routes = self._initial_solution()
		current_eval = self._evaluate(current_routes)

		best_routes = self._copy_routes(current_routes)
		best_eval = current_eval

		T = self.config.initial_temperature
		accepted = 0
		iterations = 0

		while T > self.config.final_temperature:
			for _ in range(self.config.iterations_per_temp):
				operator = self.rng.choice(ops)
				candidate_routes = self._apply_move(current_routes, operator)
				cand_eval = self._evaluate(candidate_routes)

				delta = cand_eval[0] - current_eval[0]
				if delta <= 0:
					current_routes = candidate_routes
					current_eval = cand_eval
					accepted += 1
				else:
					p = math.exp(-delta / max(T, 1e-12))
					if self.rng.random() < p:
						current_routes = candidate_routes
						current_eval = cand_eval
						accepted += 1

				if current_eval[0] < best_eval[0]:
					best_routes = self._copy_routes(current_routes)
					best_eval = current_eval

				iterations += 1
			T *= self.config.cooling_rate

		runtime = time.perf_counter() - start_time
		penalized, dist, loss, served, penalty = best_eval

		return SARunResult(
			best_routes=best_routes,
			best_distance=dist,
			demand_loss=loss,
			served_demand=served,
			total_demand=self.total_demand,
			penalized_objective=penalized,
			runtime_seconds=runtime,
			iterations=iterations,
			accepted_moves=accepted,
			infeasible_penalty=penalty,
			operators_used=ops,
		)


def compare_move_operators(
	nodes: Sequence[Node],
	stops: Sequence[Stop],
	depot: int,
	services: Sequence[Service],
	distance: Dict[Tuple[int, int], float],
	demand: Dict[Stop, int],
	max_stops_per_service: int = 4,
	max_distance_per_service: Optional[float] = None,
	demand_loss_limit: Optional[float] = None,
	repeats_per_setting: int = 5,
	base_seed: int = 100,
	config: Optional[SimulatedAnnealingConfig] = None,
) -> List[Dict[str, float]]:
	"""
	Compare SA operator sets. Returns sorted summary rows.
	"""
	operator_sets = {
		"relocate_only": ("relocate",),
		"swap_only": ("swap",),
		"two_opt_only": ("two_opt",),
		"cross_exchange_only": ("cross_exchange",),
		"insert_unserved_only": ("insert_unserved",),
		"all_operators": (
			"relocate",
			"swap",
			"two_opt",
			"cross_exchange",
			"insert_unserved",
		),
	}

	cfg = config or SimulatedAnnealingConfig()
	summary: List[Dict[str, float]] = []

	for label, ops in operator_sets.items():
		distances: List[float] = []
		runtimes: List[float] = []
		penalties: List[float] = []
		demand_losses: List[float] = []
		served_demands: List[float] = []

		for r in range(repeats_per_setting):
			run_cfg = SimulatedAnnealingConfig(
				initial_temperature=cfg.initial_temperature,
				final_temperature=cfg.final_temperature,
				cooling_rate=cfg.cooling_rate,
				iterations_per_temp=cfg.iterations_per_temp,
				penalty_weight=cfg.penalty_weight,
				random_seed=base_seed + 1000 * r + len(label),
			)

			solver = BusRouteSimulatedAnnealing(
				nodes=nodes,
				stops=stops,
				depot=depot,
				services=services,
				distance=distance,
				demand=demand,
				max_stops_per_service=max_stops_per_service,
				max_distance_per_service=max_distance_per_service,
				demand_loss_limit=demand_loss_limit,
				config=run_cfg,
			)
			result = solver.solve(operators=ops)
			distances.append(result.best_distance)
			runtimes.append(result.runtime_seconds)
			penalties.append(result.infeasible_penalty)
			demand_losses.append(result.demand_loss)
			served_demands.append(result.served_demand)

		summary.append(
			{
				"operator_set": label,
				"avg_distance": float(statistics.mean(distances)),
				"best_distance": float(min(distances)),
				"avg_runtime_s": float(statistics.mean(runtimes)),
				"avg_penalty": float(statistics.mean(penalties)),
				"avg_demand_loss": float(statistics.mean(demand_losses)),
				"avg_served_demand": float(statistics.mean(served_demands)),
			}
		)

	summary.sort(key=lambda row: (row["avg_penalty"], row["avg_distance"], row["avg_runtime_s"]))
	return summary


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
	sa_config: Optional[SimulatedAnnealingConfig] = None,
	sa_operators: Optional[Sequence[str]] = None,
	exact_time_limit: int = 180,
) -> Dict[str, object]:
	"""
	Compare SA against the exact (Gurobi) callable used in your notebook.

	The exact_solver callable should follow the same signature as your notebook's
	solve_bus_route_gurobi function and return a result dict with
	"objective_distance" and "runtime" keys.
	"""
	sa_solver = BusRouteSimulatedAnnealing(
		nodes=nodes,
		stops=stops,
		depot=depot,
		services=services,
		distance=distance,
		demand=demand,
		max_stops_per_service=max_stops_per_service,
		max_distance_per_service=max_distance_per_service,
		demand_loss_limit=demand_loss_limit,
		config=sa_config or SimulatedAnnealingConfig(),
	)

	sa_result = sa_solver.solve(operators=sa_operators)

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
	)

	exact_distance = None
	exact_runtime = None
	gap_percent = None
	speedup = None

	if exact_result is not None:
		exact_distance_obj = exact_result.get("objective_distance")
		exact_runtime_obj = exact_result.get("runtime")
		if isinstance(exact_distance_obj, (int, float)):
			exact_distance = float(exact_distance_obj)
		if isinstance(exact_runtime_obj, (int, float)):
			exact_runtime = float(exact_runtime_obj)

		if exact_distance is not None and exact_distance > 1e-12:
			gap_percent = 100.0 * (sa_result.best_distance - exact_distance) / exact_distance
		if exact_runtime is not None and sa_result.runtime_seconds > 1e-12:
			speedup = exact_runtime / sa_result.runtime_seconds

	return {
		"sa": {
			"distance": sa_result.best_distance,
			"runtime_s": sa_result.runtime_seconds,
			"demand_loss": sa_result.demand_loss,
			"served_demand": sa_result.served_demand,
			"penalty": sa_result.infeasible_penalty,
			"operators": list(sa_result.operators_used),
		},
		"exact": {
			"distance": exact_distance,
			"runtime_s": exact_runtime,
		},
		"metrics": {
			"relative_gap_percent": gap_percent,
			"runtime_speedup_exact_over_sa": speedup,
		},
	}


def routes_to_depot_paths(routes: Routes, depot: int = 0) -> Dict[int, List[int]]:
	"""
	Convert SA route format into notebook-like full paths including depot.
	"""
	output: Dict[int, List[int]] = {}
	for idx, route in enumerate(routes):
		output[idx] = [depot] + list(route) + [depot]
	return output


__all__ = [
	"SimulatedAnnealingConfig",
	"SARunResult",
	"BusRouteSimulatedAnnealing",
	"generate_data",
	"compare_move_operators",
	"benchmark_against_exact_solver",
	"routes_to_depot_paths",
]


def _print_operator_summary(summary: List[Dict[str, float]]) -> None:
	print("\n" + "=" * 90)
	print("SIMULATED ANNEALING BENCHMARK RESULTS")
	print("=" * 90)
	print(
		f"{'Rank':<6} "
		f"{'Operator':<24} "
		f"{'Avg Distance':<14} "
		f"{'Avg Demand Loss':<16} "
		f"{'Avg Served':<12} "
		f"{'Avg Runtime':<12} "
		f"{'Avg Penalty':<12}"
	)

	for rank, row in enumerate(summary, start=1):
		print(
			f"{rank:<6} "
			f"{str(row['operator_set']):<24} "
			f"{row['avg_distance']:<14.3f} "
			f"{row['avg_demand_loss']:<16.1f} "
			f"{row['avg_served_demand']:<12.1f} "
			f"{row['avg_runtime_s']:<12.3f} "
			f"{row['avg_penalty']:<12.3f}"
		)


def _plot_operator_summary(summary: List[Dict[str, float]]) -> None:
	try:
		import matplotlib.pyplot as plt
	except Exception:
		print("\nMatplotlib not available. Install with: pip install matplotlib")
		print("Skipping charts.")
		return

	labels = [str(row["operator_set"]) for row in summary]
	avg_distance = [float(row["avg_distance"]) for row in summary]
	avg_runtime = [float(row["avg_runtime_s"]) for row in summary]
	avg_penalty = [float(row["avg_penalty"]) for row in summary]
	avg_demand_loss = [float(row["avg_demand_loss"]) for row in summary]

	fig, axes = plt.subplots(2, 2, figsize=(18, 10))
	axes = axes.flatten()

	axes[0].bar(labels, avg_distance)
	axes[0].set_title("Average Distance by Operator Set")
	axes[0].set_ylabel("Distance")
	axes[0].tick_params(axis="x", rotation=45)

	axes[1].bar(labels, avg_runtime)
	axes[1].set_title("Average Runtime by Operator Set")
	axes[1].set_ylabel("Runtime (seconds)")
	axes[1].tick_params(axis="x", rotation=45)

	axes[2].bar(labels, avg_penalty)
	axes[2].set_title("Average Penalty by Operator Set")
	axes[2].set_ylabel("Penalty")
	axes[2].tick_params(axis="x", rotation=45)

	axes[3].bar(labels, avg_demand_loss)
	axes[3].set_title("Average Demand Loss by Operator Set")
	axes[3].set_ylabel("Demand Loss")
	axes[3].tick_params(axis="x", rotation=45)

	plt.tight_layout()
	plt.show()


def run_demo(
	num_stops: int = 10,
	num_services: int = 3,
	seed: int = 7,
	max_stops_per_service: int = 4,
	demand_loss_fraction: float = 0.15,
	repeats: int = 4,
) -> None:
	"""
	Run a full SA demo that prints value outputs and plots operator performance.
	"""
	nodes, stops, depot, services, _, distance, demand = generate_data(
		num_stops=num_stops,
		num_services=num_services,
		seed=seed,
	)

	total_demand = sum(demand[s] for s in stops)
	demand_loss_limit = float(int(total_demand * demand_loss_fraction))

	config = SimulatedAnnealingConfig(
		initial_temperature=250.0,
		final_temperature=1e-3,
		cooling_rate=0.995,
		iterations_per_temp=200,
		penalty_weight=10_000.0,
		random_seed=42,
	)

	print("\nInput Summary")
	print("-" * 88)
	print(f"Stops: {num_stops}")
	print(f"Services: {num_services}")
	print(f"Max stops/service: {max_stops_per_service}")
	print(f"Total demand: {total_demand}")
	print(f"Demand-loss limit: {demand_loss_limit}")
	print(f"Operator repeats: {repeats}")

	summary = compare_move_operators(
		nodes=nodes,
		stops=stops,
		depot=depot,
		services=services,
		distance=distance,
		demand=demand,
		max_stops_per_service=max_stops_per_service,
		demand_loss_limit=demand_loss_limit,
		repeats_per_setting=repeats,
		base_seed=123,
		config=config,
	)

	_print_operator_summary(summary)
	_plot_operator_summary(summary)

	best_operator_label = str(summary[0]["operator_set"])
	operator_map: Dict[str, Tuple[str, ...]] = {
		"relocate_only": ("relocate",),
		"swap_only": ("swap",),
		"two_opt_only": ("two_opt",),
		"cross_exchange_only": ("cross_exchange",),
		"insert_unserved_only": ("insert_unserved",),
		"all_operators": (
			"relocate",
			"swap",
			"two_opt",
			"cross_exchange",
			"insert_unserved",
		),
	}

	best_solver = BusRouteSimulatedAnnealing(
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
	best_result = best_solver.solve(operators=operator_map[best_operator_label])

	print("\nBest-Setting Detailed Output")
	print("-" * 88)
	print(f"Best operator setting: {best_operator_label}")
	print(f"Distance: {best_result.best_distance:.3f}")
	print(f"Demand loss: {best_result.demand_loss:.1f}")
	print(f"Served demand: {best_result.served_demand:.1f} / {best_result.total_demand:.1f}")
	print(f"Runtime (s): {best_result.runtime_seconds:.4f}")
	print(f"Iterations: {best_result.iterations}")
	print(f"Accepted moves: {best_result.accepted_moves}")
	print(f"Penalty: {best_result.infeasible_penalty:.3f}")

	best_paths = routes_to_depot_paths(best_result.best_routes, depot=depot)
	for service_id, path in best_paths.items():
		print(f"Service {service_id}: {path}")


def _parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Run SA benchmark and visualize operator performance.")
	parser.add_argument("--stops", type=int, default=10, help="Number of bus stops.")
	parser.add_argument("--services", type=int, default=3, help="Number of services.")
	parser.add_argument("--seed", type=int, default=7, help="Random seed for data generation.")
	parser.add_argument("--max-stops", type=int, default=4, help="Maximum stops per service.")
	parser.add_argument(
		"--demand-loss-fraction",
		type=float,
		default=0.15,
		help="Demand loss limit as fraction of total demand.",
	)
	parser.add_argument("--repeats", type=int, default=4, help="Repeats per operator set.")
	return parser.parse_args()


if __name__ == "__main__":
	args = _parse_args()
	run_demo(
		num_stops=args.stops,
		num_services=args.services,
		seed=args.seed,
		max_stops_per_service=args.max_stops,
		demand_loss_fraction=args.demand_loss_fraction,
		repeats=args.repeats,
	)

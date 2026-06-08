from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from src.model import (
    D_MAX,
    N_TRUCKS,
    P_MAX,
    Q,
    R,
    T,
)


class GreedyConstructor:
    def __init__(self, data: dict[str, Any], variables: dict[str, Any]):
        self.data = data
        self.variables = variables
        self.depot_id = data["depot"]["id"]
        self.distances = data["distances"]
        self.stores_by_id = {store["id"]: store for store in data["stores"]}
        self.arc_set = set(variables["arcs"])
        self.days = list(variables["days"])
        self.trips = list(variables["trips"])
        self.daily_km: dict[tuple[int, int], float] = defaultdict(float)

    def build(self) -> dict[str, Any]:
        warm_start: dict[str, Any] = {
            "x": {},
            "y": {},
            "u": {},
            "cost": 0.0,
            "distance_cost": 0.0,
            "unassigned": [],
        }

        if set(self.days).issubset(set(range(1, 8))):
            pending = self._required_visits(include_quincenal=True)
            self._assign_period(pending, self.days, warm_start)
        elif set(self.days).issubset(set(range(8, T + 1))):
            pending = self._required_visits(include_quincenal=False)
            self._assign_period(pending, self.days, warm_start)
        else:
            first_half = self._required_visits(include_quincenal=True)
            second_half = self._required_visits(include_quincenal=False)
            self._assign_period(first_half, [day for day in self.days if 1 <= day <= 7], warm_start)
            self._assign_period(second_half, [day for day in self.days if 8 <= day <= T], warm_start)
        return warm_start

    def _required_visits(self, include_quincenal: bool) -> list[str]:
        visits = []
        for store in self.data["stores"]:
            if store["fi"] == 2 or (include_quincenal and store["fi"] == 1):
                visits.append(store["id"])
        return sorted(visits, key=lambda i: self.distances[self.depot_id][i])

    def _assign_period(
        self,
        pending: list[str],
        days: range,
        warm_start: dict[str, Any],
    ) -> None:
        unassigned = list(pending)
        for day in days:
            for vehicle in range(N_TRUCKS):
                for trip in self.trips:
                    if not unassigned:
                        return

                    route = self._build_route(unassigned, vehicle, day)
                    if not route:
                        route = [unassigned[0]]

                    for store_id in route:
                        unassigned.remove(store_id)
                    route_km = self._route_km(route)
                    self.daily_km[(vehicle, day)] += route_km
                    self._write_route_start(warm_start, route, vehicle, day, trip)

        warm_start["unassigned"].extend(unassigned)

    def _build_route(self, unassigned: list[str], vehicle: int, day: int) -> list[str]:
        route: list[str] = []
        demand = 0.0

        while len(route) < P_MAX:
            candidates = sorted(
                unassigned,
                key=lambda i: self.distances[self.depot_id][i],
            )
            chosen = None
            for store_id in candidates:
                if store_id in route:
                    continue
                candidate_route = [*route, store_id]
                candidate_demand = demand + float(self.stores_by_id[store_id]["qi"])
                candidate_km = self._route_km(candidate_route)
                incremental_km = candidate_km - self._route_km(route)

                if candidate_demand > Q:
                    continue
                if not self._route_arcs_exist(candidate_route):
                    continue
                if candidate_km > D_MAX:
                    continue
                chosen = store_id
                break

            if chosen is None:
                break

            route.append(chosen)
            demand += float(self.stores_by_id[chosen]["qi"])

        return route

    def _write_route_start(
        self,
        warm_start: dict[str, Any],
        route: list[str],
        vehicle: int,
        day: int,
        trip: int,
    ) -> None:
        previous = self.depot_id
        route_cost = 0.0
        for position, store_id in enumerate(route, start=1):
            warm_start["x"][(previous, store_id, vehicle, day, trip)] = 1.0
            warm_start["y"][(store_id, vehicle, day, trip)] = 1.0
            warm_start["u"][(store_id, vehicle, day, trip)] = float(position)
            route_cost += self.distances[previous][store_id]
            previous = store_id

        warm_start["x"][(previous, self.depot_id, vehicle, day, trip)] = 1.0
        route_cost += self.distances[previous][self.depot_id]
        warm_start["distance_cost"] += route_cost
        warm_start["cost"] += route_cost

    def _route_arcs_exist(self, route: list[str]) -> bool:
        previous = self.depot_id
        for store_id in route:
            if (previous, store_id) not in self.arc_set:
                return False
            previous = store_id
        return (previous, self.depot_id) in self.arc_set

    def _route_km(self, route: list[str]) -> float:
        if not route:
            return 0.0

        previous = self.depot_id
        total = 0.0
        for store_id in route:
            total += self.distances[previous][store_id]
            previous = store_id
        return total + self.distances[previous][self.depot_id]


def validate_warm_start(
    warm_start: dict[str, Any],
    data: dict[str, Any],
    variables: dict[str, Any] | None = None,
) -> list[str]:
    depot_id = data["depot"]["id"]
    distances = data["distances"]
    stores = data["stores"]
    store_ids = [store["id"] for store in stores]
    qi = {store["id"]: float(store["qi"]) for store in stores}
    fi = {store["id"]: int(store["fi"]) for store in stores}
    nodes = [depot_id, *store_ids]
    arcs = variables["arcs"] if variables else [(i, j) for i in nodes for j in nodes if i != j]
    days = list(variables["days"]) if variables else list(range(1, T + 1))
    trips = list(variables["trips"]) if variables else list(range(1, R + 1))
    x_selected = {
        key for key, selected in warm_start.get("x", {}).items() if _is_one(selected)
    }
    y_selected = {
        key for key, selected in warm_start.get("y", {}).items() if _is_one(selected)
    }
    violations: list[str] = []
    warnings: list[str] = []

    print("Warm start validation:", flush=True)

    for i, j, v, t, r in x_selected:
        if (i, j) not in arcs:
            violations.append(f"arc-filter: x[{i},{j},{v},{t},{r}] is not a model arc")

    for store_id in store_ids:
        for vehicle in range(N_TRUCKS):
            for day in days:
                for trip in trips:
                    y_value = 1 if (store_id, vehicle, day, trip) in y_selected else 0
                    flow_in = sum(
                        1
                        for i in nodes
                        if (i, store_id, vehicle, day, trip) in x_selected
                    )
                    flow_out = sum(
                        1
                        for j in nodes
                        if (store_id, j, vehicle, day, trip) in x_selected
                    )
                    if flow_in != y_value:
                        violations.append(
                            f"eq.2: store {store_id}, v{vehicle}, d{day}, r{trip} "
                            f"has flow_in={flow_in}, y={y_value}"
                        )
                    if flow_out != y_value:
                        violations.append(
                            f"eq.3: store {store_id}, v{vehicle}, d{day}, r{trip} "
                            f"has flow_out={flow_out}, y={y_value}"
                        )

    for vehicle in range(N_TRUCKS):
        for day in days:
            for trip in trips:
                depot_out = sum(
                    1 for j in store_ids if (depot_id, j, vehicle, day, trip) in x_selected
                )
                depot_in = sum(
                    1 for i in store_ids if (i, depot_id, vehicle, day, trip) in x_selected
                )
                if depot_out != depot_in or depot_out > 1:
                    violations.append(
                        f"eq.4: v{vehicle}, d{day}, r{trip} has "
                        f"depot_out={depot_out}, depot_in={depot_in}"
                    )

                stops = sum(
                    1 for i in store_ids if (i, vehicle, day, trip) in y_selected
                )
                demand = sum(
                    qi[i] for i in store_ids if (i, vehicle, day, trip) in y_selected
                )
                if demand > Q:
                    violations.append(
                        f"eq.10: v{vehicle}, d{day}, r{trip} demand={demand:.2f}"
                    )
                if stops > P_MAX:
                    violations.append(
                        f"eq.11: v{vehicle}, d{day}, r{trip} stops={stops}"
                    )

            for trip in trips:
                route_km = sum(
                    distances[i][j]
                    for i, j, v, t, r in x_selected
                    if v == vehicle and t == day and r == trip
                )
                if route_km > D_MAX + 1e-6:
                    violations.append(
                        f"eq.5: v{vehicle}, d{day}, r{trip} "
                        f"route_km={route_km:.2f} > D_MAX={D_MAX}"
                    )

    visit_counts = Counter(i for i, _, _, _ in y_selected)
    day_set = set(days)
    is_week_1 = day_set.issubset(set(range(1, 8)))
    is_week_2 = day_set.issubset(set(range(8, T + 1)))
    for store_id in store_ids:
        if fi[store_id] == 1 and not is_week_2 and visit_counts[store_id] != 1:
            violations.append(
                f"eq.7: store {store_id} has {visit_counts[store_id]} visits"
            )
        if fi[store_id] == 2:
            if is_week_1:
                if visit_counts[store_id] != 1:
                    violations.append(
                        f"eq.8: store {store_id} has {visit_counts[store_id]} week-1 visits"
                    )
            elif is_week_2:
                if visit_counts[store_id] != 1:
                    violations.append(
                        f"eq.9: store {store_id} has {visit_counts[store_id]} week-2 visits"
                    )
            else:
                first_half = sum(
                    1 for i, _, day, _ in y_selected if i == store_id and 1 <= day <= 7
                )
                second_half = sum(
                    1 for i, _, day, _ in y_selected if i == store_id and 8 <= day <= 14
                )
                if first_half != 1:
                    violations.append(
                        f"eq.8: store {store_id} has {first_half} first-half visits"
                    )
                if second_half != 1:
                    violations.append(
                        f"eq.9: store {store_id} has {second_half} second-half visits"
                    )

    if violations:
        print(f"Warm start violations found: {len(violations)}", flush=True)
        for violation in violations[:50]:
            print(f"  - {violation}", flush=True)
        if len(violations) > 50:
            print(f"  ... {len(violations) - 50} more violations", flush=True)
    else:
        print("Warm start violations found: 0", flush=True)

    if warnings:
        print(f"Warm start warnings found: {len(warnings)}", flush=True)
        for warning in warnings[:50]:
            print(f"  - {warning}", flush=True)
        if len(warnings) > 50:
            print(f"  ... {len(warnings) - 50} more warnings", flush=True)
    else:
        print("Warm start warnings found: 0", flush=True)

    return violations


def _is_one(value: Any) -> bool:
    return value is not None and float(value) >= 0.5

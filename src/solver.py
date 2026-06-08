from __future__ import annotations

import time
from collections import Counter, defaultdict
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any

import pandas as pd
from pulp import LpProblem, LpStatus, PULP_CBC_CMD, constants, value
from pulp.apis.core import PulpSolverError


class Solver:
    def __init__(self, prob: LpProblem, variables: dict[str, Any], data: dict[str, Any]):
        self.prob = prob
        self.variables = variables
        self.data = data
        self.data.setdefault("days", list(self.variables.get("days", [])))
        self.status: str | None = None
        self.solve_time_sec: float | None = None

    def solve(
        self,
        time_limit: int = 300,
        mip_gap: float = 0.05,
        warm_start: dict[str, dict[Any, float]] | None = None,
        stream_progress: bool = False,
        checkpoint_path: str | Path | None = None,
    ) -> str:
        if warm_start:
            self._apply_warm_start(warm_start)

        if stream_progress:
            return self._solve_streaming_cbc(
                time_limit=time_limit,
                mip_gap=mip_gap,
                warm_start=warm_start,
                checkpoint_path=checkpoint_path,
            )

        solver = PULP_CBC_CMD(
            timeLimit=time_limit,
            gapRel=mip_gap,
            warmStart=bool(warm_start),
            keepFiles=bool(warm_start),
        )

        start = time.perf_counter()
        self.prob.solve(solver)
        self.solve_time_sec = time.perf_counter() - start
        self.status = LpStatus[self.prob.status]
        if warm_start:
            self._cleanup_pulp_files()
        return self.status

    def _solve_streaming_cbc(
        self,
        time_limit: int,
        mip_gap: float,
        warm_start: dict[str, dict[Any, float]] | None,
        checkpoint_path: str | Path | None,
    ) -> str:
        solver = PULP_CBC_CMD(
            timeLimit=time_limit,
            gapRel=mip_gap,
            warmStart=bool(warm_start),
            keepFiles=True,
        )
        if not solver.executable(solver.path):
            raise PulpSolverError(f"Pulp: cannot execute {solver.path}")

        tmp_lp, tmp_mps, tmp_sol, tmp_mst = solver.create_tmp_files(
            self.prob.name,
            "lp",
            "mps",
            "sol",
            "mst",
        )
        vs, variable_names, constraint_names, _ = self.prob.writeMPS(tmp_mps, rename=1)
        args = [solver.path, tmp_mps]

        if self.prob.sense == constants.LpMaximize:
            args.append("-max")
        if warm_start:
            solver.writesol(tmp_mst, self.prob, vs, variable_names, constraint_names)
            args.extend(["-mips", tmp_mst])
            if checkpoint_path:
                self._write_checkpoint(checkpoint_path)
        args.extend(["-sec", str(time_limit), "-ratio", str(mip_gap), "-timeMode", "elapsed"])
        args.extend(["-solve", "-printingOptions", "all", "-solution", tmp_sol])

        best_objective: float | None = None
        start = time.perf_counter()
        process = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )

        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            incumbent = self._parse_incumbent_objective(line)
            if incumbent is None:
                continue
            if best_objective is not None and incumbent >= best_objective - 1e-9:
                continue

            best_objective = incumbent
            elapsed = int(time.perf_counter() - start)
            gap = self._parse_gap_percent(line)
            gap_text = f"{gap:.0f}%" if gap is not None else "unknown"
            print(
                f"[t={elapsed}s] New best solution found: "
                f"{incumbent:.2f} km (gap: {gap_text})",
                flush=True,
            )
            if checkpoint_path:
                self._try_checkpoint_solution(
                    tmp_sol,
                    solver,
                    vs,
                    variable_names,
                    constraint_names,
                    checkpoint_path,
                )

        return_code = process.wait()
        self.solve_time_sec = time.perf_counter() - start
        if return_code != 0:
            raise PulpSolverError("Pulp: Error while trying to execute CBC")

        if not os.path.exists(tmp_sol):
            raise PulpSolverError("Pulp: CBC did not produce a solution file")

        self._assign_solution_file(
            solver,
            tmp_sol,
            vs,
            variable_names,
            constraint_names,
        )
        self.status = LpStatus[self.prob.status]
        if checkpoint_path:
            self._write_checkpoint(checkpoint_path)
        solver.delete_tmp_files(tmp_mps, tmp_lp, tmp_sol, tmp_mst)
        return self.status

    def extract_solution(self) -> list[dict[str, Any]]:
        if not self._has_solution_values():
            print(
                "WARNING: extract_solution called before solver values are available; "
                "Routes sheet will be empty.",
                flush=True,
            )
            return []

        rows: list[dict[str, Any]] = []
        stores_by_id = {store["id"]: store for store in self.data["stores"]}

        for route in self._extract_routes():
            cumulative_km = 0.0
            previous = self.data["depot"]["id"]
            for stop_order, store_id in enumerate(route["stores"], start=1):
                cumulative_km += self.data["distances"][previous][store_id]
                store = stores_by_id[store_id]
                rows.append(
                    {
                        "day": route["day"],
                        "vehicle": route["vehicle"],
                        "trip": route["trip"],
                        "stop_order": stop_order,
                        "store_id": store_id,
                        "store_name": store["name"],
                        "demand_boxes": store["qi"],
                        "cumulative_km": cumulative_km,
                        "route_km": route["km"],
                        "route_boxes": route["boxes"],
                    }
                )
                previous = store_id

        return rows

    def print_route_summary(self) -> None:
        routes = self._extract_routes()
        route_by_key = {
            (route["day"], route["vehicle"], route["trip"]): route for route in routes
        }
        stores_by_id = {store["id"]: store for store in self.data["stores"]}

        print("Route summary:", flush=True)
        for day in self._days():
            day_routes = [route for route in routes if route["day"] == day]
            if not day_routes:
                print(f"Day {day}: (no routes)", flush=True)
                continue

            print(f"Day {day}:", flush=True)
            for vehicle in self.variables["trucks"]:
                for trip in self.variables["trips"]:
                    route = route_by_key.get((day, vehicle, trip))
                    if not route:
                        print(f"  Vehicle {vehicle}, Trip {trip}: no trip", flush=True)
                        continue

                    stops = " -> ".join(
                        f"[{store_id} {stores_by_id[store_id]['name']}]"
                        for store_id in route["stores"]
                    )
                    print(
                        f"  Vehicle {vehicle}, Trip {trip}: DEPOT -> {stops} -> DEPOT "
                        f"(km: {route['km']:.2f}, boxes: {route['boxes']:.2f})",
                        flush=True,
                    )

        self._print_solution_checks(routes)

    def export(self, path: str | Path = "outputs/solution_mazatlan.xlsx") -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        route_rows = self.extract_solution()
        routes_df = pd.DataFrame(route_rows)
        by_day_df = self._build_by_day_dataframe()
        summary_df = pd.DataFrame(
            [
                {
                    "status": self.status or LpStatus[self.prob.status],
                    "solve_time_sec": self.solve_time_sec,
                    "total_km": self._total_km(),
                    "total_cost": value(self.prob.objective),
                    "n_routes_used": self._n_routes_used(),
                }
            ]
        )

        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            summary_df.to_excel(writer, sheet_name="Summary", index=False)
            routes_df.to_excel(writer, sheet_name="Routes", index=False)
            by_day_df.to_excel(writer, sheet_name="By_Day")

        return output_path

    def _extract_routes(self) -> list[dict[str, Any]]:
        if not self._has_solution_values():
            return []

        routes: list[dict[str, Any]] = []
        stores_by_id = {store["id"]: store for store in self.data["stores"]}
        store_ids = list(stores_by_id)
        y = self.variables["y"]

        for vehicle, day, trip in self._route_keys():
            visited = [
                store_id
                for store_id in store_ids
                if self._is_selected(y[(store_id, vehicle, day, trip)])
            ]
            if not visited:
                continue

            ordered_stores = self._reconstruct_route(vehicle, day, trip, visited)
            route_km = self._route_km(ordered_stores)
            route_boxes = sum(float(stores_by_id[store_id]["qi"]) for store_id in ordered_stores)
            routes.append(
                {
                    "day": day,
                    "vehicle": vehicle,
                    "trip": trip,
                    "stores": ordered_stores,
                    "km": route_km,
                    "boxes": route_boxes,
                }
            )

        return routes

    def _route_keys(self) -> list[tuple[int, int, int]]:
        vehicles = self.variables["trucks"]
        days = self._days()
        trips = self.variables["trips"]
        return [(vehicle, day, trip) for vehicle in vehicles for day in days for trip in trips]

    def _days(self) -> list[int]:
        return list(self.data.get("days", self.variables["days"]))

    def _reconstruct_route(
        self,
        vehicle: int,
        day: int,
        trip: int,
        visited: list[str],
    ) -> list[str]:
        depot_id = self.data["depot"]["id"]
        x = self.variables["x"]
        selected_outgoing: dict[str, list[str]] = defaultdict(list)

        for i, j in self.variables["arcs"]:
            if (i, j, vehicle, day, trip) in x and self._is_selected(
                x[(i, j, vehicle, day, trip)]
            ):
                selected_outgoing[i].append(j)

        visited_set = set(visited)
        ordered: list[str] = []
        current = depot_id
        seen_nodes = {depot_id}

        while True:
            next_nodes = sorted(selected_outgoing.get(current, []))
            if not next_nodes:
                break
            next_node = next_nodes[0]
            if next_node == depot_id:
                break
            if next_node in seen_nodes:
                break
            seen_nodes.add(next_node)
            if next_node in visited_set:
                ordered.append(next_node)
                visited_set.remove(next_node)
            current = next_node

        if visited_set:
            ordered.extend(sorted(visited_set))

        return ordered

    def _route_km(self, stores: list[str]) -> float:
        if not stores:
            return 0.0

        distances = self.data["distances"]
        depot_id = self.data["depot"]["id"]
        previous = depot_id
        total = 0.0
        for store_id in stores:
            total += distances[previous][store_id]
            previous = store_id
        return total + distances[previous][depot_id]

    def _print_solution_checks(self, routes: list[dict[str, Any]]) -> None:
        store_visits = Counter(
            store_id for route in routes for store_id in route["stores"]
        )
        required_visits = {
            store["id"]: 1 if int(store["fi"]) == 1 else 2 for store in self.data["stores"]
        }
        vehicles_by_day: dict[int, set[int]] = defaultdict(set)
        for route in routes:
            vehicles_by_day[route["day"]].add(route["vehicle"])

        over_visited = [
            store_id
            for store_id, count in store_visits.items()
            if count > required_visits[store_id]
        ]
        not_visited = [
            store_id for store_id in required_visits if store_visits[store_id] == 0
        ]

        print("Route summary totals:", flush=True)
        print(f"  Total km: {self._total_km():.2f}", flush=True)
        print(f"  Total routes used: {len(routes)}", flush=True)
        print("  Vehicles used per day:", flush=True)
        for day in self._days():
            vehicles = sorted(vehicles_by_day.get(day, set()))
            label = ", ".join(str(vehicle) for vehicle in vehicles) if vehicles else "none"
            print(f"    Day {day}: {label}", flush=True)
        print(
            "  Stores visited more than required: "
            + (", ".join(over_visited) if over_visited else "none"),
            flush=True,
        )
        print(
            "  Stores not visited at all: "
            + (", ".join(not_visited) if not_visited else "none"),
            flush=True,
        )

    def _build_by_day_dataframe(self) -> pd.DataFrame:
        records = [
            {"day": day, "vehicle": vehicle, "km": km}
            for (vehicle, day), km in self._daily_vehicle_km().items()
        ]
        if not records:
            return pd.DataFrame()

        return (
            pd.DataFrame(records)
            .pivot_table(index="day", columns="vehicle", values="km", aggfunc="sum", fill_value=0)
            .sort_index()
        )

    def _daily_vehicle_km(self) -> dict[tuple[int, int], float]:
        daily_km: dict[tuple[int, int], float] = {}
        for route in self._extract_routes():
            key = (route["vehicle"], route["day"])
            daily_km[key] = daily_km.get(key, 0.0) + route["km"]
        return daily_km

    def _total_km(self) -> float:
        return sum(self._daily_vehicle_km().values())

    def _n_routes_used(self) -> int:
        return len(self._extract_routes())

    def _apply_warm_start(self, warm_start: dict[str, dict[Any, float]]) -> None:
        for variable_name in ("x", "y", "u"):
            variable_values = warm_start.get(variable_name, {})
            variables = self.variables[variable_name]
            for variable in variables.values():
                variable.setInitialValue(0.0)
                variable.varValue = 0.0
            for key, initial_value in variable_values.items():
                if key in variables:
                    variables[key].setInitialValue(initial_value)
                    variables[key].varValue = initial_value

    def _cleanup_pulp_files(self) -> None:
        for suffix in (".mps", ".mst", ".sol"):
            path = Path(f"{self.prob.name}-pulp{suffix}")
            if path.exists():
                path.unlink()

    def _has_solution_values(self) -> bool:
        y_variables = self.variables.get("y", {})
        if not y_variables:
            return False
        return any(self._variable_value(variable) is not None for variable in y_variables.values())

    def _assign_solution_file(
        self,
        solver: PULP_CBC_CMD,
        solution_path: str,
        variables: list[Any],
        variable_names: dict[str, str],
        constraint_names: dict[str, str],
    ) -> None:
        status, values, reduced_costs, shadow_prices, slacks, sol_status = solver.readsol_MPS(
            solution_path,
            self.prob,
            variables,
            variable_names,
            constraint_names,
        )
        self.prob.assignVarsVals(values)
        self.prob.assignVarsDj(reduced_costs)
        self.prob.assignConsPi(shadow_prices)
        self.prob.assignConsSlack(slacks, activity=True)
        self.prob.assignStatus(status, sol_status)

    def _try_checkpoint_solution(
        self,
        solution_path: str,
        solver: PULP_CBC_CMD,
        variables: list[Any],
        variable_names: dict[str, str],
        constraint_names: dict[str, str],
        checkpoint_path: str | Path,
    ) -> None:
        try:
            if not os.path.exists(solution_path):
                print(
                    "[checkpoint] CBC has not written a readable solution file yet; "
                    "skipping checkpoint.",
                    flush=True,
                )
                return

            self._assign_solution_file(
                solver,
                solution_path,
                variables,
                variable_names,
                constraint_names,
            )
            self._write_checkpoint(checkpoint_path)
        except Exception as exc:
            print(f"[checkpoint] Could not write checkpoint yet: {exc}", flush=True)

    def _write_checkpoint(self, checkpoint_path: str | Path) -> None:
        path = Path(checkpoint_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        route_rows = self.extract_solution()
        if not route_rows:
            if path.exists():
                print(
                    "[checkpoint] No readable route values yet; keeping previous "
                    f"checkpoint at {path}",
                    flush=True,
                )
            else:
                print(
                    "[checkpoint] No readable route values yet; checkpoint not written.",
                    flush=True,
                )
            return

        self._export_with_route_rows(path, route_rows)
        print(f"[checkpoint] Saved current best routes to {path}", flush=True)
        self._verify_checkpoint(path)

    def _export_with_route_rows(
        self,
        path: Path,
        route_rows: list[dict[str, Any]],
    ) -> None:
        routes_df = pd.DataFrame(route_rows)
        by_day_df = self._build_by_day_dataframe()
        summary_df = pd.DataFrame(
            [
                {
                    "status": self.status or LpStatus[self.prob.status],
                    "solve_time_sec": self.solve_time_sec,
                    "total_km": self._total_km(),
                    "total_cost": value(self.prob.objective),
                    "n_routes_used": self._n_routes_used(),
                }
            ]
        )

        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            summary_df.to_excel(writer, sheet_name="Summary", index=False)
            routes_df.to_excel(writer, sheet_name="Routes", index=False)
            by_day_df.to_excel(writer, sheet_name="By_Day")

    @staticmethod
    def _verify_checkpoint(path: Path) -> None:
        try:
            routes_df = pd.read_excel(path, sheet_name="Routes")
        except Exception as exc:
            print(f"WARNING: checkpoint verification failed: {exc}", flush=True)
            return

        if len(routes_df) >= 1:
            route_count = len(
                routes_df[["day", "vehicle", "trip"]].drop_duplicates()
            )
            print(f"Checkpoint verified: {route_count} routes saved", flush=True)
        else:
            print("WARNING: checkpoint Routes sheet is empty", flush=True)

    @staticmethod
    def _parse_incumbent_objective(line: str) -> float | None:
        patterns = [
            r"MIPStart provided solution with cost\s+([-+]?\d+(?:\.\d+)?)",
            r"Integer solution of\s+([-+]?\d+(?:\.\d+)?)\s+found",
        ]
        for pattern in patterns:
            match = re.search(pattern, line)
            if match:
                return float(match.group(1))
        return None

    @staticmethod
    def _parse_gap_percent(line: str) -> float | None:
        match = re.search(r"Gap:\s*([-+]?\d+(?:\.\d+)?)", line)
        if match:
            return float(match.group(1)) * 100
        return None

    @staticmethod
    def _variable_value(variable: Any) -> float | None:
        try:
            result = value(variable)
        except Exception:
            result = getattr(variable, "varValue", None)
        return None if result is None else float(result)

    @classmethod
    def _is_selected(cls, variable: Any) -> bool:
        variable_value = cls._variable_value(variable)
        return variable_value is not None and abs(variable_value) > 0.5


MTPVRPSolver = Solver

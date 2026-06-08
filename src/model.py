from __future__ import annotations

from typing import Any

from pulp import LpBinary, LpMinimize, LpProblem, LpVariable, lpSum


Q = 1050
D_MAX = 400
P_MAX = 8
k = 1.0
N_TRUCKS = 2
T = 14
R = 1
STORE_TO_STORE_ARC_MAX_KM = 200


def build(
    data: dict[str, Any],
    days: list[int] | range | None = None,
) -> tuple[LpProblem, dict[str, Any]]:
    depot_id = data["depot"]["id"]
    stores = data["stores"]
    distances = data["distances"]

    store_ids = [store["id"] for store in stores]
    nodes = [depot_id, *store_ids]
    trucks = range(N_TRUCKS)
    days = list(days) if days is not None else list(range(1, T + 1))
    trips = range(1, R + 1)
    qi = {store["id"]: float(store["qi"]) for store in stores}
    fi = {store["id"]: int(store["fi"]) for store in stores}

    _validate_distances(nodes, distances)
    arcs = _build_filtered_arcs(nodes, store_ids, depot_id, distances)
    arc_set = set(arcs)

    problem = LpProblem("MT_PVRP_Mazatlan", LpMinimize)

    x_keys = [
        (i, j, v, t, r)
        for i, j in arcs
        for v in trucks
        for t in days
        for r in trips
    ]
    y_keys = [
        (i, v, t, r)
        for i in store_ids
        for v in trucks
        for t in days
        for r in trips
    ]
    u_keys = [
        (i, v, t, r)
        for i in store_ids
        for v in trucks
        for t in days
        for r in trips
    ]
    x = LpVariable.dicts("x", x_keys, lowBound=0, upBound=1, cat=LpBinary)
    y = LpVariable.dicts("y", y_keys, lowBound=0, upBound=1, cat=LpBinary)
    u = LpVariable.dicts("u", u_keys, lowBound=0)

    # eq.1: minimize total distance cost
    problem += (
        k
        * lpSum(
            distances[i][j] * x[(i, j, v, t, r)]
            for i, j in arcs
            for v in trucks
            for t in days
            for r in trips
        ),
        "eq_1_total_distance_cost",
    )

    # eq.2: flow in = visit activation
    for j in store_ids:
        incoming = [i for i in nodes if (i, j) in arc_set]
        for v in trucks:
            for t in days:
                for r in trips:
                    problem += (
                        lpSum(x[(i, j, v, t, r)] for i in incoming)
                        == y[(j, v, t, r)]
                    )

    # eq.3: flow out = visit activation
    for i in store_ids:
        outgoing = [j for j in nodes if (i, j) in arc_set]
        for v in trucks:
            for t in days:
                for r in trips:
                    problem += (
                        lpSum(x[(i, j, v, t, r)] for j in outgoing)
                        == y[(i, v, t, r)]
                    )

    # eq.4: depot return
    depot_out_nodes = [j for j in nodes if (depot_id, j) in arc_set]
    depot_in_nodes = [i for i in nodes if (i, depot_id) in arc_set]
    for v in trucks:
        for t in days:
            for r in trips:
                depot_out = lpSum(x[(depot_id, j, v, t, r)] for j in depot_out_nodes)
                depot_in = lpSum(x[(i, depot_id, v, t, r)] for i in depot_in_nodes)
                problem += depot_out == depot_in
                problem += depot_out <= 1

    # eq.5: hard max km per route
    for v in trucks:
        for t in days:
            for r in trips:
                problem += (
                    lpSum(distances[i][j] * x[(i, j, v, t, r)] for i, j in arcs)
                    <= D_MAX
                )

    _add_frequency_constraints(problem, y, store_ids, fi, trucks, days, trips)

    # eq.10: vehicle capacity per trip
    for v in trucks:
        for t in days:
            for r in trips:
                problem += lpSum(qi[i] * y[(i, v, t, r)] for i in store_ids) <= Q

    # eq.11: maximum stops per route
    for v in trucks:
        for t in days:
            for r in trips:
                problem += lpSum(y[(i, v, t, r)] for i in store_ids) <= P_MAX

    # eq.12: MTZ subtour elimination
    for i, j in arcs:
        if i not in store_ids or j not in store_ids:
            continue
        for v in trucks:
            for t in days:
                for r in trips:
                    problem += (
                        u[(i, v, t, r)]
                        - u[(j, v, t, r)]
                        + P_MAX * x[(i, j, v, t, r)]
                        <= P_MAX - 1
                    )

    return problem, {
        "x": x,
        "y": y,
        "u": u,
        "arcs": arcs,
        "nodes": nodes,
        "store_ids": store_ids,
        "trucks": list(trucks),
        "days": list(days),
        "trips": list(trips),
    }


def _validate_distances(
    nodes: list[str],
    distances: dict[str, dict[str, float]],
) -> None:
    missing_arcs = []
    for i in nodes:
        if i not in distances:
            missing_arcs.extend((i, j) for j in nodes)
            continue
        for j in nodes:
            if j not in distances[i]:
                missing_arcs.append((i, j))

    if missing_arcs:
        sample = ", ".join(f"{i}->{j}" for i, j in missing_arcs[:5])
        raise ValueError(f"Distance matrix is missing required arcs: {sample}")


def _add_frequency_constraints(
    problem: LpProblem,
    y: dict[tuple[str, int, int, int], Any],
    store_ids: list[str],
    fi: dict[str, int],
    trucks: range,
    days: list[int],
    trips: range,
) -> None:
    day_set = set(days)
    is_week_1 = day_set.issubset(set(range(1, 8)))
    is_week_2 = day_set.issubset(set(range(8, T + 1)))

    for i in store_ids:
        visits_in_window = lpSum(
            y[(i, v, t, r)] for t in days for v in trucks for r in trips
        )

        if fi[i] == 1:
            if is_week_2:
                continue
            # eq.7: quincenal stores receive one visit in week 1 / active window
            problem += visits_in_window == 1
        elif fi[i] == 2:
            if is_week_1:
                # eq.8: semanal stores receive one visit in week 1
                problem += visits_in_window == 1
            elif is_week_2:
                # eq.9: semanal stores receive one visit in week 2
                problem += visits_in_window == 1
            else:
                # eq.8/eq.9: full-horizon weekly stores receive one visit per half
                problem += (
                    lpSum(
                        y[(i, v, t, r)]
                        for t in days
                        if 1 <= t <= 7
                        for v in trucks
                        for r in trips
                    )
                    == 1
                )
                problem += (
                    lpSum(
                        y[(i, v, t, r)]
                        for t in days
                        if 8 <= t <= T
                        for v in trucks
                        for r in trips
                    )
                    == 1
                )


def _build_filtered_arcs(
    nodes: list[str],
    store_ids: list[str],
    depot_id: str,
    distances: dict[str, dict[str, float]],
) -> list[tuple[str, str]]:
    store_set = set(store_ids)
    arcs = []
    for i in nodes:
        for j in nodes:
            if i == j:
                continue
            if (
                i in store_set
                and j in store_set
                and distances[i][j] > STORE_TO_STORE_ARC_MAX_KM
            ):
                continue
            arcs.append((i, j))

    arc_set = set(arcs)
    for store_id in store_ids:
        forced = []
        if (depot_id, store_id) not in arc_set:
            forced.append((depot_id, store_id))
        if (store_id, depot_id) not in arc_set:
            forced.append((store_id, depot_id))

        if forced:
            print(
                "Warning: forcing depot arcs for isolated store "
                f"{store_id}: {forced}",
                flush=True,
            )
            arcs.extend(forced)
            arc_set.update(forced)

    return arcs


class MTPVRPModel:
    """Factory wrapper for the MT-PVRP PuLP model."""

    build = staticmethod(build)

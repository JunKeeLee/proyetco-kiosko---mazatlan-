from __future__ import annotations

from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
CHECKPOINT_PATH = BASE_DIR / "outputs" / "checkpoint_week1.xlsx"
DATA_PATH = BASE_DIR / "data" / "CROSS_DOCK_MAZATLÁN.csv"
DISTANCE_PATH = BASE_DIR / "data" / "matriz_distancias_mazatlan.csv"
DEPOT_ID = "CROSS_DOCK_MAZATLAN"
STORE_TO_DEBUG = "31001"


def main() -> None:
    routes = pd.read_excel(CHECKPOINT_PATH, sheet_name="Routes")
    stores = load_expected_week1_stores()
    expected_stores = set(stores["ID_TIENDA"].astype(str))

    if routes.empty:
        print("Routes sheet is empty.")
        print(f"Stores NOT appearing in routes: {', '.join(sorted(expected_stores))}")
        print_store_debug(stores, load_distances(), STORE_TO_DEBUG)
        return

    route_groups = routes.groupby(["day", "vehicle", "trip"], dropna=False)
    route_km = route_groups["cumulative_km"].max()
    total_distance_km = route_km.sum()
    n_routes = len(route_km)
    average_km_per_route = route_km.mean() if n_routes else 0.0
    longest_route = route_km.idxmax()
    shortest_route = route_km.idxmin()
    routed_stores = set(routes["store_id"].astype(str))
    missing_stores = sorted(expected_stores - routed_stores)

    print(f"Total distance km: {total_distance_km:.2f}")
    print(f"Number of routes used: {n_routes}")
    print(f"Average km per route: {average_km_per_route:.2f}")
    print(
        "Longest route: "
        f"day {longest_route[0]}, vehicle {longest_route[1]}, trip {longest_route[2]} "
        f"({route_km.loc[longest_route]:.2f} km)"
    )
    print(
        "Shortest route: "
        f"day {shortest_route[0]}, vehicle {shortest_route[1]}, trip {shortest_route[2]} "
        f"({route_km.loc[shortest_route]:.2f} km)"
    )
    if missing_stores:
        print(f"Stores NOT appearing in routes: {', '.join(missing_stores)}")
    else:
        print("Stores NOT appearing in routes: none")
    print()
    print("Route details:")
    print_route_details(route_groups, route_km)
    print()
    print_store_debug(stores, load_distances(), STORE_TO_DEBUG)


def load_expected_week1_stores() -> pd.DataFrame:
    stores = pd.read_csv(DATA_PATH, dtype={"ID_TIENDA": str})
    stores = stores[stores["ROL_DE_ENTREGA"].astype(str).str.upper() != "CROSS"].copy()
    stores = stores.drop_duplicates(subset="ID_TIENDA", keep="first")
    return stores


def load_distances() -> dict[str, dict[str, float]]:
    matrix = pd.read_csv(DISTANCE_PATH)
    matrix.columns = [str(column).strip() for column in matrix.columns]
    row_id_column = matrix.columns[0]
    matrix[row_id_column] = matrix[row_id_column].astype(str).replace(
        {"CROSS": DEPOT_ID}
    )
    matrix = matrix.rename(columns={"CROSS": DEPOT_ID})

    distances: dict[str, dict[str, float]] = {}
    for _, row in matrix.iterrows():
        origin = str(row[row_id_column])
        distances[origin] = {
            str(destination): float(row[destination])
            for destination in matrix.columns[1:]
        }
    return distances


def print_route_details(
    route_groups: pd.core.groupby.DataFrameGroupBy,
    route_km: pd.Series,
) -> None:
    for key, group in route_groups:
        day, vehicle, trip = key
        ordered = group.sort_values("stop_order")
        stops = [
            f"{int(row.stop_order)}. [{row.store_id} {row.store_name}] "
            f"boxes={float(row.demand_boxes):.2f}, cumulative_km={float(row.cumulative_km):.2f}"
            for row in ordered.itertuples(index=False)
        ]
        print(
            f"Day {day}, Vehicle {vehicle}, Trip {trip} "
            f"({route_km.loc[key]:.2f} km):"
        )
        for stop in stops:
            print(f"  {stop}")


def print_store_debug(
    stores: pd.DataFrame,
    distances: dict[str, dict[str, float]],
    store_id: str,
) -> None:
    store_rows = stores[stores["ID_TIENDA"].astype(str) == store_id]
    print(f"Store {store_id} diagnostic:")
    if store_rows.empty:
        print("  Store not found in CROSS_DOCK_MAZATLÁN.csv")
        return

    store = store_rows.iloc[0]
    demand = float(store["DEMANDA_POR_VIAJE"])
    outbound = distances.get(DEPOT_ID, {}).get(store_id)
    inbound = distances.get(store_id, {}).get(DEPOT_ID)
    if outbound is None or inbound is None:
        print(f"  Demand boxes: {demand:.2f}")
        print("  Depot distance: missing from distance matrix")
        return

    print(f"  Name: {store['TIENDA']}")
    print(f"  Demand boxes: {demand:.2f}")
    print(f"  Depot -> store km: {outbound:.2f}")
    print(f"  Store -> depot km: {inbound:.2f}")
    print(f"  Depot round-trip km: {outbound + inbound:.2f}")


if __name__ == "__main__":
    main()

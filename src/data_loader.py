from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


class DataLoader:
    """Load mini-supermarket distribution data for the MT-PVRP model."""

    CROSS_DOCK_FILE = "CROSS_DOCK_MAZATLÁN.csv"
    DISTANCE_MATRIX_FILE = "matriz_distancias_mazatlan.csv"
    DEPOT_ID = "CROSS_DOCK_MAZATLAN"
    DEPOT_NAME = "Cross Dock Mazatlan"

    def __init__(
        self,
        data_dir: str | Path | None = None,
        cross_dock_file: str | None = None,
        distance_matrix_file: str | None = None,
    ) -> None:
        if data_dir is None:
            data_dir = Path(__file__).parent.parent / "data"
        self.data_dir = Path(data_dir).resolve()
        self.cross_dock_file = cross_dock_file or self.CROSS_DOCK_FILE
        self.distance_matrix_file = distance_matrix_file or self.DISTANCE_MATRIX_FILE

    def load(self) -> dict[str, Any]:
        cross_dock_df = self._read_cross_dock_file()
        depot_row = self._get_depot_row(cross_dock_df)
        stores = self._extract_stores(cross_dock_df)
        distances = self._read_distance_matrix()

        data = {
            "depot": {
                "id": self.DEPOT_ID,
                "name": self.DEPOT_NAME,
            },
            "stores": stores,
            "distances": distances,
        }

        self._print_summary(stores, distances)
        return data

    def _read_cross_dock_file(self) -> pd.DataFrame:
        path = self.data_dir / self.cross_dock_file
        return pd.read_csv(path, dtype={"ID_TIENDA": str})

    def _get_depot_row(self, df: pd.DataFrame) -> pd.Series:
        cross_rows = df[df["ROL_DE_ENTREGA"].astype(str).str.upper() == "CROSS"]
        if cross_rows.empty:
            raise ValueError("No depot row found where ROL_DE_ENTREGA == 'CROSS'.")
        return cross_rows.iloc[0]

    def _extract_stores(self, df: pd.DataFrame) -> list[dict[str, Any]]:
        stores_df = df[df["ROL_DE_ENTREGA"].astype(str).str.upper() != "CROSS"].copy()
        stores_df["ID_TIENDA"] = stores_df["ID_TIENDA"].astype(str)
        stores_df = stores_df.drop_duplicates(subset="ID_TIENDA", keep="first")

        stores: list[dict[str, Any]] = []
        for _, row in stores_df.iterrows():
            stores.append(
                {
                    "id": str(row["ID_TIENDA"]),
                    "name": row["TIENDA"],
                    "lat": float(row["LATITUD"]),
                    "lon": float(row["LONGITUD"]),
                    "qi": float(row["DEMANDA_POR_VIAJE"]),
                    "fi": self._frequency_to_visits(row["FRECUENCIA"]),
                }
            )

        return stores

    def _frequency_to_visits(self, frequency: Any) -> int:
        value = float(frequency)
        if value == 14:
            return 1
        if value in (7, 3.5):
            return 2
        raise ValueError(f"Unsupported FRECUENCIA value: {frequency!r}")

    def _read_distance_matrix(self) -> dict[str, dict[str, float]]:
        path = self.data_dir / self.distance_matrix_file
        matrix_df = pd.read_csv(path)

        if matrix_df.empty and len(matrix_df.columns) == 0:
            raise ValueError("Distance matrix file has no columns.")

        actual_first_column = str(matrix_df.columns[0])
        print(f"Distance matrix first column found: {actual_first_column}", flush=True)

        matrix_df.columns = [str(column).strip() for column in matrix_df.columns]
        row_id_column = matrix_df.columns[0]
        matrix_df[row_id_column] = matrix_df[row_id_column].astype(str)
        matrix_df[row_id_column] = matrix_df[row_id_column].replace(
            {"CROSS": self.DEPOT_ID}
        )
        matrix_df = matrix_df.rename(columns={"CROSS": self.DEPOT_ID})
        distance_columns = [str(column) for column in matrix_df.columns[1:]]

        distances: dict[str, dict[str, float]] = {}
        for _, row in matrix_df.iterrows():
            origin = str(row[row_id_column])
            distances[origin] = {
                destination: float(row[destination]) for destination in distance_columns
            }

        return distances

    def _print_summary(
        self,
        stores: list[dict[str, Any]],
        distances: dict[str, dict[str, float]],
    ) -> None:
        total_stores = len(stores)
        weekly_stores = sum(1 for store in stores if store["fi"] == 2)
        biweekly_stores = sum(1 for store in stores if store["fi"] == 1)
        demands = [store["qi"] for store in stores]

        if demands:
            min_demand = min(demands)
            max_demand = max(demands)
            avg_demand = sum(demands) / len(demands)
        else:
            min_demand = max_demand = avg_demand = 0.0

        row_count = len(distances)
        column_count = max((len(row) for row in distances.values()), default=0)
        round_trips = [
            (
                store["id"],
                distances[self.DEPOT_ID][store["id"]]
                + distances[store["id"]][self.DEPOT_ID],
            )
            for store in stores
            if self.DEPOT_ID in distances
            and store["id"] in distances[self.DEPOT_ID]
            and store["id"] in distances
            and self.DEPOT_ID in distances[store["id"]]
        ]
        max_round_trip_store, max_round_trip_km = max(
            round_trips,
            key=lambda item: item[1],
            default=("", 0.0),
        )

        print(f"Total unique stores: {total_stores}", flush=True)
        print(f"Weekly stores (fi=2): {weekly_stores}", flush=True)
        print(f"Biweekly stores (fi=1): {biweekly_stores}", flush=True)
        print(
            "Min/max/avg demand per delivery (boxes): "
            f"{min_demand:.2f} / {max_demand:.2f} / {avg_demand:.2f}",
            flush=True,
        )
        print(f"Distance matrix size: {row_count} x {column_count}", flush=True)
        print(
            "Max depot round-trip distance: "
            f"{max_round_trip_km:.2f} km"
            + (f" (store {max_round_trip_store})" if max_round_trip_store else ""),
            flush=True,
        )

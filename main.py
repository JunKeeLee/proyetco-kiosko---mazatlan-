from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

import pandas as pd

from src.data_loader import DataLoader
from src.heuristic import GreedyConstructor, validate_warm_start
from src.model import build
from src.solver import Solver


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Solve the Mazatlan MT-PVRP instance.")
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Use N random stores for a smaller test instance. Default: all stores.",
    )
    parser.add_argument(
        "--timelimit",
        type=int,
        default=300,
        help="CBC time limit in seconds. Default: 300.",
    )
    parser.add_argument(
        "--mipgap",
        type=float,
        default=0.05,
        help="CBC relative MIP gap. Default: 0.05.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used with --sample. Default: 42.",
    )
    parser.add_argument(
        "--output",
        default="outputs/solution_mazatlan.xlsx",
        help="Excel output path. Default: outputs/solution_mazatlan.xlsx.",
    )
    parser.add_argument(
        "--heuristic",
        action="store_true",
        help="Use greedy warm start. Default: enabled.",
    )
    parser.add_argument(
        "--no-heuristic",
        action="store_false",
        dest="heuristic",
        help="Disable greedy warm start.",
    )
    parser.add_argument(
        "--decompose",
        action="store_true",
        help="Solve week 1 and week 2 as sequential rolling-horizon sub-problems.",
    )
    parser.add_argument(
        "--week1only",
        action="store_true",
        help="Run only the week 1 rolling-horizon solve and save its checkpoint.",
    )
    parser.add_argument(
        "--week2only",
        action="store_true",
        help=(
            "Skip week 1, load outputs/checkpoint_week1.xlsx as the week 1 result, "
            "then run only the week 2 rolling-horizon solve."
        ),
    )
    parser.set_defaults(heuristic=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = Path(__file__).resolve().parent

    loader = DataLoader(data_dir=base_dir / "data")
    data = loader.load()
    data = sample_data(data, args.sample, args.seed)

    if args.week1only and args.week2only:
        raise ValueError("--week1only and --week2only cannot be used together.")

    if args.decompose or args.week1only or args.week2only:
        run_decomposed(data, args, base_dir)
    else:
        solve_instance(
            label="Full horizon",
            data=data,
            days=list(range(1, 15)),
            args=args,
            output_path=base_dir / args.output,
            checkpoint_path=base_dir / "outputs" / "checkpoint_best.xlsx",
        )


def run_decomposed(data: dict[str, Any], args: argparse.Namespace, base_dir: Path) -> None:
    week_1_path = base_dir / "outputs" / "solution_week1.xlsx"
    week_1_checkpoint_path = base_dir / "outputs" / "checkpoint_week1.xlsx"
    week_2_path = base_dir / "outputs" / "solution_week2.xlsx"
    week_2_checkpoint_path = base_dir / "outputs" / "checkpoint_week2.xlsx"
    final_path = base_dir / "outputs" / "solution_final.xlsx"

    week_1_result_path = week_1_path
    if args.week2only:
        if not week_1_checkpoint_path.exists():
            raise FileNotFoundError(
                "Cannot run --week2only because outputs/checkpoint_week1.xlsx "
                "does not exist."
            )
        week_1_result_path = week_1_checkpoint_path
        print(f"Using existing Week 1 checkpoint: {week_1_result_path}", flush=True)
    else:
        solve_instance(
            label="Week 1 rolling horizon",
            data=data,
            days=list(range(1, 8)),
            args=args,
            output_path=week_1_path,
            checkpoint_path=week_1_checkpoint_path,
        )
        if args.week1only:
            print(f"Week 1 checkpoint saved to: {week_1_checkpoint_path}", flush=True)
            return

    weekly_data = filter_stores(data, lambda store: int(store["fi"]) == 2)
    solve_instance(
        label="Week 2 rolling horizon",
        data=weekly_data,
        days=list(range(8, 15)),
        args=args,
        output_path=week_2_path,
        checkpoint_path=week_2_checkpoint_path,
    )

    merge_weekly_outputs(week_1_result_path, week_2_path, final_path)
    print(f"Final decomposed solution exported to: {final_path}", flush=True)


def solve_instance(
    label: str,
    data: dict[str, Any],
    days: list[int],
    args: argparse.Namespace,
    output_path: Path,
    checkpoint_path: Path,
) -> Solver:
    print(f"{label}: days {days[0]}-{days[-1]}", flush=True)
    prob, variables = build(data, days=days)
    print_variable_counts(prob, variables)

    warm_start = None
    if args.heuristic:
        constructor = GreedyConstructor(data, variables)
        warm_start = constructor.build()
        print(
            "Greedy solution cost: "
            f"{warm_start['cost']:.2f} "
            f"(distance={warm_start['distance_cost']:.2f})",
            flush=True,
        )
        if warm_start["unassigned"]:
            print(
                "Greedy unassigned stores: "
                + ", ".join(str(store_id) for store_id in warm_start["unassigned"]),
                flush=True,
            )
        violations = validate_warm_start(warm_start, data, variables)
        print(f"Warm start validation summary: {len(violations)} violations", flush=True)

    solver = Solver(prob, variables, data)
    status = solver.solve(
        time_limit=args.timelimit,
        mip_gap=args.mipgap,
        warm_start=warm_start,
        stream_progress=True,
        checkpoint_path=checkpoint_path,
    )
    solver.print_route_summary()
    exported_path = solver.export(output_path)

    print(f"Solver status: {status}", flush=True)
    print(f"Solve time (sec): {solver.solve_time_sec:.2f}", flush=True)
    print(f"Solution exported to: {exported_path}", flush=True)
    return solver


def sample_data(
    data: dict[str, Any],
    sample_size: int | None,
    seed: int,
) -> dict[str, Any]:
    if sample_size is None:
        return data

    if sample_size < 1:
        raise ValueError("--sample must be a positive integer.")

    stores = data["stores"]
    if sample_size >= len(stores):
        return data

    rng = random.Random(seed)
    sampled_stores = rng.sample(stores, sample_size)
    print(f"Sampled stores: {sample_size} of {len(stores)}", flush=True)
    return filter_stores(
        data,
        lambda store: store in sampled_stores,
    )


def filter_stores(
    data: dict[str, Any],
    predicate: Any,
) -> dict[str, Any]:
    selected_stores = [store for store in data["stores"] if predicate(store)]
    selected_nodes = {data["depot"]["id"], *[store["id"] for store in selected_stores]}
    selected_distances = {
        origin: {
            destination: distance
            for destination, distance in destinations.items()
            if destination in selected_nodes
        }
        for origin, destinations in data["distances"].items()
        if origin in selected_nodes
    }

    return {
        "depot": data["depot"],
        "stores": selected_stores,
        "distances": selected_distances,
    }


def merge_weekly_outputs(
    week_1_path: Path,
    week_2_path: Path,
    final_path: Path,
) -> None:
    routes_frames = []
    summary_frames = []

    for week_label, path in (("week1", week_1_path), ("week2", week_2_path)):
        routes_df = pd.read_excel(path, sheet_name="Routes")
        routes_df["window"] = week_label
        routes_frames.append(routes_df)

        summary_df = pd.read_excel(path, sheet_name="Summary")
        summary_df["window"] = week_label
        summary_frames.append(summary_df)

    routes = pd.concat(routes_frames, ignore_index=True)
    summaries = pd.concat(summary_frames, ignore_index=True)
    combined_summary = pd.DataFrame(
        [
            {
                "status": "; ".join(summaries["status"].astype(str)),
                "solve_time_sec": summaries["solve_time_sec"].sum(),
                "total_km": summaries["total_km"].sum(),
                "total_cost": summaries["total_cost"].sum(),
                "n_routes_used": summaries["n_routes_used"].sum(),
            }
        ]
    )

    final_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(final_path, engine="openpyxl") as writer:
        combined_summary.to_excel(writer, sheet_name="Summary", index=False)
        routes.to_excel(writer, sheet_name="Routes", index=False)
        summaries.to_excel(writer, sheet_name="Subproblem_Summary", index=False)


def print_variable_counts(prob: Any, variables: dict[str, Any]) -> None:
    counts = {
        name: count_variable_leaves(variables[name])
        for name in ("x", "y", "u")
        if name in variables
    }
    print("Variable counts:", flush=True)
    for name, count in counts.items():
        print(f"  {name}: {count}", flush=True)
    print(f"  total: {sum(counts.values())}", flush=True)
    print(f"Constraint count: {len(prob.constraints)}", flush=True)


def count_variable_leaves(variable: Any) -> int:
    if hasattr(variable, "varValue"):
        return 1
    if isinstance(variable, dict):
        return sum(count_variable_leaves(value) for value in variable.values())
    return 0


if __name__ == "__main__":
    main()

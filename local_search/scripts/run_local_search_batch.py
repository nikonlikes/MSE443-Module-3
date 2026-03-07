"""
Batch runner: execute local_search_simulation on all 90 generated datasets
(30 small + 30 medium + 30 large) and write a simulation_results.csv
matching the format produced by the greedy batch runner.
"""

import os
import csv
import time as time_module

from local_search_simulation import (
    read_csv_raw,
    build_round_robin_queues,
    greedy_tote_sequence,
    local_search,
    compute_makespan,
    write_schedule_csv,
    NUM_CONVEYORS,
)


def build_data_from_files(it_path, qt_path, tt_path):
    """build_data variant that accepts explicit file paths."""
    it_data = read_csv_raw(it_path)
    qt_data = read_csv_raw(qt_path)
    tt_data = read_csv_raw(tt_path)

    orders = {}
    totes = {}

    for order_idx, (it_row, qt_row, tt_row) in enumerate(
        zip(it_data, qt_data, tt_data)
    ):
        order_id = order_idx + 1
        orders[order_id] = {"items": [], "total_items": 0}
        for it, qt, tt in zip(it_row, qt_row, tt_row):
            if it is None or qt is None or tt is None:
                continue
            tote_id = int(tt)
            item_type = int(it)
            quantity = int(qt)
            orders[order_id]["items"].append((item_type, quantity))
            orders[order_id]["total_items"] += quantity
            totes.setdefault(tote_id, []).append(
                {"order": order_id, "item_type": item_type, "quantity": quantity}
            )
    return orders, totes


SAMPLE_CONFIGS = [
    ("small", os.path.join("ranDataGen copy", "small sized samples")),
    ("medium", os.path.join("ranDataGen copy", "medium sized samples")),
    ("large", os.path.join("ranDataGen copy", "large sized samples")),
]

OUTPUT_ROOT = os.path.join("local_search", "outputs")
N_DATASETS = 30


def run_single_dataset(size, dataset_id, folder):
    """Run baseline + local search on one dataset. Returns two result dicts."""
    it_path = os.path.join(folder, f"order_itemtypes_{dataset_id}.csv")
    qt_path = os.path.join(folder, f"order_quantities_{dataset_id}.csv")
    tt_path = os.path.join(folder, f"orders_totes_{dataset_id}.csv")

    if not os.path.exists(it_path):
        print(f"  SKIP {size} #{dataset_id} — files not found")
        return None

    orders, totes = build_data_from_files(it_path, qt_path, tt_path)
    n_orders = len(orders)

    # --- baseline: round-robin queues + sorted tote sequence ---
    init_queues = build_round_robin_queues(orders)
    init_tote_seq = sorted(totes.keys())

    baseline_ms, baseline_res = compute_makespan(
        orders, totes, init_queues, init_tote_seq
    )
    b_ct = baseline_res["order_completion_times"]
    finite_b = [t for t in b_ct.values() if t != float("inf")]
    sum_b = sum(finite_b)
    avg_b = sum_b / len(finite_b) if finite_b else float("inf")

    # --- local search: greedy initial tote sequence → local search ---
    greedy_seq = greedy_tote_sequence(orders, totes, init_queues)

    best_q, best_ts, best_ms, best_res, _ = local_search(
        orders, totes, init_queues, greedy_seq,
        max_iterations=1000, verbose=False,
    )
    ls_ct = best_res["order_completion_times"]
    finite_ls = [t for t in ls_ct.values() if t != float("inf")]
    sum_ls = sum(finite_ls)
    avg_ls = sum_ls / len(finite_ls) if finite_ls else float("inf")

    # --- write per-dataset schedule CSVs ---
    ds_dir = os.path.join(OUTPUT_ROOT, size, f"dataset_{dataset_id}")
    os.makedirs(ds_dir, exist_ok=True)
    write_schedule_csv(best_q, orders, os.path.join(ds_dir, "input_local_search.csv"))
    write_schedule_csv(init_queues, orders, os.path.join(ds_dir, "input_baseline.csv"))

    def fmt_ms(ms):
        return ms if ms != float("inf") else ""

    return [
        {
            "size": size,
            "dataset_id": dataset_id,
            "method": "local_search",
            "makespan": fmt_ms(best_ms),
            "sum_completion_times": sum_ls,
            "avg_completion_time": avg_ls,
            "n_orders": n_orders,
        },
        {
            "size": size,
            "dataset_id": dataset_id,
            "method": "baseline",
            "makespan": fmt_ms(baseline_ms),
            "sum_completion_times": sum_b,
            "avg_completion_time": avg_b,
            "n_orders": n_orders,
        },
    ]


def main():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    all_results = []

    total_start = time_module.time()

    for size, folder in SAMPLE_CONFIGS:
        print(f"\n{'='*60}")
        print(f"  Processing {size} datasets from {folder}")
        print(f"{'='*60}")

        for dataset_id in range(1, N_DATASETS + 1):
            t0 = time_module.time()
            pair = run_single_dataset(size, dataset_id, folder)
            elapsed = time_module.time() - t0

            if pair is None:
                continue

            all_results.extend(pair)
            ls_row, bl_row = pair

            ls_ms = ls_row["makespan"] if ls_row["makespan"] != "" else "INF"
            bl_ms = bl_row["makespan"] if bl_row["makespan"] != "" else "INF"
            print(
                f"  {size:>6} #{dataset_id:<2}  "
                f"local_search={ls_ms}  baseline={bl_ms}  "
                f"({elapsed:.1f}s)"
            )

    # --- write combined CSV ---
    out_path = os.path.join(OUTPUT_ROOT, "simulation_results.csv")
    fieldnames = [
        "size", "dataset_id", "method", "makespan",
        "sum_completion_times", "avg_completion_time", "n_orders",
    ]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)

    total_elapsed = time_module.time() - total_start
    print(f"\n{'='*60}")
    print(f"  Done — {len(all_results)} rows written to {out_path}")
    print(f"  Total time: {total_elapsed:.1f}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

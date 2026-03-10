"""
Batch runner: execute SPT and LPT scheduling heuristics on all 90 generated
datasets (30 small + 30 medium + 30 large) and write simulation_results.csv.
"""

import os
import csv
import pandas as pd
import numpy as np

N_CONVEYORS = 4
N_SHAPE_TYPES = 8
SHAPE_NAMES = ['circle', 'pentagon', 'trapezoid', 'triangle',
               'star', 'moon', 'heart', 'cross']

SAMPLE_CONFIGS = [
    ("small",  os.path.join("ranDataGen copy", "small sized samples")),
    ("medium", os.path.join("ranDataGen copy", "medium sized samples")),
    ("large",  os.path.join("ranDataGen copy", "large sized samples")),
]

OUTPUT_ROOT = os.path.join("scheduling_heuristic", "outputs")
N_DATASETS = 30


def load_dataset(folder, dataset_id):
    it_path = os.path.join(folder, f"order_itemtypes_{dataset_id}.csv")
    qt_path = os.path.join(folder, f"order_quantities_{dataset_id}.csv")
    if not os.path.exists(it_path):
        return None, None
    item_types_df = pd.read_csv(it_path, header=None)
    quantities_df = pd.read_csv(qt_path, header=None)

    orders = []
    processing_times = []
    for i in range(len(item_types_df)):
        types = item_types_df.iloc[i].dropna().astype(int).tolist()
        qtys = quantities_df.iloc[i].dropna().astype(int).tolist()
        orders.append(list(zip(types, qtys)))
        processing_times.append(sum(qtys))
    return orders, processing_times


def schedule_orders(processing_times, n_conveyors, reverse=False):
    sorted_indices = sorted(
        range(len(processing_times)),
        key=lambda i: processing_times[i],
        reverse=reverse,
    )
    loads = [0] * n_conveyors
    assignment = {c: [] for c in range(n_conveyors)}
    for order_idx in sorted_indices:
        min_load = min(loads)
        conveyor = loads.index(min_load)
        assignment[conveyor].append(order_idx)
        loads[conveyor] += processing_times[order_idx]
    return assignment


def compute_metrics(assignment, processing_times, n_conveyors):
    loads = []
    for c in range(n_conveyors):
        loads.append(sum(processing_times[i] for i in assignment[c]))
    makespan = max(loads)
    sum_ct = sum(loads)
    active = [l for l in loads if l > 0]
    avg_ct = sum_ct / len(active) if active else 0
    return makespan, sum_ct, avg_ct


def build_output_df(assignment, orders, n_conveyors):
    rows = []
    for c in range(n_conveyors):
        for order_idx in assignment[c]:
            counts = [0] * N_SHAPE_TYPES
            for item_type, qty in orders[order_idx]:
                counts[item_type] += qty
            rows.append([c + 1] + counts)
    return pd.DataFrame(rows, columns=['conv_num'] + SHAPE_NAMES)


def main():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    all_results = []

    for size, folder in SAMPLE_CONFIGS:
        print(f"\n{'='*60}")
        print(f"  Processing {size} datasets")
        print(f"{'='*60}")

        for dataset_id in range(1, N_DATASETS + 1):
            orders, ptimes = load_dataset(folder, dataset_id)
            if orders is None:
                print(f"  SKIP {size} #{dataset_id} — files not found")
                continue

            n_orders = len(orders)
            ds_dir = os.path.join(OUTPUT_ROOT, size, f"dataset_{dataset_id}")
            os.makedirs(ds_dir, exist_ok=True)

            for method_name, reverse in [("SPT", False), ("LPT", True)]:
                assignment = schedule_orders(ptimes, N_CONVEYORS, reverse=reverse)
                makespan, sum_ct, avg_ct = compute_metrics(
                    assignment, ptimes, N_CONVEYORS
                )
                df_out = build_output_df(assignment, orders, N_CONVEYORS)
                csv_path = os.path.join(ds_dir, f"input_{method_name}.csv")
                df_out.to_csv(csv_path, index=False)

                all_results.append({
                    "size": size,
                    "dataset_id": dataset_id,
                    "method": method_name,
                    "makespan": makespan,
                    "sum_completion_times": sum_ct,
                    "avg_completion_time": avg_ct,
                    "n_orders": n_orders,
                })

            spt_ms = all_results[-2]["makespan"]
            lpt_ms = all_results[-1]["makespan"]
            print(f"  {size:>6} #{dataset_id:<2}  SPT={spt_ms}  LPT={lpt_ms}  n_orders={n_orders}")

    out_path = os.path.join(OUTPUT_ROOT, "simulation_results.csv")
    fieldnames = [
        "size", "dataset_id", "method", "makespan",
        "sum_completion_times", "avg_completion_time", "n_orders",
    ]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)

    print(f"\n{'='*60}")
    print(f"  Done — {len(all_results)} rows written to {out_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

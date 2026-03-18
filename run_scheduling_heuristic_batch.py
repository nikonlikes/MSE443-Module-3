"""
Batch runner: execute SPT and LPT scheduling heuristics on all 90 generated
datasets (30 small + 30 medium + 30 large) and write simulation_results.csv.

SPT and LPT now use the same belt simulation as the Greedy heuristic, producing
real-second completion times comparable across all four methods.

Order-to-conveyor assignment uses the constraint-satisfying greedy graph coloring
(orders sharing a tote must be on different conveyors). Within each conveyor queue:
  - SPT: orders sorted by total items ascending  (shortest processing time first)
  - LPT: orders sorted by total items descending (longest processing time first)

Tote sequence: sorted by tote ID (ascending).
"""

import os
import csv
import sys

# Import the belt simulation from greedy_sim
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'greedy'))
from greedy_sim import (
    load_data_from_csv,
    build_constraint_satisfying_queues,
    greedy_tote_sequence,
    simulate,
)

SAMPLE_CONFIGS = [
    ("small",  os.path.join("ranDataGen copy", "small sized samples")),
    ("medium", os.path.join("ranDataGen copy", "medium sized samples")),
    ("large",  os.path.join("ranDataGen copy", "large sized samples")),
]

OUTPUT_ROOT = os.path.join("scheduling_heuristic", "outputs")
N_DATASETS  = 30


def sort_queues_by_pt(conv_queues, orders, reverse=False):
    """Return new queues with orders within each queue sorted by total_items."""
    return [
        sorted(q, key=lambda oid: orders[oid]['total_items'], reverse=reverse)
        for q in conv_queues
    ]


def main():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    all_results = []

    for size, folder in SAMPLE_CONFIGS:
        print(f"\n{'='*60}")
        print(f"  Processing {size} datasets")
        print(f"{'='*60}")

        for dataset_id in range(1, N_DATASETS + 1):
            try:
                orders, totes = load_data_from_csv(folder, dataset_id)
            except Exception as e:
                print(f"  SKIP {size} #{dataset_id} — {e}")
                continue

            if not orders or not totes:
                print(f"  SKIP {size} #{dataset_id} — empty data")
                continue

            n_orders = len(orders)

            # Constraint-satisfying assignment (shared by SPT and LPT)
            base_queues, _ = build_constraint_satisfying_queues(orders, totes)

            for method_name, reverse in [("SPT", False), ("LPT", True)]:
                queues        = sort_queues_by_pt(base_queues, orders, reverse=reverse)
                tote_sequence = greedy_tote_sequence(orders, totes, queues)
                results       = simulate(tote_sequence, orders, totes, queues)

                makespan = results['makespan']
                sum_ct   = results['sum_completion_times']
                avg_ct   = results['avg_completion_time']

                all_results.append({
                    "size":                 size,
                    "dataset_id":           dataset_id,
                    "method":               method_name,
                    "makespan":             makespan if makespan != float('inf') else '',
                    "sum_completion_times": sum_ct,
                    "avg_completion_time":  avg_ct   if avg_ct   != float('inf') else '',
                    "n_orders":             n_orders,
                })

            spt = all_results[-2]
            lpt = all_results[-1]
            print(f"  {size:>6} #{dataset_id:<2}  "
                  f"SPT={spt['sum_completion_times']:.1f}s  "
                  f"LPT={lpt['sum_completion_times']:.1f}s  "
                  f"n_orders={n_orders}")

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

"""
Batch runner: execute the MIP model (time-based) from MIP.ipynb on all 90
generated datasets.  Writes simulation_results.csv.

NOTE: The free Gurobi license limits model size to 2000 variables / 2000
constraints.  With 4 orders & 19 items the model already has ~6 000 variables,
so most datasets will exceed the limit.  We catch the error and record the
result as failed.
"""

import os
import csv
import time as time_module
import pandas as pd

SHAPE_NAMES = ['circle', 'pentagon', 'trapezoid', 'triangle',
               'star', 'moon', 'heart', 'cross']
N_SHAPE_TYPES = 8

SAMPLE_CONFIGS = [
    ("small",  os.path.join("ranDataGen copy", "small sized samples")),
    ("medium", os.path.join("ranDataGen copy", "medium sized samples")),
    ("large",  os.path.join("ranDataGen copy", "large sized samples")),
]

OUTPUT_ROOT = os.path.join("mip", "outputs")
N_DATASETS = 30
TIME_LIMIT = 300  # seconds per solve


def load_df(folder, dataset_id):
    """Load one dataset into the long-format DataFrame used by MIP.ipynb."""
    base = folder
    it_path = os.path.join(base, f"order_itemtypes_{dataset_id}.csv")
    qt_path = os.path.join(base, f"order_quantities_{dataset_id}.csv")
    tt_path = os.path.join(base, f"orders_totes_{dataset_id}.csv")
    if not os.path.exists(it_path):
        return None

    it_df = pd.read_csv(it_path, header=None)
    qt_df = pd.read_csv(qt_path, header=None)
    tt_df = pd.read_csv(tt_path, header=None)

    rows = []
    for order_idx, (it_row, qt_row, tt_row) in enumerate(
        zip(it_df.itertuples(index=False),
            qt_df.itertuples(index=False),
            tt_df.itertuples(index=False))
    ):
        order_id = order_idx + 1
        for slot, (it, qt, tt) in enumerate(zip(it_row, qt_row, tt_row)):
            if pd.isna(it) or pd.isna(qt) or pd.isna(tt):
                continue
            rows.append({
                'order': order_id, 'item_slot': slot,
                'itemtype': int(it), 'quantity': int(qt), 'tote': int(tt),
            })
    return pd.DataFrame(rows)


def solve_mip(df):
    """
    Build and solve the time-based MIP from MIP.ipynb.
    Returns (completion_time, status_str) or (None, error_msg).
    """
    import gurobipy as gp
    from gurobipy import GRB

    m = gp.Model("MIP_batch")
    m.setParam("OutputFlag", 0)
    m.setParam("TimeLimit", TIME_LIMIT)

    # Build items
    items, tote_items = [], {}
    item_id = 0
    for _, row in df.iterrows():
        i, t, qty = row["itemtype"], row["tote"], int(row["quantity"])
        for _ in range(qty):
            item_id += 1
            item = (item_id, i, t)
            items.append(item)
            tote_items.setdefault(t, []).append(item)

    N = len(items)
    positions = range(1, N + 1)
    totes = list(tote_items.keys())
    tote_size = {t: len(tote_items[t]) for t in totes}
    orders = df["order"].unique()

    demand = {}
    for _, row in df.iterrows():
        o, i, qty = row["order"], row["itemtype"], int(row["quantity"])
        demand[(o, i)] = demand.get((o, i), 0) + qty

    # --- Variables ---
    x = m.addVars(
        [(iid, p) for (iid, i, t) in items for p in positions],
        vtype=GRB.BINARY, name="x",
    )
    m.addConstrs(
        gp.quicksum(x[iid, p] for p in positions) == 1
        for (iid, i, t) in items
    )
    m.addConstrs(
        gp.quicksum(x[iid, p] for (iid, i, t) in items) == 1
        for p in positions
    )

    # Tote contiguity
    start = m.addVars(totes, vtype=GRB.INTEGER, lb=1, ub=N, name="tote_start")
    pos_expr = {
        iid: gp.quicksum(p * x[iid, p] for p in positions)
        for (iid, i, t) in items
    }
    for (iid, i, t) in items:
        m.addConstr(pos_expr[iid] >= start[t])
        m.addConstr(pos_expr[iid] <= start[t] + tote_size[t] - 1)

    z = m.addVars(totes, totes, vtype=GRB.BINARY, name="tote_order")
    M = N
    for t1 in totes:
        for t2 in totes:
            if t1 != t2:
                m.addConstr(start[t1] + tote_size[t1] <= start[t2] + M * (1 - z[t1, t2]))
                m.addConstr(start[t2] + tote_size[t2] <= start[t1] + M * z[t1, t2])

    # Belt assignment
    belts = range(1, 5)
    y = m.addVars(
        [(b, o) for b in belts for o in orders],
        vtype=GRB.BINARY, name="assign",
    )
    m.addConstrs(gp.quicksum(y[b, o] for o in orders) == 1 for b in belts)
    m.addConstrs(gp.quicksum(y[b, o] for b in belts) == 1 for o in orders)

    # Picking
    pick = m.addVars(
        [(b, iid, o, p)
         for b in belts for (iid, i, t) in items
         for o in orders for p in positions],
        vtype=GRB.BINARY, name="pick",
    )
    m.addConstrs(
        pick[b, iid, o, p] <= x[iid, p]
        for b in belts for (iid, i, t) in items for o in orders for p in positions
    )
    m.addConstrs(
        gp.quicksum(pick[b, iid, o, p] for b in belts for o in orders) <= 1
        for (iid, i, t) in items for p in positions
    )
    m.addConstrs(
        pick[b + 1, iid, o, p] <=
        1 - gp.quicksum(pick[b, iid, o2, p] for o2 in orders)
        for b in range(1, 4) for (iid, i, t) in items for o in orders for p in positions
    )
    m.addConstrs(
        pick[b, iid, o, p] <= y[b, o]
        for b in belts for (iid, i, t) in items for o in orders for p in positions
    )
    for (iid, i, t) in items:
        for o in orders:
            if (o, i) not in demand:
                for b in belts:
                    for p in positions:
                        m.addConstr(pick[b, iid, o, p] == 0)
    for (o, i), qty in demand.items():
        m.addConstr(
            gp.quicksum(
                pick[b, iid, o, p]
                for b in belts for (iid, it, t) in items if it == i
                for p in positions
            ) == qty
        )

    # Time-based objective
    TIME = m.addVar(vtype=GRB.CONTINUOUS, name="completion_time")
    m.addConstrs(
        TIME >= (3 * (p - 1) + 7.5 * (b - 1) + 3) * pick[b, iid, o, p]
        for b in belts for (iid, i, t) in items for o in orders for p in positions
    )
    m.setObjective(TIME, GRB.MINIMIZE)

    try:
        m.optimize()
    except Exception as e:
        return None, f"solve_error: {e}"

    if m.status == GRB.OPTIMAL:
        return TIME.X, "optimal"
    elif m.status == GRB.TIME_LIMIT and m.SolCount > 0:
        return TIME.X, "time_limit"
    else:
        return None, f"status_{m.status}"


def build_picklist_csv(df, solve_result, output_path):
    """Placeholder — only written when the MIP solves."""
    pass


def main():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    all_results = []

    for size, folder in SAMPLE_CONFIGS:
        print(f"\n{'='*60}")
        print(f"  Processing {size} datasets")
        print(f"{'='*60}")

        for dataset_id in range(1, N_DATASETS + 1):
            df = load_df(folder, dataset_id)
            if df is None:
                print(f"  SKIP {size} #{dataset_id} — files not found")
                continue

            n_orders = df["order"].nunique()
            n_items = df["quantity"].sum()

            t0 = time_module.time()
            try:
                completion_time, status = solve_mip(df)
            except Exception as e:
                completion_time, status = None, str(e)
            elapsed = time_module.time() - t0

            ds_dir = os.path.join(OUTPUT_ROOT, size, f"dataset_{dataset_id}")
            os.makedirs(ds_dir, exist_ok=True)

            result = {
                "size": size,
                "dataset_id": dataset_id,
                "method": "MIP",
                "makespan": completion_time if completion_time is not None else "",
                "sum_completion_times": completion_time if completion_time is not None else "",
                "avg_completion_time": completion_time if completion_time is not None else "",
                "n_orders": n_orders,
                "status": status,
            }
            all_results.append(result)

            ms_str = f"{completion_time:.1f}" if completion_time is not None else "FAIL"
            print(
                f"  {size:>6} #{dataset_id:<2}  "
                f"MIP={ms_str}  status={status}  "
                f"items={int(n_items)}  ({elapsed:.1f}s)"
            )

    out_path = os.path.join(OUTPUT_ROOT, "simulation_results.csv")
    fieldnames = [
        "size", "dataset_id", "method", "makespan",
        "sum_completion_times", "avg_completion_time", "n_orders", "status",
    ]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)

    solved = sum(1 for r in all_results if r["makespan"] != "")
    print(f"\n{'='*60}")
    print(f"  Done — {len(all_results)} rows, {solved} solved")
    print(f"  Results: {out_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

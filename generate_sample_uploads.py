"""
Generate sample datasets for the "Run on Your Data" dashboard page.
Produces three sets of files (small, medium, large) in sample_uploads/.

Each set contains:
  order_itemtypes.csv  — item type (0-7) per slot per order
  order_quantities.csv — quantity per slot per order
  orders_totes.csv     — tote ID per slot per order
"""

import csv
import os
import random

NUM_ITEM_TYPES = 8
OUT_DIR = "sample_uploads"

SIZES = {
    "small":  {"n_orders": 8,   "max_slots": 3, "max_qty": 4, "share_rate": 0.35, "seed": 1},
    "medium": {"n_orders": 20,  "max_slots": 4, "max_qty": 5, "share_rate": 0.30, "seed": 2},
    "large":  {"n_orders": 50,  "max_slots": 5, "max_qty": 6, "share_rate": 0.25, "seed": 3},
}


def generate(n_orders, max_slots, max_qty, share_rate, seed):
    rng = random.Random(seed)

    # Build orders: each order has 1..max_slots distinct item types with random quantities
    orders = []
    for _ in range(n_orders):
        n_slots = rng.randint(1, max_slots)
        item_types = rng.sample(range(NUM_ITEM_TYPES), n_slots)
        slots = [(it, rng.randint(1, max_qty)) for it in item_types]
        orders.append(slots)

    # Assign each slot its own unique tote to start
    next_tote = [1]
    tote_assignments = []
    for slots in orders:
        tote_assignments.append([next_tote[0] + i for i in range(len(slots))])
        next_tote[0] += len(slots)

    # Track which orders are in each tote
    tote_orders = {}
    for oi, totes in enumerate(tote_assignments):
        for tid in totes:
            tote_orders[tid] = {oi}

    # Merge totes across orders to create shared totes (max 4 orders per tote)
    all_slots = [(oi, si) for oi, slots in enumerate(orders) for si in range(len(slots))]
    n_merges = int(len(all_slots) * share_rate)

    for _ in range(n_merges):
        s1, s2 = rng.sample(all_slots, 2)
        oi1, si1 = s1
        oi2, si2 = s2
        if oi1 == oi2:
            continue
        tid1 = tote_assignments[oi1][si1]
        tid2 = tote_assignments[oi2][si2]
        if tid1 == tid2:
            continue
        combined = tote_orders.get(tid1, set()) | tote_orders.get(tid2, set())
        if len(combined) > 4:
            continue
        # Remap all slots pointing to tid2 → tid1
        for oi, totes in enumerate(tote_assignments):
            for si in range(len(totes)):
                if tote_assignments[oi][si] == tid2:
                    tote_assignments[oi][si] = tid1
        tote_orders[tid1] = combined
        tote_orders.pop(tid2, None)

    # Renumber totes sequentially from 1
    seen, counter = {}, [1]
    for oi, totes in enumerate(tote_assignments):
        for si in range(len(totes)):
            old = tote_assignments[oi][si]
            if old not in seen:
                seen[old] = counter[0]
                counter[0] += 1
            tote_assignments[oi][si] = seen[old]

    return orders, tote_assignments


def write_csvs(orders, tote_assignments, folder):
    os.makedirs(folder, exist_ok=True)
    max_slots = max(len(s) for s in orders)

    it_rows, qt_rows, tt_rows = [], [], []
    for slots, totes in zip(orders, tote_assignments):
        it_row = [str(it) if i < len(slots) else '' for i, (it, _) in
                  enumerate(slots + [(0, 0)] * (max_slots - len(slots)))]
        qt_row = [str(qty) if i < len(slots) else '' for i, (_, qty) in
                  enumerate(slots + [(0, 0)] * (max_slots - len(slots)))]
        tt_row = [str(tid) if i < len(totes) else '' for i, tid in
                  enumerate(totes + [0] * (max_slots - len(totes)))]
        it_rows.append([v for v in it_row[:max_slots]])
        qt_rows.append([v for v in qt_row[:max_slots]])
        tt_rows.append([v for v in tt_row[:max_slots]])

    for fname, rows in [
        ("order_itemtypes.csv", it_rows),
        ("order_quantities.csv", qt_rows),
        ("orders_totes.csv",    tt_rows),
    ]:
        with open(os.path.join(folder, fname), "w", newline="") as f:
            csv.writer(f).writerows(rows)


def main():
    for size, cfg in SIZES.items():
        orders, totes = generate(
            cfg["n_orders"], cfg["max_slots"],
            cfg["max_qty"],  cfg["share_rate"], cfg["seed"]
        )
        folder = os.path.join(OUT_DIR, size)
        write_csvs(orders, totes, folder)

        n_totes     = len({tid for row in totes for tid in row})
        shared      = sum(1 for tid in set(tid for row in totes for tid in row)
                         if sum(row.count(tid) for row in totes) > 1)
        total_items = sum(qty for slots in orders for _, qty in slots)

        print(f"{size:>6}: {len(orders)} orders | {n_totes} totes "
              f"({shared} shared) | {total_items} total items  → {folder}/")

    print(f"\nDone. Upload any folder's 3 CSV files at localhost:8501 → 'Run on Your Data'")


if __name__ == "__main__":
    main()

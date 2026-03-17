    """
MSE 433 - Module 3: Warehousing
Greedy Marginal Gain Heuristic — Tote Sequencing Simulation

=== BELT TIMING MODEL ===
  Layout (clockwise from LOAD):
    LOAD (top-left corner)
      → 3.5s → Conv 0 (midpoint of top belt)
      → 9.5s → corner → 3.5s → Conv 1 (midpoint of right belt)
      → 9.5s → corner → 3.5s → Conv 2 (midpoint of bottom belt)
      → 9.5s → corner → 3.5s → Conv 3 (midpoint of left belt)

  Travel time from LOAD to each conveyor:
    Conv 0:  3.5s
    Conv 1: 16.5s
    Conv 2: 29.5s
    Conv 3: 42.5s

  Item placement timing:
    - Items placed one at a time with 4.5s buffer between consecutive items
    - First item of each tote also has 4.5s buffer from last item of previous tote
      (i.e., there is NO extra inter-tote gap beyond the standard 4.5s item spacing)
    - Item placed at belt_time T going to Conv c arrives at T + TRAVEL[c]

  Tote clearance: next item can be placed 4.5s after the previous one.
  The belt "clears" (all items delivered) at the latest arrival time across all items.

=== THREE DECISIONS ===
1. ORDER-TO-CONVEYOR ASSIGNMENT: which orders go on which conveyor, in what order.
   Constraint: for multi-order totes, all orders served must be simultaneously
   active (at the front of their conveyor queues) when that tote is loaded.

2. ITEM ROW SEQUENCE WITHIN TOTE: furthest conveyor first so items arrive
   closer together in time (reduces spread of delivery times).

3. TOTE LOADING SEQUENCE: greedy marginal gain — pick tote that delivers
   the most useful items to currently active orders. Ties broken by secondary
   score (unblocking value), then smallest tote ID.

=== DATA ===
  Can load from CSV: ranDataGen copy/{small|medium|large} sized samples/
  Files per dataset N: order_itemtypes_N.csv, order_quantities_N.csv, orders_totes_N.csv
  Rows = orders (order_id = 1-based row index), columns = item slots (aligned across files).
"""

import csv
import os

# ─────────────────────────────────────────────────────────────────────────────
# TIMING CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
ITEM_BUFFER        = 4.5   # seconds between consecutive item placements (and before first item)
BELT_TO_CORNER     = 9.5   # seconds for an item to cross a belt (midpoint to next corner)
CORNER_TO_MID      = 3.5   # seconds for an item to be picked by belt (corner to midpoint of next belt)

# Travel time from LOAD position to each conveyor's midpoint
TRAVEL = {
    0:  3.5,          # LOAD → Conv 0
    1: 16.5,          # LOAD → Conv 1  (3.5 + 9.5 + 3.5)
    2: 29.5,          # LOAD → Conv 2  (16.5 + 9.5 + 3.5)
    3: 42.5,          # LOAD → Conv 3  (29.5 + 9.5 + 3.5)
}

NUM_CONVEYORS  = 4
# NUM_ITEM_TYPES set from data when loading CSVs; default for legacy build_data()
NUM_ITEM_TYPES = 8
ITEM_NAMES     = ['circle','pentagon','trapezoid','triangle','star','moon','heart','cross']

# Root folder for generated datasets (relative to project root)
DATA_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'ranDataGen copy')
SAMPLE_FOLDERS = [
    'small sized samples',
    'medium sized samples',
    'large sized samples',
]


# ─────────────────────────────────────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────────────────────────────────────

def _parse_cell(s):
    """Return int from cell string (e.g. '6.0' or '6' or '' -> None)."""
    s = (s or '').strip()
    if not s:
        return None
    try:
        return int(float(s))
    except ValueError:
        return None


def load_data_from_csv(folder_path, dataset_id):
    """
    Load orders and totes from the three CSVs for one dataset.
    folder_path: path to folder containing order_itemtypes_{id}.csv, etc.
    dataset_id: integer (e.g. 1..30).
    Returns (orders, totes) in same format as build_data().
    Also sets global NUM_ITEM_TYPES from data (max item_type + 1).
    Raises on missing/invalid files; caller may catch to skip dataset.
    """
    global NUM_ITEM_TYPES
    base = os.path.join(folder_path, '')
    types_path = os.path.join(base, f'order_itemtypes_{dataset_id}.csv')
    qty_path   = os.path.join(base, f'order_quantities_{dataset_id}.csv')
    totes_path = os.path.join(base, f'orders_totes_{dataset_id}.csv')

    def read_csv_rows(path):
        rows = []
        with open(path, 'r', newline='', encoding='utf-8') as f:
            for row in csv.reader(f):
                rows.append(row)
        return rows

    types_rows = read_csv_rows(types_path)
    qty_rows   = read_csv_rows(qty_path)
    totes_rows = read_csv_rows(totes_path)

    # Align by row count (orders)
    n_rows = min(len(types_rows), len(qty_rows), len(totes_rows))
    orders = {}
    totes  = {}
    max_item_type = -1

    for row_idx in range(n_rows):
        order_id = row_idx + 1
        t_row = types_rows[row_idx] if row_idx < len(types_rows) else []
        q_row = qty_rows[row_idx]   if row_idx < len(qty_rows) else []
        to_row = totes_rows[row_idx] if row_idx < len(totes_rows) else []

        n_cols = min(len(t_row), len(q_row), len(to_row))
        for col_idx in range(n_cols):
            item_type = _parse_cell(t_row[col_idx])
            qty       = _parse_cell(q_row[col_idx])
            tote_id   = _parse_cell(to_row[col_idx])
            if item_type is None or qty is None or tote_id is None:
                continue
            if qty <= 0:
                continue
            max_item_type = max(max_item_type, item_type)

            if order_id not in orders:
                orders[order_id] = {'items': [], 'total_items': 0}
            orders[order_id]['items'].append((item_type, qty))
            orders[order_id]['total_items'] += qty

            if tote_id not in totes:
                totes[tote_id] = []
            totes[tote_id].append({
                'order':     order_id,
                'item_type': item_type,
                'quantity':  qty,
            })

    NUM_ITEM_TYPES = max(max_item_type + 1, 1)
    return orders, totes


def discover_dataset_ids(folder_path):
    """
    Find all dataset IDs in folder by scanning for orders_totes_*.csv.
    Returns sorted list of integers (e.g. [1, 2, ..., 30]).
    """
    ids = []
    prefix = 'orders_totes_'
    suffix = '.csv'
    try:
        for name in os.listdir(folder_path):
            if name.startswith(prefix) and name.endswith(suffix):
                mid = name[len(prefix):-len(suffix)]
                if mid.isdigit():
                    ids.append(int(mid))
    except OSError:
        pass
    return sorted(ids)


def build_data():
    """
    Hard-coded from the provided CSVs.
    Aligned by (order, slot): item_type, quantity, tote all correspond positionally.
    Item types are 0-based as provided.
    """
    raw = [
        # (order_id, item_type, quantity, tote_id)
        # Order 1
        (1, 3, 3, 1),
        (1, 1, 2, 1),
        # Order 2
        (2, 2, 3, 2),
        (2, 3, 1, 3),
        (2, 0, 1, 2),
        # Order 3
        (3, 3, 3, 3),
        (3, 2, 3, 2),
        (3, 0, 1, 1),
        # Order 4
        (4, 1, 1, 0),
        (4, 2, 1, 0),
    ]

    orders = {}
    totes  = {}

    for (oid, item_type, qty, tote_id) in raw:
        if oid not in orders:
            orders[oid] = {'items': [], 'total_items': 0}
        orders[oid]['items'].append((item_type, qty))
        orders[oid]['total_items'] += qty

        if tote_id not in totes:
            totes[tote_id] = []
        totes[tote_id].append({
            'order':     oid,
            'item_type': item_type,
            'quantity':  qty,
        })

    return orders, totes


# ─────────────────────────────────────────────────────────────────────────────
# DECISION 1: ORDER-TO-CONVEYOR ASSIGNMENT
# ─────────────────────────────────────────────────────────────────────────────

def build_constraint_satisfying_queues(orders, totes):
    """
    Assign orders to conveyor queues satisfying the simultaneous-active constraint.
    Orders that share a tote must be on different conveyors (so they can be
    active at the same time when that tote is loaded). Uses greedy graph coloring
    with NUM_CONVEYORS colors; orders in the same queue are ordered by order ID.
    """
    # Build conflict graph: orders that share a tote must be on different conveyors
    order_ids = sorted(orders.keys())
    neighbors = {oid: set() for oid in order_ids}
    for tote_id, entries in totes.items():
        oids_in_tote = {e['order'] for e in entries}
        for oid in oids_in_tote:
            neighbors[oid].update(oids_in_tote)
        for oid in oids_in_tote:
            neighbors[oid].discard(oid)

    # Greedy coloring: assign each order to first conveyor (color) not used by a neighbor
    order_to_conv = {}
    for oid in order_ids:
        used = {order_to_conv[n] for n in neighbors[oid] if n in order_to_conv}
        for c in range(NUM_CONVEYORS):
            if c not in used:
                order_to_conv[oid] = c
                break
        else:
            # More than NUM_CONVEYORS orders in a clique — put on conv 0 as fallback
            order_to_conv[oid] = 0

    # Build queues: same conveyor -> same list, sorted by order ID
    conv_queues = [[] for _ in range(NUM_CONVEYORS)]
    for oid in order_ids:
        conv_queues[order_to_conv[oid]].append(oid)
    for c in range(NUM_CONVEYORS):
        conv_queues[c].sort()

    return conv_queues, order_to_conv


# ─────────────────────────────────────────────────────────────────────────────
# DECISION 2: ROW ORDER WITHIN TOTE (furthest conveyor first)
# ─────────────────────────────────────────────────────────────────────────────

def order_rows_within_tote(conv_rows):
    """
    Sort rows so the furthest conveyor is placed on the belt first.
    This means items heading further travel more of their journey while
    closer-destination items are still being loaded, tightening arrival spread.
    """
    rows = list(conv_rows.items())   # [(conv_idx, [counts]), ...]
    rows.sort(key=lambda x: x[0], reverse=True)
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# DECISION 3: GREEDY TOTE SEQUENCING
# ─────────────────────────────────────────────────────────────────────────────

def score_tote(tote_id, totes, remaining, active_orders, queues, queue_pos):
    """
    Score a tote:
      Primary   : total still-needed items delivered to currently active orders.
      Secondary : number of orders completed, with +1 bonus if completing an
                  order unblocks a successor on the same conveyor.
    """
    primary = 0
    temp_remaining = {oid: dict(d) for oid, d in remaining.items()}

    for entry in totes[tote_id]:
        oid       = entry['order']
        item_type = entry['item_type']
        qty       = entry['quantity']
        if oid in active_orders:
            useful = min(qty, max(temp_remaining.get(oid, {}).get(item_type, 0), 0))
            primary += useful
            if item_type in temp_remaining.get(oid, {}):
                temp_remaining[oid][item_type] = max(
                    0, temp_remaining[oid][item_type] - qty)

    secondary = 0
    for ci in range(NUM_CONVEYORS):
        if queue_pos[ci] < len(queues[ci]):
            oid = queues[ci][queue_pos[ci]]
            if oid in active_orders:
                if all(v <= 0 for v in temp_remaining.get(oid, {}).values()):
                    has_successor = (queue_pos[ci] + 1) < len(queues[ci])
                    secondary += 2 if has_successor else 1

    return (primary, secondary)


def greedy_tote_sequence(orders, totes, conv_queues, verbose=False):
    """
    Build tote loading sequence using Greedy Marginal Gain.
    At each step: score all remaining totes, pick best, update state.
    """
    remaining = {}
    for oid, odata in orders.items():
        remaining[oid] = {}
        for (item_type, qty) in odata['items']:
            remaining[oid][item_type] = qty

    queues    = [list(q) for q in conv_queues]
    queue_pos = [0] * NUM_CONVEYORS

    def active_orders():
        result = set()
        for ci in range(NUM_CONVEYORS):
            if queue_pos[ci] < len(queues[ci]):
                result.add(queues[ci][queue_pos[ci]])
        return result

    remaining_totes = list(totes.keys())
    sequence        = []

    if verbose:
        print("\n=== Greedy Tote Sequencing ===")

    while remaining_totes:
        active = active_orders()
        best_tote      = None
        best_primary   = -1
        best_secondary = -1

        for tote_id in remaining_totes:
            p, s = score_tote(tote_id, totes, remaining, active, queues, queue_pos)
            better = (
                p > best_primary or
                (p == best_primary and s > best_secondary) or
                (p == best_primary and s == best_secondary and
                 (best_tote is None or tote_id < best_tote))
            )
            if better:
                best_primary   = p
                best_secondary = s
                best_tote      = tote_id

        sequence.append(best_tote)
        remaining_totes.remove(best_tote)

        if verbose:
            print(f"  Tote {best_tote}  score=({best_primary},{best_secondary})"
                  f"  active={sorted(active)}")

        for entry in totes[best_tote]:
            oid = entry['order']
            it  = entry['item_type']
            qty = entry['quantity']
            if oid in remaining and it in remaining[oid]:
                remaining[oid][it] = max(0, remaining[oid][it] - qty)

        for ci in range(NUM_CONVEYORS):
            if queue_pos[ci] < len(queues[ci]):
                oid = queues[ci][queue_pos[ci]]
                if all(v <= 0 for v in remaining.get(oid, {}).values()):
                    queue_pos[ci] += 1

    return sequence


# ─────────────────────────────────────────────────────────────────────────────
# SIMULATION
# ─────────────────────────────────────────────────────────────────────────────

def simulate(tote_sequence, orders, totes, conv_queues,
             verbose=False):
    """
    Simulate the circular belt.

    Timing:
      - Items placed one at a time, ITEM_BUFFER seconds apart.
      - First item of each tote is placed ITEM_BUFFER seconds after the
        previous item (no extra inter-tote gap).
      - Item placed at time T going to Conv c arrives at T + TRAVEL[c].
      - Next tote's first item can be placed 3s after the last item placed
        from the current tote (belt does NOT need to clear first).

    Returns results dict including event_log for CSV generation.
    """
    remaining = {}
    for oid, odata in orders.items():
        remaining[oid] = {}
        for (item_type, qty) in odata['items']:
            remaining[oid][item_type] = qty

    queues    = [list(q) for q in conv_queues]
    queue_pos = [0] * NUM_CONVEYORS

    def active_order(ci):
        return queues[ci][queue_pos[ci]] if queue_pos[ci] < len(queues[ci]) else None

    def conv_of(oid):
        for ci, q in enumerate(queues):
            if oid in q:
                return ci
        return None

    order_completion_times = {}
    # Tracks when the last item was placed on the belt
    # First item ever placed at t=3 (ITEM_BUFFER after t=0)
    last_item_placed_time = 0.0
    event_log = []

    # For tracking arrivals per order across multiple totes
    # remaining_remaining[oid] tracks items still undelivered
    items_delivered = {oid: {} for oid in orders}

    if verbose:
        print(f"\n{'='*68}")
        print(f"  SIMULATION")
        print(f"{'='*68}")
        for ci, q in enumerate(queues):
            print(f"  Conv {ci}: {q}")
        print(f"\n  Timing: TRAVEL = {TRAVEL}")
        print(f"  Item buffer = {ITEM_BUFFER}s between placements\n")

    for step, tote_id in enumerate(tote_sequence):
        if verbose:
            active = {ci: active_order(ci) for ci in range(NUM_CONVEYORS)}
            print(f"[Step {step+1}] Tote {tote_id}  active={active}")

        # Build conv_rows: items per active conveyor from this tote
        conv_rows = {}
        skipped   = []
        for entry in totes[tote_id]:
            oid       = entry['order']
            item_type = entry['item_type']
            qty       = entry['quantity']
            ci        = conv_of(oid)
            if ci is None:
                continue
            if active_order(ci) != oid:
                skipped.append(oid)
                continue
            if ci not in conv_rows:
                conv_rows[ci] = [0] * NUM_ITEM_TYPES
            conv_rows[ci][item_type] += qty

        if skipped and verbose:
            print(f"  ⚠ Orders {list(set(skipped))} not active — items skipped")

        if not conv_rows:
            if verbose:
                print(f"  (no active orders served)\n")
            continue

        # Decision 2: furthest conveyor first
        ordered_rows = order_rows_within_tote(conv_rows)

        # Log rows for output CSV
        for ci, counts in ordered_rows:
            event_log.append({'conv': ci, 'items': list(counts)})

        # Place items one at a time, ITEM_BUFFER apart
        # Track arrival times per conveyor from this tote
        conv_arrival_times = {}   # ci -> list of arrival times

        for (ci, counts) in ordered_rows:
            total_items = sum(counts)
            for k in range(total_items):
                place_t   = last_item_placed_time + ITEM_BUFFER
                arrival_t = place_t + TRAVEL[ci]
                last_item_placed_time = place_t
                if ci not in conv_arrival_times:
                    conv_arrival_times[ci] = []
                conv_arrival_times[ci].append(arrival_t)

            if verbose:
                names = get_item_names()
                named = {names[i]: counts[i]
                         for i in range(NUM_ITEM_TYPES) if counts[i] > 0}
                last_arr = max(conv_arrival_times[ci]) if ci in conv_arrival_times else 0
                print(f"  Row → Conv {ci} (Order {active_order(ci)}): "
                      f"{named}  last_arrival=t{last_arr:.1f}s")

        # Apply deliveries and check order completions
        for ci, arrival_list in conv_arrival_times.items():
            oid = active_order(ci)
            if oid is None:
                continue

            # Deliver items from this tote to this order
            for item_type, qty in enumerate(conv_rows[ci]):
                if qty > 0:
                    remaining[oid][item_type] = max(
                        0, remaining[oid].get(item_type, 0) - qty)

            still_needed = sum(remaining[oid].values())
            last_arr     = max(arrival_list)

            if verbose:
                print(f"  Conv {ci} (Order {oid}): last item arrives t={last_arr:.1f}s"
                      f" | still needed: {still_needed} items")

            if all(v <= 0 for v in remaining[oid].values()):
                # Order complete — completion time = last arrival for this order
                # across ALL totes (need to track the latest delivery)
                prev = order_completion_times.get(oid, 0)
                order_completion_times[oid] = max(prev, last_arr)
                queue_pos[ci] += 1
                next_oid = active_order(ci)
                if verbose:
                    print(f"  ✅ Order {oid} COMPLETE at t={order_completion_times[oid]:.1f}s")
                    if next_oid:
                        print(f"  → Conv {ci} now serving Order {next_oid}")
            else:
                # Track partial completion time (last item so far)
                prev = order_completion_times.get(oid, 0)
                # Don't record as complete yet, but track latest delivery
                # We'll use a separate dict for "last delivery so far"
                pass

        if verbose:
            print(f"  Last item placed at t={last_item_placed_time:.1f}s\n")

    # Any orders not yet completed
    for oid in orders:
        if oid not in order_completion_times:
            order_completion_times[oid] = float('inf')

    makespan       = max(order_completion_times.values())
    finite_times   = [t for t in order_completion_times.values()
                      if t != float('inf')]
    sum_completion = sum(finite_times)
    avg_completion = sum_completion / len(finite_times) if finite_times else float('inf')

    return {
        'order_completion_times': order_completion_times,
        'makespan':               makespan,
        'sum_completion_times':   sum_completion,
        'avg_completion_time':    avg_completion,
        'event_log':              event_log,
    }


# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT CSV
# ─────────────────────────────────────────────────────────────────────────────

def get_item_names():
    """Column names for item types (pad beyond len(ITEM_NAMES) with item0, item1, ...)."""
    return [ITEM_NAMES[i] if i < len(ITEM_NAMES) else f'item{i}' for i in range(NUM_ITEM_TYPES)]


def write_input_csv(event_log, output_path, quiet=False):
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['conv_num'] + get_item_names())
        for event in event_log:
            writer.writerow([event['conv']] + event['items'])
    if not quiet:
        print(f"  ✔ CSV written: {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# REPORTING
# ─────────────────────────────────────────────────────────────────────────────

def print_results(label, sequence, results, conv_queues):
    ms = results['makespan']
    print(f"\n{'='*68}")
    print(f"  {label}")
    print(f"{'='*68}")
    for ci, q in enumerate(conv_queues):
        print(f"  Conv {ci}: {q}")
    print(f"  Tote sequence       : {sequence}")
    if ms == float('inf'):
        print(f"  Makespan            : ∞  ← some orders never completed!")
    else:
        print(f"  Makespan            : {ms:.1f}s")
    print(f"  Sum completions     : {results['sum_completion_times']:.1f}s")
    print(f"  Avg completion      : {results['avg_completion_time']:.1f}s")
    print(f"\n  Per-order completion times:")
    for oid in sorted(results['order_completion_times']):
        t = results['order_completion_times'][oid]
        if t == float('inf'):
            print(f"    Order {oid}: ∞  ← never completed")
        else:
            print(f"    Order {oid}: {t:>7.1f}s")


def compare_table(results_dict):
    print(f"\n{'='*68}")
    print("  COMPARISON SUMMARY")
    print(f"{'='*68}")
    print(f"  {'Method':<42} {'Makespan':>9} {'SumC':>9} {'AvgC':>9}")
    print(f"  {'-'*42} {'-'*9} {'-'*9} {'-'*9}")
    for label, res in results_dict.items():
        ms = f"{res['makespan']:.1f}s" if res['makespan'] != float('inf') else "∞"
        sc = f"{res['sum_completion_times']:.1f}s"
        ac = (f"{res['avg_completion_time']:.1f}s"
              if res['avg_completion_time'] != float('inf') else "∞")
        print(f"  {label:<42} {ms:>9} {sc:>9} {ac:>9}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def run_single_dataset(folder_path, dataset_id, size_label, output_base, verbose=False):
    """
    Load one dataset, run greedy and baseline, optionally write CSVs.
    Returns (size_label, dataset_id, res_greedy, res_baseline).
    """
    orders, totes = load_data_from_csv(folder_path, dataset_id)
    if not orders or not totes:
        return (size_label, dataset_id, None, None)

    cq, _ = build_constraint_satisfying_queues(orders, totes)
    seq_g = greedy_tote_sequence(orders, totes, cq, verbose=verbose)
    seq_b = sorted(totes.keys())
    res_g = simulate(seq_g, orders, totes, cq, verbose=verbose)
    res_b = simulate(seq_b, orders, totes, cq, verbose=verbose)

    if output_base:
        out_dir = os.path.join(output_base, size_label, f'dataset_{dataset_id}')
        os.makedirs(out_dir, exist_ok=True)
        write_input_csv(res_g['event_log'], os.path.join(out_dir, 'input_greedy.csv'), quiet=True)
        write_input_csv(res_b['event_log'], os.path.join(out_dir, 'input_baseline.csv'), quiet=True)

    return (size_label, dataset_id, res_g, res_b)


if __name__ == '__main__':
    print("=" * 68)
    print("  MSE 433 — Greedy Tote Sequencing Simulation")
    print("=" * 68)

    # Output directory: greedy/outputs/ under project root
    output_base = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'greedy', 'outputs')

    # Optional: run built-in hard-coded data only (no CSV folders)
    import sys
    run_builtin_only = '--builtin' in sys.argv

    if run_builtin_only:
        orders, totes = build_data()
        print(f"\nLoaded {len(orders)} orders, {len(totes)} totes (built-in data)")
        cq, _ = build_constraint_satisfying_queues(orders, totes)
        seq_g = greedy_tote_sequence(orders, totes, cq, verbose=True)
        res_g = simulate(seq_g, orders, totes, cq, verbose=True)
        seq_b = sorted(totes.keys())
        res_b = simulate(seq_b, orders, totes, cq, verbose=False)
        print_results("Greedy | constraint queues | furthest first", seq_g, res_g, cq)
        print_results("Baseline | sorted totes | furthest first", seq_b, res_b, cq)
        compare_table({"Greedy": res_g, "Baseline": res_b})
        print(f"\n  ITEM_BUFFER = {ITEM_BUFFER}s, TRAVEL = {TRAVEL}")
        sys.exit(0)

    # Run all datasets from ranDataGen copy
    if not os.path.isdir(DATA_ROOT):
        print(f"\n  Data root not found: {DATA_ROOT}")
        print("  Run with --builtin to use the built-in example data only.")
        sys.exit(1)

    # Size label from folder name: "small sized samples" -> "small"
    def size_label_from_folder(folder_name):
        if folder_name.startswith('small'):
            return 'small'
        if folder_name.startswith('medium'):
            return 'medium'
        if folder_name.startswith('large'):
            return 'large'
        return folder_name.replace(' ', '_')[:10]

    all_rows = []
    for folder_name in SAMPLE_FOLDERS:
        folder_path = os.path.join(DATA_ROOT, folder_name)
        if not os.path.isdir(folder_path):
            print(f"  Skip (not a directory): {folder_path}")
            continue
        size_label = size_label_from_folder(folder_name)
        dataset_ids = discover_dataset_ids(folder_path)
        print(f"\n  {folder_name}: {len(dataset_ids)} datasets (ids {dataset_ids[:5]}{'...' if len(dataset_ids) > 5 else ''})")

        for did in dataset_ids:
            try:
                _, _, res_g, res_b = run_single_dataset(
                    folder_path, did, size_label, output_base, verbose=False)
            except Exception as e:
                print(f"    Dataset {did} failed: {e}")
                continue
            if res_g is not None and res_b is not None:
                all_rows.append({
                    'size': size_label,
                    'id': did,
                    'makespan_greedy': res_g['makespan'],
                    'makespan_baseline': res_b['makespan'],
                    'sum_greedy': res_g['sum_completion_times'],
                    'sum_baseline': res_b['sum_completion_times'],
                    'avg_greedy': res_g['avg_completion_time'],
                    'avg_baseline': res_b['avg_completion_time'],
                    'n_orders': len(res_g['order_completion_times']),
                })

    # Write results CSV for analysis (finite values only for numeric columns)
    results_path = os.path.join(output_base, 'simulation_results.csv')
    with open(results_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['size', 'dataset_id', 'method', 'makespan', 'sum_completion_times', 'avg_completion_time', 'n_orders'])
        for r in all_rows:
            ms_g = r['makespan_greedy'] if r['makespan_greedy'] != float('inf') else ''
            ms_b = r['makespan_baseline'] if r['makespan_baseline'] != float('inf') else ''
            ac_g = r['avg_greedy'] if r['avg_greedy'] != float('inf') else ''
            ac_b = r['avg_baseline'] if r['avg_baseline'] != float('inf') else ''
            n = r['n_orders']
            w.writerow([r['size'], r['id'], 'greedy', ms_g, r['sum_greedy'], ac_g, n])
            w.writerow([r['size'], r['id'], 'baseline', ms_b, r['sum_baseline'], ac_b, n])
    print(f"\n  Results saved to: {results_path}")

    # Summary table
    print(f"\n{'='*68}")
    print("  SUMMARY (all datasets)")
    print(f"{'='*68}")
    print(f"  {'Size':<8} {'ID':>4} {'Makespan (G)':>14} {'Makespan (B)':>14} {'SumC (G)':>12} {'SumC (B)':>12}")
    print(f"  {'-'*8} {'-'*4} {'-'*14} {'-'*14} {'-'*12} {'-'*12}")
    for r in all_rows:
        ms_g = f"{r['makespan_greedy']:.1f}" if r['makespan_greedy'] != float('inf') else "inf"
        ms_b = f"{r['makespan_baseline']:.1f}" if r['makespan_baseline'] != float('inf') else "inf"
        print(f"  {r['size']:<8} {r['id']:>4} {ms_g:>14} {ms_b:>14} {r['sum_greedy']:>12.1f} {r['sum_baseline']:>12.1f}")

    if output_base:
        print(f"\n  Outputs written to: {output_base}")
    print(f"\n  ITEM_BUFFER = {ITEM_BUFFER}s, TRAVEL = {TRAVEL}")

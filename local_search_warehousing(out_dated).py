"""
MSE 433 - Module 3: Warehousing
Local Search Algorithm for Conveyor Belt Order Consolidation & Sortation

=== PROBLEM OVERVIEW ===

Four conveyor belts (parallel machines) form a circular loop:
    LOAD -> conv0 -> conv1 -> conv2 -> conv3 -> back to LOAD

Items are loaded from totes onto the belt one at a time. Each conveyor has
a pneumatic arm that pushes items off when a scanner detects they belong to
the current order. One order bin sits at each conveyor at a time; when that
order is filled, the next order in that conveyor's queue becomes active.

=== THREE DECISION VARIABLES ===

1. Conveyor order assignment — which orders go on which belt, in what queue order.
2. Tote sequence            — the global order in which totes are loaded onto the belt.
3. Item sequence per tote   — within a tote, the order of conveyor-row placements.

=== LOCAL SEARCH NEIGHBORHOODS ===

1. Tote swap  — swap two totes' positions in the tote sequence.
2. Order move — move one order from its belt to a different belt at any position.

Strategy: first-improvement (accept the first neighbor that improves makespan).

=== TIME COMPLEXITY ===

Let T = #totes, I = max items per tote, N = #orders, C = #conveyors, P = max queue length.

  compute_makespan : O(T * I)                per call
  Tote swap scan   : O(T^2)   neighbors  x  O(T*I)  = O(T^3 * I)     per iteration
  Order move scan  : O(N*C*P) neighbors  x  O(T*I)  = O(N*C*P*T*I)   per iteration

With first-improvement, the expected cost per iteration is much lower than the worst
case because the search stops as soon as any improving neighbor is found.

Total worst case: O(max_iter * (T^3*I + N*C*P*T*I))
For this dataset (T~15, N=11, C=4, P~4, I~5): each iteration examines at most
~105 tote swaps + ~528 order moves, each costing ~75 operations => very fast.
"""

import csv
import os
import copy
import time as time_module


# ─────────────────────────────────────────────────────────────────────────────
# TUNABLE PARAMETERS  (calibrate after physical conveyor testing)
# ─────────────────────────────────────────────────────────────────────────────
TIME_PER_SEGMENT      = 7.5   # seconds between adjacent belt positions
LOOP_TIME             = 4 * TIME_PER_SEGMENT  # full belt circulation (80 s)
TOTE_LOAD_TIME        = 5.0    # seconds to physically place a tote on the belt
TIME_PER_ITEM_SPACING = 3.0    # seconds between consecutive items placed on belt
NUM_CONVEYORS         = 4
NUM_ITEM_TYPES        = 8

ITEM_NAMES = ['circle', 'pentagon', 'trapezoid', 'triangle',
              'star', 'moon', 'heart', 'cross']


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def read_csv_raw(path):
    """Read a CSV file into a list of rows, each row a list of float | None."""
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            row = [float(p.strip()) if p.strip() != '' else None for p in parts]
            rows.append(row)
    return rows


def build_data(base_path='ranDataGen'):
    """
    Parse the three data CSVs and construct:
      orders : {order_id: {'items': [(item_type, qty), ...], 'total_items': int}}
      totes  : {tote_id:  [{'order': oid, 'item_type': it, 'quantity': qty}, ...]}

    Each row in the CSV files corresponds to one order (1-indexed).
    Columns within a row correspond to each other across the three files:
      order_itemtypes  -> item type code
      order_quantities -> how many of that item
      orders_totes     -> which tote contains that item
    """
    it_data = read_csv_raw(os.path.join(base_path, 'order_itemtypes.csv'))
    qt_data = read_csv_raw(os.path.join(base_path, 'order_quantities.csv'))
    tt_data = read_csv_raw(os.path.join(base_path, 'orders_totes.csv'))

    orders = {}
    totes  = {}

    for order_idx, (it_row, qt_row, tt_row) in enumerate(zip(it_data, qt_data, tt_data)):
        order_id = order_idx + 1
        orders[order_id] = {'items': [], 'total_items': 0}
        for it, qt, tt in zip(it_row, qt_row, tt_row):
            if it is None or qt is None or tt is None:
                continue
            tote_id   = int(tt)
            item_type = int(it)
            quantity  = int(qt)
            orders[order_id]['items'].append((item_type, quantity))
            orders[order_id]['total_items'] += quantity
            totes.setdefault(tote_id, []).append({
                'order':     order_id,
                'item_type': item_type,
                'quantity':  quantity,
            })
    return orders, totes


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: order-to-conveyor lookup
# ─────────────────────────────────────────────────────────────────────────────

def _build_order_conv_map(conv_queues):
    """Return {order_id: conveyor_index} from conv_queues."""
    m = {}
    for ci, q in enumerate(conv_queues):
        for oid in q:
            m[oid] = ci
    return m


# ─────────────────────────────────────────────────────────────────────────────
# CORE: MAKESPAN COMPUTATION (full simulation)
# ─────────────────────────────────────────────────────────────────────────────

def compute_makespan(orders, totes, conv_queues, tote_sequence,
                     row_strategy='furthest_first'):
    """
    Simulate the circular belt system and return (makespan, results_dict).

    Timing model
    ------------
    Belt positions: LOAD -> conv0 -> conv1 -> conv2 -> conv3 -> LOAD

    Item placed on belt at time T, destined for conveyor c:
        arrival_time = T + TIME_PER_SEGMENT * (c + 1)

    Totes are loaded one at a time. Between totes the belt must clear
    (all items from the previous tote have arrived at their destinations).

    Within a tote, items for different conveyors are grouped into "rows"
    (one row per destination conveyor). Rows are ordered by `row_strategy`:
        'furthest_first' — highest conveyor number placed first (recommended)
        'nearest_first'  — lowest conveyor number placed first

    Within a row, individual items are spaced by TIME_PER_ITEM_SPACING.

    Constraint
    ----------
    Only items destined for the *currently active* order on each conveyor
    are placed. If an order served by a tote is not currently active (it's
    still queued behind an earlier unfinished order), those items are SKIPPED.
    This means a bad conveyor assignment can make orders uncompletable.

    Returns
    -------
    makespan : float
        Time when the last conveyor finishes (inf if some order never completes).
    result   : dict
        'order_completion_times', 'makespan', 'event_log'
    """
    # Remaining items per order
    remaining = {}
    for oid, odata in orders.items():
        remaining[oid] = {}
        for item_type, qty in odata['items']:
            remaining[oid][item_type] = qty

    queues    = [list(q) for q in conv_queues]
    queue_pos = [0] * NUM_CONVEYORS
    order_conv = _build_order_conv_map(conv_queues)

    def active_order(ci):
        return queues[ci][queue_pos[ci]] if queue_pos[ci] < len(queues[ci]) else None

    order_completion_times = {}
    current_time = 0.0
    event_log    = []

    for tote_id in tote_sequence:
        tote_load_start = current_time + TOTE_LOAD_TIME

        # Gather which items from this tote go to which conveyor (only active orders)
        conv_rows = {}
        for entry in totes[tote_id]:
            oid       = entry['order']
            item_type = entry['item_type']
            qty       = entry['quantity']
            ci = order_conv.get(oid)
            if ci is None or active_order(ci) != oid:
                continue
            if ci not in conv_rows:
                conv_rows[ci] = [0] * NUM_ITEM_TYPES
            conv_rows[ci][item_type] += qty

        if not conv_rows:
            current_time = tote_load_start
            continue

        # Decision 3: order rows within this tote
        rows = list(conv_rows.items())
        if row_strategy == 'furthest_first':
            rows.sort(key=lambda x: x[0], reverse=True)
        else:
            rows.sort(key=lambda x: x[0])

        for ci, counts in rows:
            event_log.append({'conv': ci, 'items': list(counts)})

        # Compute placement / arrival times
        belt_cursor  = tote_load_start
        clearance_t  = tote_load_start
        conv_last_arr = {}

        for ci, counts in rows:
            total = sum(counts)
            for k in range(total):
                place_t   = belt_cursor + k * TIME_PER_ITEM_SPACING
                arrival_t = place_t + TIME_PER_SEGMENT * (ci + 1)
                conv_last_arr[ci] = max(conv_last_arr.get(ci, 0), arrival_t)
                clearance_t = max(clearance_t, arrival_t)
            belt_cursor += total * TIME_PER_ITEM_SPACING

        # Deliver items and check order completions
        for ci, last_arr in conv_last_arr.items():
            oid = active_order(ci)
            if oid is None or oid in order_completion_times:
                continue

            for item_type_idx in range(NUM_ITEM_TYPES):
                qty = conv_rows[ci][item_type_idx]
                if qty > 0 and item_type_idx in remaining.get(oid, {}):
                    remaining[oid][item_type_idx] -= qty

            if all(v <= 0 for v in remaining[oid].values()):
                order_completion_times[oid] = last_arr
                queue_pos[ci] += 1

        current_time = clearance_t

    # Any order not completed gets infinite time
    for oid in orders:
        if oid not in order_completion_times:
            order_completion_times[oid] = float('inf')

    makespan = max(order_completion_times.values())

    return makespan, {
        'order_completion_times': order_completion_times,
        'makespan':               makespan,
        'event_log':              event_log,
    }


# ─────────────────────────────────────────────────────────────────────────────
# GREEDY TOTE SEQUENCING (for generating a good initial solution)
# ─────────────────────────────────────────────────────────────────────────────

def greedy_tote_sequence(orders, totes, conv_queues):
    """
    Build a tote sequence using greedy marginal gain.

    At each step pick the unprocessed tote that delivers the most useful items
    to currently active orders. Ties broken by number of orders completed, then
    smallest tote ID.
    """
    remaining = {}
    for oid, odata in orders.items():
        remaining[oid] = {}
        for item_type, qty in odata['items']:
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
    sequence = []

    while remaining_totes:
        active = active_orders()
        best_tote = None
        best_score = (-1, -1, float('inf'))  # (primary, secondary, tote_id for min)

        for tid in remaining_totes:
            primary = 0
            temp_rem = {oid: dict(d) for oid, d in remaining.items()}

            for entry in totes[tid]:
                oid = entry['order']
                it  = entry['item_type']
                qty = entry['quantity']
                if oid in active:
                    useful = min(qty, max(temp_rem.get(oid, {}).get(it, 0), 0))
                    primary += useful
                    if it in temp_rem.get(oid, {}):
                        temp_rem[oid][it] -= qty

            secondary = 0
            for ci in range(NUM_CONVEYORS):
                if queue_pos[ci] < len(queues[ci]):
                    oid = queues[ci][queue_pos[ci]]
                    if oid in active:
                        if all(v <= 0 for v in temp_rem.get(oid, {}).values()):
                            has_next = (queue_pos[ci] + 1) < len(queues[ci])
                            secondary += 2 if has_next else 1

            score = (primary, secondary, -tid)  # negate tid so larger = better (smaller id)
            if score > best_score:
                best_score = score
                best_tote = tid

        sequence.append(best_tote)
        remaining_totes.remove(best_tote)

        for entry in totes[best_tote]:
            oid = entry['order']
            it  = entry['item_type']
            qty = entry['quantity']
            if oid in remaining and it in remaining[oid]:
                remaining[oid][it] -= qty

        for ci in range(NUM_CONVEYORS):
            if queue_pos[ci] < len(queues[ci]):
                oid = queues[ci][queue_pos[ci]]
                if all(v <= 0 for v in remaining.get(oid, {}).values()):
                    queue_pos[ci] += 1

    return sequence


# ─────────────────────────────────────────────────────────────────────────────
# LOCAL SEARCH
# ─────────────────────────────────────────────────────────────────────────────

def local_search(orders, totes, initial_queues, initial_tote_seq,
                 max_iterations=1000, row_strategy='furthest_first',
                 min_orders_per_conv=1, verbose=True):
    """
    First-improvement local search over two neighborhoods.

    Neighborhood 1 — Tote swap
        Swap the positions of two totes in the tote sequence.
        Size: C(T,2) = T*(T-1)/2 neighbors.

    Neighborhood 2 — Order move
        Move one order from its current conveyor to a different conveyor
        at every feasible insertion position.  Moves that would leave fewer
        than `min_orders_per_conv` orders on the source conveyor are skipped,
        ensuring all 4 belts stay active.
        Size: sum over orders of (C-1) * (len(target_queue) + 1).

    The search alternates: first it scans all tote swaps; if no improvement
    is found, it scans all order moves. If neither neighborhood yields an
    improvement, the algorithm terminates (local optimum).

    Parameters
    ----------
    orders, totes        : problem data (from build_data)
    initial_queues       : list of 4 lists of order IDs
    initial_tote_seq     : list of tote IDs
    max_iterations       : hard cap on iterations
    row_strategy         : within-tote item ordering strategy
    min_orders_per_conv  : minimum orders each conveyor must keep (default 1)
    verbose              : print progress

    Returns
    -------
    best_queues, best_tote_seq, best_makespan, best_result, history
    """
    best_queues   = [list(q) for q in initial_queues]
    best_tote_seq = list(initial_tote_seq)
    best_makespan, best_result = compute_makespan(
        orders, totes, best_queues, best_tote_seq, row_strategy
    )

    if verbose:
        ms_str = f"{best_makespan:.1f}s" if best_makespan != float('inf') else "INF"
        print(f"  Initial makespan: {ms_str}")
        if best_makespan == float('inf'):
            print("  WARNING: initial solution has infinite makespan "
                  "(some orders never complete)")

    history = [(0, best_makespan)]

    for iteration in range(1, max_iterations + 1):
        improved = False

        # ── Neighborhood 1: Tote swaps ──────────────────────────────────
        n = len(best_tote_seq)
        for i in range(n - 1):
            if improved:
                break
            for j in range(i + 1, n):
                new_seq = list(best_tote_seq)
                new_seq[i], new_seq[j] = new_seq[j], new_seq[i]
                ms, res = compute_makespan(
                    orders, totes, best_queues, new_seq, row_strategy
                )
                if ms < best_makespan:
                    old_ms = best_makespan
                    best_tote_seq = new_seq
                    best_makespan = ms
                    best_result   = res
                    improved = True
                    if verbose:
                        print(f"  Iter {iteration:>3}: Tote swap "
                              f"pos {i}<->{j} "
                              f"(tote {best_tote_seq[j]}<->tote {best_tote_seq[i]}) "
                              f"=> {ms:.1f}s  (was {old_ms:.1f}s)")
                    break

        # ── Neighborhood 2: Order moves (only if tote swap didn't help) ─
        if not improved:
            for from_ci in range(NUM_CONVEYORS):
                if improved:
                    break
                if len(best_queues[from_ci]) <= min_orders_per_conv:
                    continue  # don't strip a belt below the minimum
                for oid in list(best_queues[from_ci]):
                    if improved:
                        break
                    for to_ci in range(NUM_CONVEYORS):
                        if to_ci == from_ci:
                            continue
                        if improved:
                            break
                        max_pos = len(best_queues[to_ci]) + 1
                        for pos in range(max_pos):
                            new_q = [list(q) for q in best_queues]
                            new_q[from_ci].remove(oid)
                            new_q[to_ci].insert(pos, oid)
                            ms, res = compute_makespan(
                                orders, totes, new_q, best_tote_seq, row_strategy
                            )
                            if ms < best_makespan:
                                old_ms = best_makespan
                                best_queues   = new_q
                                best_makespan = ms
                                best_result   = res
                                improved = True
                                if verbose:
                                    print(f"  Iter {iteration:>3}: Move order {oid} "
                                          f"conv {from_ci} -> conv {to_ci} pos {pos} "
                                          f"=> {ms:.1f}s  (was {old_ms:.1f}s)")
                                break

        history.append((iteration, best_makespan))

        if not improved:
            if verbose:
                print(f"  Iter {iteration:>3}: No improving neighbor found. "
                      "Local optimum reached.")
            break

    if verbose:
        ms_str = f"{best_makespan:.1f}s" if best_makespan != float('inf') else "INF"
        print(f"\n  Local search finished after {len(history)-1} iteration(s).")
        print(f"  Best makespan: {ms_str}")

    return best_queues, best_tote_seq, best_makespan, best_result, history


# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT: CSV WRITERS
# ─────────────────────────────────────────────────────────────────────────────

def write_schedule_csv(conv_queues, orders, path):
    """
    Write the conveyor-input CSV — one row per order.
    Format: conv_num, circle, pentagon, trapezoid, triangle, star, moon, heart, cross

    Rows are grouped by conveyor in queue order (all Conv 0 orders first, then
    Conv 1, etc.).  The conveyor processes them top-to-bottom for its belt number.
    """
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['conv_num'] + ITEM_NAMES)
        for ci, queue in enumerate(conv_queues):
            for oid in queue:
                counts = [0] * NUM_ITEM_TYPES
                for item_type, qty in orders[oid]['items']:
                    counts[item_type] += qty
                writer.writerow([ci] + counts)
    print(f"    -> {path}")


def write_tote_sequence_csv(tote_seq, path):
    """Write tote processing order: step, tote_id."""
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['step', 'tote_id'])
        for i, tid in enumerate(tote_seq):
            writer.writerow([i + 1, tid])
    print(f"    -> {path}")


def write_item_sequence_csv(tote_seq, totes, conv_queues, orders, path,
                            row_strategy='furthest_first'):
    """
    Write the full item-level loading sequence.
    Each row: step, tote_id, conveyor, order_id, item_type, item_name
    """
    remaining = {}
    for oid, odata in orders.items():
        remaining[oid] = {}
        for it, qty in odata['items']:
            remaining[oid][it] = qty

    queues    = [list(q) for q in conv_queues]
    queue_pos = [0] * NUM_CONVEYORS
    order_conv = _build_order_conv_map(conv_queues)

    def active_order(ci):
        return queues[ci][queue_pos[ci]] if queue_pos[ci] < len(queues[ci]) else None

    rows_out = []
    step = 0

    for tote_id in tote_seq:
        conv_rows    = {}
        conv_entries = {}
        for entry in totes[tote_id]:
            oid = entry['order']
            it  = entry['item_type']
            qty = entry['quantity']
            ci  = order_conv.get(oid)
            if ci is None or active_order(ci) != oid:
                continue
            if ci not in conv_rows:
                conv_rows[ci]    = [0] * NUM_ITEM_TYPES
                conv_entries[ci] = []
            conv_rows[ci][it] += qty
            conv_entries[ci].append(entry)

        if not conv_rows:
            continue

        ordered = list(conv_rows.items())
        if row_strategy == 'furthest_first':
            ordered.sort(key=lambda x: x[0], reverse=True)
        else:
            ordered.sort(key=lambda x: x[0])

        for ci, _ in ordered:
            for entry in conv_entries[ci]:
                for _ in range(entry['quantity']):
                    step += 1
                    it = entry['item_type']
                    rows_out.append([
                        step, tote_id, ci, entry['order'], it,
                        ITEM_NAMES[it] if it < len(ITEM_NAMES) else f"type_{it}"
                    ])

        # Advance queue state (mirrors compute_makespan logic)
        for ci in conv_rows:
            oid = active_order(ci)
            if oid is None:
                continue
            for it_idx in range(NUM_ITEM_TYPES):
                qty = conv_rows[ci][it_idx]
                if qty > 0 and it_idx in remaining.get(oid, {}):
                    remaining[oid][it_idx] -= qty
            if all(v <= 0 for v in remaining[oid].values()):
                queue_pos[ci] += 1

    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['step', 'tote_id', 'conveyor', 'order_id',
                         'item_type', 'item_name'])
        writer.writerows(rows_out)
    print(f"    -> {path}")


# ─────────────────────────────────────────────────────────────────────────────
# INITIAL SOLUTION CONSTRUCTORS
# ─────────────────────────────────────────────────────────────────────────────

def build_constraint_satisfying_queues():
    """
    Hand-crafted conveyor assignment that satisfies the simultaneous-active
    constraint derived from multi-order tote analysis.

    Multi-order totes in the reference dataset:
        Tote  1: Orders {1, 11}   -> must be on different conveyors, both active
        Tote  8: Orders {4, 6, 9} -> need 3 different conveyors, all active
        Tote  9: Orders {2, 4}    -> different conveyors, both active
        Tote 10: Orders {2, 8}    -> different conveyors, both active

    Assignment:
        Conv 0: [O1,  O2]
        Conv 1: [O11, O4]
        Conv 2: [O3,  O6,  O8]
        Conv 3: [O7,  O9,  O5, O10]
    """
    return [
        [1, 2],
        [11, 4],
        [3, 6, 8],
        [7, 9, 5, 10],
    ]


def build_round_robin_queues(orders):
    """Simple round-robin (does NOT guarantee the simultaneous-active constraint)."""
    order_list = sorted(orders.keys())
    queues = [[] for _ in range(NUM_CONVEYORS)]
    for i, oid in enumerate(order_list):
        queues[i % NUM_CONVEYORS].append(oid)
    return queues


# ─────────────────────────────────────────────────────────────────────────────
# DISPLAY HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def print_solution(label, queues, tote_seq, makespan, completion_times):
    """Pretty-print a solution summary."""
    print(f"\n{'='*68}")
    print(f"  {label}")
    print(f"{'='*68}")
    for ci, q in enumerate(queues):
        print(f"  Conv {ci}: {q}")
    print(f"  Tote sequence: {tote_seq}")
    ms_str = f"{makespan:.1f}s" if makespan != float('inf') else "INF"
    print(f"  Makespan: {ms_str}")
    print(f"\n  Per-order completion times:")
    for oid in sorted(completion_times):
        t = completion_times[oid]
        t_str = f"{t:>8.1f}s" if t != float('inf') else "     INF"
        print(f"    Order {oid:>2}: {t_str}")


# ─────────────────────────────────────────────────────────────────────────────
# MOCK DATA EXAMPLE
# ─────────────────────────────────────────────────────────────────────────────

def run_mock_example():
    """
    Small self-contained example: 4 orders, 3 totes, 4 conveyors.
    Demonstrates that the local search can find improving moves.

    Orders:
      O1: 2x circle  + 1x pentagon  (3 items)
      O2: 3x trapezoid              (3 items)
      O3: 1x circle  + 2x triangle  (3 items)
      O4: 2x pentagon               (2 items)

    Totes (shared):
      T1: O1 2x circle,  O3 1x circle       <- multi-order
      T2: O1 1x pentagon, O2 3x trapezoid   <- multi-order
      T3: O3 2x triangle, O4 2x pentagon    <- multi-order

    Because every tote is shared, all four orders must be simultaneously
    active when any tote loads => one order per conveyor works.
    """
    print("\n" + "=" * 68)
    print("  MOCK DATA EXAMPLE (4 orders, 3 totes)")
    print("=" * 68)

    mock_orders = {
        1: {'items': [(0, 2), (1, 1)], 'total_items': 3},
        2: {'items': [(2, 3)],          'total_items': 3},
        3: {'items': [(0, 1), (3, 2)], 'total_items': 3},
        4: {'items': [(1, 2)],          'total_items': 2},
    }

    mock_totes = {
        1: [{'order': 1, 'item_type': 0, 'quantity': 2},
            {'order': 3, 'item_type': 0, 'quantity': 1}],
        2: [{'order': 1, 'item_type': 1, 'quantity': 1},
            {'order': 2, 'item_type': 2, 'quantity': 3}],
        3: [{'order': 3, 'item_type': 3, 'quantity': 2},
            {'order': 4, 'item_type': 1, 'quantity': 2}],
    }

    init_queues   = [[1], [2], [3], [4]]
    init_tote_seq = [1, 2, 3]

    ms0, res0 = compute_makespan(mock_orders, mock_totes,
                                 init_queues, init_tote_seq)
    print(f"\n  Initial queues:   {init_queues}")
    print(f"  Initial tote seq: {init_tote_seq}")
    print(f"  Initial makespan: {ms0:.1f}s\n")

    best_q, best_ts, best_ms, best_res, hist = local_search(
        mock_orders, mock_totes, init_queues, init_tote_seq,
        max_iterations=50, verbose=True
    )

    print(f"\n  Final queues:   {best_q}")
    print(f"  Final tote seq: {best_ts}")
    print(f"  Final makespan: {best_ms:.1f}s")
    if ms0 != float('inf') and best_ms != float('inf'):
        imp = ms0 - best_ms
        pct = (imp / ms0) * 100 if ms0 > 0 else 0
        print(f"  Improvement:    {imp:.1f}s ({pct:.1f}%)")

    return best_ms


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 68)
    print("  MSE 433 - Local Search for Warehousing Optimization")
    print("=" * 68)

    # ── 1. Run mock example ──────────────────────────────────────────────
    run_mock_example()

    # ── 2. Load real dataset ─────────────────────────────────────────────
    print("\n\n" + "=" * 68)
    print("  REAL DATASET")
    print("=" * 68)

    orders, totes = build_data('ranDataGen')
    print(f"\n  Loaded {len(orders)} orders, {len(totes)} totes")

    print("\n  Order summary:")
    for oid in sorted(orders):
        items_str = ', '.join(
            f"{ITEM_NAMES[it] if it < len(ITEM_NAMES) else it} x{qty}"
            for it, qty in orders[oid]['items'])
        print(f"    Order {oid:>2} ({orders[oid]['total_items']:>2} items): {items_str}")

    print("\n  Tote contents:")
    for tid in sorted(totes):
        contents = ', '.join(
            f"O{e['order']} {ITEM_NAMES[e['item_type']]}x{e['quantity']}"
            for e in totes[tid])
        print(f"    Tote {tid:>2}: {contents}")

    print("\n  Multi-order totes (require simultaneous activation):")
    for tid in sorted(totes):
        oids = sorted({e['order'] for e in totes[tid]})
        if len(oids) > 1:
            print(f"    Tote {tid:>2}: Orders {oids}")

    # ── 3. Baseline (sorted totes, constraint-satisfying queues) ─────────
    init_queues   = build_constraint_satisfying_queues()
    init_tote_seq = sorted(totes.keys())

    init_ms, init_res = compute_makespan(
        orders, totes, init_queues, init_tote_seq)

    print_solution("BASELINE: sorted totes + constraint queues",
                   init_queues, init_tote_seq, init_ms,
                   init_res['order_completion_times'])

    # ── 4. Greedy initial solution ───────────────────────────────────────
    greedy_seq = greedy_tote_sequence(orders, totes, init_queues)
    greedy_ms, greedy_res = compute_makespan(
        orders, totes, init_queues, greedy_seq)

    print_solution("GREEDY: marginal-gain tote ordering + constraint queues",
                   init_queues, greedy_seq, greedy_ms,
                   greedy_res['order_completion_times'])

    # ── 5. Local search from greedy solution ─────────────────────────────
    print("\n" + "=" * 68)
    print("  RUNNING LOCAL SEARCH  (starting from greedy solution)")
    print("=" * 68 + "\n")

    t0 = time_module.time()
    best_q, best_ts, best_ms, best_res, history = local_search(
        orders, totes, init_queues, greedy_seq,
        max_iterations=1000, verbose=True
    )
    elapsed = time_module.time() - t0

    print_solution(f"LOCAL SEARCH RESULT  ({elapsed:.2f}s wall time)",
                   best_q, best_ts, best_ms,
                   best_res['order_completion_times'])

    # ── 6. Improvement summary ───────────────────────────────────────────
    print(f"\n{'='*68}")
    print("  COMPARISON")
    print(f"{'='*68}")
    print(f"  {'Method':<45} {'Makespan':>10}")
    print(f"  {'-'*45} {'-'*10}")
    for label, ms in [("Baseline (sorted totes)",           init_ms),
                      ("Greedy (marginal-gain totes)",      greedy_ms),
                      ("Local Search (from greedy)",        best_ms)]:
        ms_str = f"{ms:.1f}s" if ms != float('inf') else "INF"
        print(f"  {label:<45} {ms_str:>10}")

    if greedy_ms != float('inf') and best_ms != float('inf'):
        imp = greedy_ms - best_ms
        pct = (imp / greedy_ms) * 100 if greedy_ms > 0 else 0
        print(f"\n  Improvement over greedy: {imp:.1f}s ({pct:.1f}%)")
    if init_ms != float('inf') and best_ms != float('inf'):
        imp = init_ms - best_ms
        pct = (imp / init_ms) * 100 if init_ms > 0 else 0
        print(f"  Improvement over baseline: {imp:.1f}s ({pct:.1f}%)")

    # ── 7. Write output files ────────────────────────────────────────────
    print(f"\n{'='*68}")
    print("  OUTPUT FILES")
    print(f"{'='*68}")
    write_schedule_csv(
        best_q, orders,
        'local_search_best_schedule.csv')
    write_tote_sequence_csv(
        best_ts,
        'local_search_best_tote_sequence.csv')
    write_item_sequence_csv(
        best_ts, totes, best_q, orders,
        'local_search_best_tote_item_sequence.csv')

    print(f"\n{'='*68}")
    print("  DONE")
    print(f"{'='*68}")

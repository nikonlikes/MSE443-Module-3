"""
MSE 433 - Module 3: Warehousing
Greedy Marginal Gain Heuristic — Tote Sequencing Simulation

=== BELT TIMING MODEL ===
  Layout (clockwise from LOAD):
    LOAD (top-left corner)
      → 3s → Conv 0 (midpoint of top belt)
      → 7.5s → corner → 3s → Conv 1 (midpoint of right belt)
      → 7.5s → corner → 3s → Conv 2 (midpoint of bottom belt)
      → 7.5s → corner → 3s → Conv 3 (midpoint of left belt)

  Travel time from LOAD to each conveyor:
    Conv 0:  3.0s
    Conv 1: 13.5s
    Conv 2: 24.0s
    Conv 3: 34.5s

  Item placement timing:
    - Items placed one at a time with 3s buffer between consecutive items
    - First item of each tote also has 3s buffer from last item of previous tote
      (i.e., there is NO extra inter-tote gap beyond the standard 3s item spacing)
    - Item placed at belt_time T going to Conv c arrives at T + TRAVEL[c]

  Tote clearance: next item can be placed 3s after the previous one.
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

=== DATA (4 orders, totes 0–3) ===
  Order 1: item_type=3 qty=3 tote=1 | item_type=1 qty=2 tote=1
  Order 2: item_type=2 qty=3 tote=2 | item_type=3 qty=1 tote=3 | item_type=0 qty=1 tote=2
  Order 3: item_type=3 qty=3 tote=3 | item_type=2 qty=3 tote=2 | item_type=0 qty=1 tote=1
  Order 4: item_type=1 qty=1 tote=0 | item_type=2 qty=1 tote=0
"""

import csv

# ─────────────────────────────────────────────────────────────────────────────
# TIMING CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
ITEM_BUFFER        = 3.0   # seconds between consecutive item placements (and before first item)
BELT_TO_CORNER     = 7.5   # seconds from midpoint of one belt side to the next corner
CORNER_TO_MID      = 3.0   # seconds from corner to midpoint of next belt (= travel to conv)

# Travel time from LOAD position to each conveyor's midpoint
TRAVEL = {
    0:  3.0,          # LOAD → Conv 0
    1: 13.5,          # LOAD → Conv 1  (3 + 7.5 + 3)
    2: 24.0,          # LOAD → Conv 2  (13.5 + 7.5 + 3)
    3: 34.5,          # LOAD → Conv 3  (24 + 7.5 + 3)
}

NUM_CONVEYORS  = 4
NUM_ITEM_TYPES = 8
ITEM_NAMES     = ['circle','pentagon','trapezoid','triangle','star','moon','heart','cross']


# ─────────────────────────────────────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────────────────────────────────────

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

    Multi-order totes in this dataset:
      Tote 1: Orders 1, 3  → must be on DIFFERENT conveyors, active at same time
      Tote 2: Orders 2, 3  → must be on DIFFERENT conveyors, active at same time
      Tote 3: Orders 2, 3  → must be on DIFFERENT conveyors, active at same time

    Since Orders 2 and 3 share TWO totes (2 and 3), they must be simultaneously
    active for BOTH of those totes. This means they must both be at the front of
    their queues across a stretch of tote loads — they effectively run in parallel
    until both are complete.

    Order 1 shares Tote 1 with Order 3, so Order 1 must be active when Order 3 is.
    Order 4 has only its own totes (Tote 0), so it can go on any remaining conveyor.

    Valid assignment:
      Conv 0: [Order 1]        — shares Tote 1 with Order 3 (Conv 1), active together
      Conv 1: [Order 3]        — shares Tote 1 with Order 1, Totes 2&3 with Order 2
      Conv 2: [Order 2]        — shares Totes 2&3 with Order 3 (Conv 1)
      Conv 3: [Order 4]        — independent (Tote 0 only)
    """
    conv_queues = [
        [1],   # Conv 0
        [3],   # Conv 1
        [2],   # Conv 2
        [4],   # Conv 3
    ]

    order_to_conv = {}
    for ci, q in enumerate(conv_queues):
        for oid in q:
            order_to_conv[oid] = ci

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
                named = {ITEM_NAMES[i]: counts[i]
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

def write_input_csv(event_log, output_path):
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['conv_num'] + ITEM_NAMES)
        for event in event_log:
            writer.writerow([event['conv']] + event['items'])
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

if __name__ == '__main__':
    print("=" * 68)
    print("  MSE 433 — Greedy Tote Sequencing Simulation")
    print("=" * 68)

    orders, totes = build_data()

    print(f"\nLoaded {len(orders)} orders, {len(totes)} totes")

    print("\nOrder summary:")
    for oid in sorted(orders):
        items_str = ', '.join(
            f"item{it}×{qty}" for (it, qty) in orders[oid]['items'])
        print(f"  Order {oid} ({orders[oid]['total_items']} items): {items_str}")

    print("\nTote contents:")
    for tid in sorted(totes):
        contents = ', '.join(
            f"O{e['order']} item{e['item_type']}×{e['quantity']}"
            for e in totes[tid])
        print(f"  Tote {tid}: {contents}")

    print("\nMulti-order totes (all listed orders must be simultaneously active):")
    for tid in sorted(totes):
        orders_in_tote = sorted({e['order'] for e in totes[tid]})
        if len(orders_in_tote) > 1:
            print(f"  Tote {tid}: Orders {orders_in_tote}")

    output_dir = '/mnt/user-data/outputs/'
    all_results = {}

    # ── Config 1: Constraint-satisfying queues + greedy ──────────────────
    label = "Greedy | constraint queues | furthest first"
    cq, _ = build_constraint_satisfying_queues(orders, totes)
    seq   = greedy_tote_sequence(orders, totes, cq, verbose=True)
    res   = simulate(seq, orders, totes, cq, verbose=True)
    all_results[label] = res
    print_results(label, seq, res, cq)
    write_input_csv(res['event_log'],
                    output_dir + 'input_greedy_constraint_furthest.csv')

    # ── Config 2: Baseline (sorted tote IDs) ─────────────────────────────
    label2  = "Baseline | sorted totes | furthest first"
    seq_b   = sorted(totes.keys())
    res_b   = simulate(seq_b, orders, totes, cq, verbose=False)
    all_results[label2] = res_b
    print_results(label2, seq_b, res_b, cq)
    write_input_csv(res_b['event_log'],
                    output_dir + 'input_baseline_sorted_furthest.csv')

    # ── Comparison ────────────────────────────────────────────────────────
    compare_table(all_results)

    print(f"""
{'='*68}
  TIMING PARAMETERS USED
{'='*68}
  ITEM_BUFFER        = {ITEM_BUFFER}s   (between consecutive item placements)
  TRAVEL[Conv 0]     = {TRAVEL[0]}s  (LOAD → Conv 0 midpoint)
  TRAVEL[Conv 1]     = {TRAVEL[1]}s (LOAD → Conv 1 midpoint)
  TRAVEL[Conv 2]     = {TRAVEL[2]}s (LOAD → Conv 2 midpoint)
  TRAVEL[Conv 3]     = {TRAVEL[3]}s (LOAD → Conv 3 midpoint)
""")

"""
MSE 433 - Module 3: Warehousing
Greedy Marginal Gain Heuristic — Tote Sequencing Simulation

=== THREE DECISIONS ===

1. SEQUENCE OF ORDERS per conveyor (which orders go on which conveyor, in what order)
   - 11 orders, 4 conveyors. Each conveyor works through a queue of orders.
   - Order numbering is ARBITRARY — we assign orders to conveyors however we want.
   - KEY CONSTRAINT: for every multi-order tote, all orders it serves must be
     simultaneously active (at the front of their respective conveyor queues)
     when that tote is loaded. Otherwise items pass the arm and are lost.
   - This simulation uses a hand-crafted constraint-satisfying assignment
     discovered by analyzing which orders share totes.

2. SEQUENCE OF ITEMS WITHIN A TOTE (row order in output CSV)
   - One tote can serve multiple conveyors. Each conveyor = one row in the CSV.
   - Rows are loaded onto the belt in CSV order, spaced by TIME_PER_ITEM_SPACING.
   - Strategy: load furthest conveyor first so all items arrive closer together.

3. SEQUENCE OF TOTE LOADING (greedy marginal gain)
   - One tote at a time. Belt must clear before next tote loads.
   - Greedy rule: pick the tote delivering the most items to currently active orders.
   - Ties broken by smallest tote ID.

=== TIMING MODEL ===
  Belt: LOAD -> conv0 -> conv1 -> conv2 -> conv3 -> back to LOAD
  TIME_PER_SEGMENT  : travel time between adjacent belt positions
  LOOP_TIME         : full loop = 4 * TIME_PER_SEGMENT
  TOTE_LOAD_TIME    : time to physically place a tote on the belt
  TIME_PER_ITEM_SPACING : gap between consecutive items placed from same tote row

  Item placed at belt time T, going to conveyor c:
    arrives at t = T + TIME_PER_SEGMENT * (c + 1)

  Tote clearance time = latest item arrival across all rows.
  Order completion time = arrival of the last item that satisfies that order.

=== OUTPUT ===
  Produces the input CSV for the IDEAS Clinic conveyor machine.
  Format: conv_num, circle, pentagon, trapezoid, triangle, star, moon, heart, cross
  Rows ordered by loading sequence (Decision 2 within each tote).
"""

import copy
import csv

# ─────────────────────────────────────────────────────────────────────────────
# TUNABLE PARAMETERS  (calibrate after physical conveyor testing)
# ─────────────────────────────────────────────────────────────────────────────
TIME_PER_SEGMENT      = 20.0   # [ASSUMPTION] seconds: LOAD->conv0, conv0->conv1, etc.
LOOP_TIME             = 4 * TIME_PER_SEGMENT
TOTE_LOAD_TIME        = 5.0    # [ASSUMPTION] seconds to physically load a tote
TIME_PER_ITEM_SPACING = 2.0    # [ASSUMPTION] seconds between consecutive items on belt
NUM_CONVEYORS         = 4
NUM_ITEM_TYPES        = 8

ITEM_NAMES = ['circle','pentagon','trapezoid','triangle','star','moon','heart','cross']


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def read_csv_raw(path):
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


def build_data():
    base = '/mnt/user-data/uploads/'
    it_data = read_csv_raw(base + 'order_itemtypes.csv')
    qt_data = read_csv_raw(base + 'order_quantities.csv')
    tt_data = read_csv_raw(base + 'orders_totes.csv')

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
            if tote_id not in totes:
                totes[tote_id] = []
            totes[tote_id].append({
                'order':     order_id,
                'item_type': item_type,
                'quantity':  quantity
            })
    return orders, totes


# ─────────────────────────────────────────────────────────────────────────────
# DECISION 1: ORDER-TO-CONVEYOR ASSIGNMENT
# ─────────────────────────────────────────────────────────────────────────────

def build_constraint_satisfying_queues(orders, totes):
    """
    Assign orders to conveyor queues satisfying the simultaneous-active constraint.

    For every multi-order tote, all orders it serves must be on DIFFERENT conveyors
    AND must all be simultaneously at the front of their queues when that tote loads.

    Multi-order totes in this dataset:
      Tote 1:  Orders 1, 11  -> need different conveyors, active at same time
      Tote 8:  Orders 4, 6, 9 -> need 3 different conveyors, active at same time
      Tote 9:  Orders 2, 4   -> need different conveyors, active at same time
      Tote 10: Orders 2, 8   -> need different conveyors, active at same time

    Valid assignment derived from constraint analysis:
      Conv 0: [O1,  O2]           (O1 and O11 active at start for Tote1;
      Conv 1: [O11, O4]            O2 and O4 active together at pos1 for Tote9)
      Conv 2: [O3,  O6,  O8]      (O4,O6,O9 all at pos1 across C1,C2,C3 for Tote8;
      Conv 3: [O7,  O9,  O5, O10]  O2 still active when O8 becomes active for Tote10)

    Returns:
      conv_queues   : list of 4 lists of order IDs
      order_to_conv : { order_id: conveyor_index }
    """
    conv_queues = [
        [1,  2],        # Conv 0
        [11, 4],        # Conv 1
        [3,  6,  8],    # Conv 2
        [7,  9,  5, 10] # Conv 3
    ]

    order_to_conv = {}
    for conv_idx, queue in enumerate(conv_queues):
        for oid in queue:
            order_to_conv[oid] = conv_idx

    return conv_queues, order_to_conv


def build_round_robin_queues(orders, totes):
    """
    Simple round-robin assignment (baseline comparison).
    Does NOT guarantee the simultaneous-active constraint.
    """
    order_list  = list(orders.keys())
    conv_queues = [[] for _ in range(NUM_CONVEYORS)]
    for i, oid in enumerate(order_list):
        conv_queues[i % NUM_CONVEYORS].append(oid)
    order_to_conv = {oid: conv_idx
                     for conv_idx, q in enumerate(conv_queues)
                     for oid in q}
    return conv_queues, order_to_conv


# ─────────────────────────────────────────────────────────────────────────────
# DECISION 2: ITEM SEQUENCE WITHIN TOTE (row order)
# ─────────────────────────────────────────────────────────────────────────────

def order_rows_within_tote(conv_rows, strategy='furthest_first'):
    """
    Order the rows (one per destination conveyor) within a tote.

    furthest_first: put highest conveyor number first on the belt.
                    Those items travel furthest so loading them first
                    means all items arrive at their conveyors closer together.
    nearest_first:  put lowest conveyor number first (baseline).
    """
    rows = list(conv_rows.items())   # [(conv_idx, [item counts]), ...]
    if strategy == 'furthest_first':
        rows.sort(key=lambda x: x[0], reverse=True)
    else:
        rows.sort(key=lambda x: x[0])
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# DECISION 3: GREEDY TOTE SEQUENCING
# ─────────────────────────────────────────────────────────────────────────────

def score_tote(tote_id, totes, remaining, active_orders, queues, queue_pos):
    """
    Score a tote using two criteria (returned as a tuple for comparison):

    Primary   : total still-needed items delivered to currently active orders.
    Secondary : number of orders this tote completes, with a bonus if the
                completed order has a successor waiting on the same conveyor.
                Completing a blocked order unblocks its successor, which is
                more valuable than an equal-score tote that doesn't.

    Ties on primary are broken by secondary, then by smallest tote ID.
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
                temp_remaining[oid][item_type] -= qty

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

    At each step:
      - Determine active orders (front of each conveyor queue)
      - Score remaining totes: primary = useful items, secondary = unblocking value
      - Pick best tote (ties on primary broken by secondary, then smallest tote ID)
      - Update remaining items; advance queues when orders complete
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
            print(f"  Tote {best_tote:>2}  score=({best_primary},{best_secondary})  "
                  f"active={sorted(active)}")

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
# SIMULATION
# ─────────────────────────────────────────────────────────────────────────────

def simulate(tote_sequence, orders, totes, conv_queues,
             row_strategy='furthest_first', verbose=False):
    """
    Simulate the circular belt system with all three decisions applied.

    Returns results dict including event_log for CSV generation.
    """
    # ── Setup ─────────────────────────────────────────────────────────────
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
    current_time = 0.0
    event_log    = []

    if verbose:
        print(f"\n{'='*68}")
        print(f"  SIMULATION  (row strategy: {row_strategy})")
        print(f"{'='*68}")
        for ci, q in enumerate(queues):
            print(f"  Conv {ci}: {q}")
        print()

    # ── Process totes ─────────────────────────────────────────────────────
    for step, tote_id in enumerate(tote_sequence):
        tote_load_start = current_time + TOTE_LOAD_TIME

        if verbose:
            active = {ci: active_order(ci) for ci in range(NUM_CONVEYORS)}
            print(f"[Step {step+1}] Tote {tote_id}  "
                  f"load_start=t{tote_load_start:.0f}s  active={active}")

        # Build conv_rows: only for orders that are currently active
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
            print(f"  ⚠ Orders {list(set(skipped))} not active — items skipped "
                  f"(check conveyor queue assignment)")

        if not conv_rows:
            current_time = tote_load_start
            if verbose:
                print(f"  (no active orders served — belt idle)\n")
            continue

        # Decision 2: order rows within tote
        ordered_rows = order_rows_within_tote(conv_rows, strategy=row_strategy)

        # Log rows for output CSV
        for ci, counts in ordered_rows:
            event_log.append({'conv': ci, 'items': counts})
            if verbose:
                named = {ITEM_NAMES[i]: counts[i]
                         for i in range(NUM_ITEM_TYPES) if counts[i] > 0}
                print(f"  Row → Conv {ci} (Order {active_order(ci)}): {named}")

        # Compute belt placement times and item arrivals
        belt_cursor   = tote_load_start
        clearance_t   = tote_load_start
        conv_last_arr = {}   # conv -> time of last item arrival from this tote

        for (ci, counts) in ordered_rows:
            total = sum(counts)
            for k in range(total):
                place_t   = belt_cursor + k * TIME_PER_ITEM_SPACING
                arrival_t = place_t + TIME_PER_SEGMENT * (ci + 1)
                conv_last_arr[ci] = max(conv_last_arr.get(ci, 0), arrival_t)
                clearance_t = max(clearance_t, arrival_t)
            belt_cursor += total * TIME_PER_ITEM_SPACING

        # Apply deliveries and check order completions
        for ci, last_arr in conv_last_arr.items():
            oid = active_order(ci)
            if oid is None or oid in order_completion_times:
                continue

            for item_type, qty in enumerate(conv_rows[ci]):
                if qty > 0 and item_type in remaining.get(oid, {}):
                    remaining[oid][item_type] -= qty

            still_needed = sum(max(v, 0) for v in remaining[oid].values())

            if verbose:
                print(f"  Conv {ci} (Order {oid}): last item t={last_arr:.1f}s | "
                      f"still needed: {still_needed}")

            if all(v <= 0 for v in remaining[oid].values()):
                order_completion_times[oid] = last_arr
                queue_pos[ci] += 1
                next_oid = active_order(ci)
                if verbose:
                    print(f"  ✅ Order {oid} COMPLETE at t={last_arr:.1f}s")
                    if next_oid:
                        print(f"  → Conv {ci} now serving Order {next_oid}")

        current_time = clearance_t
        if verbose:
            print(f"  Belt cleared at t={current_time:.1f}s\n")

    # Mark any unfinished orders
    for oid in orders:
        if oid not in order_completion_times:
            order_completion_times[oid] = float('inf')

    makespan       = max(order_completion_times.values())
    finite_times   = [t for t in order_completion_times.values()
                      if t != float('inf')]
    sum_completion = sum(finite_times)
    avg_completion = (sum_completion / len(finite_times)
                      if finite_times else float('inf'))

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
            print(f"    Order {oid:>2}: ∞  ← never completed")
        else:
            print(f"    Order {oid:>2}: {t:>7.1f}s")


def compare_table(results_dict):
    print(f"\n{'='*68}")
    print("  COMPARISON SUMMARY")
    print(f"{'='*68}")
    print(f"  {'Method':<40} {'Makespan':>9} {'SumC':>9} {'AvgC':>9}")
    print(f"  {'-'*40} {'-'*9} {'-'*9} {'-'*9}")
    for label, res in results_dict.items():
        ms = f"{res['makespan']:.1f}s" if res['makespan'] != float('inf') else "∞"
        sc = f"{res['sum_completion_times']:.1f}s"
        ac = (f"{res['avg_completion_time']:.1f}s"
              if res['avg_completion_time'] != float('inf') else "∞")
        print(f"  {label:<40} {ms:>9} {sc:>9} {ac:>9}")


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
        items_str = ', '.join(f"t{it}×{qty}" for (it, qty) in orders[oid]['items'])
        print(f"  Order {oid:>2} ({orders[oid]['total_items']:>2} items): {items_str}")

    print("\nTote contents:")
    for tid in sorted(totes):
        contents = ', '.join(
            f"O{e['order']} t{e['item_type']}×{e['quantity']}" for e in totes[tid])
        print(f"  Tote {tid:>2}: {contents}")

    # Identify multi-order totes (important constraint)
    print("\nMulti-order totes (orders must be simultaneously active when loaded):")
    for tid in sorted(totes):
        orders_in_tote = list({e['order'] for e in totes[tid]})
        if len(orders_in_tote) > 1:
            print(f"  Tote {tid:>2}: Orders {sorted(orders_in_tote)}")

    all_results   = {}
    all_sequences = {}
    all_queues    = {}
    output_dir    = '/mnt/user-data/outputs/'

    # ── Config 1: Constraint-satisfying queues + greedy + furthest first ──
    label = "Greedy | constraint queues | furthest first"
    cq, _   = build_constraint_satisfying_queues(orders, totes)
    seq     = greedy_tote_sequence(orders, totes, cq, verbose=True)
    res     = simulate(seq, orders, totes, cq,
                       row_strategy='furthest_first', verbose=False)
    all_results[label]   = res
    all_sequences[label] = seq
    all_queues[label]    = cq
    print_results(label, seq, res, cq)
    write_input_csv(res['event_log'], output_dir + 'input_greedy_constraint_furthest.csv')

    # ── Config 2: Constraint-satisfying queues + greedy + nearest first ───
    label = "Greedy | constraint queues | nearest first"
    res   = simulate(seq, orders, totes, cq,
                     row_strategy='nearest_first', verbose=False)
    all_results[label]   = res
    all_sequences[label] = seq
    all_queues[label]    = cq
    print_results(label, seq, res, cq)
    write_input_csv(res['event_log'], output_dir + 'input_greedy_constraint_nearest.csv')

    # ── Config 3: Constraint-satisfying queues + sorted totes (baseline) ──
    label   = "Baseline | constraint queues | furthest first"
    seq_b   = sorted(totes.keys())
    res_b   = simulate(seq_b, orders, totes, cq,
                       row_strategy='furthest_first', verbose=False)
    all_results[label]   = res_b
    all_sequences[label] = seq_b
    all_queues[label]    = cq
    print_results(label, seq_b, res_b, cq)
    write_input_csv(res_b['event_log'], output_dir + 'input_baseline_constraint_furthest.csv')

    # ── Config 4: Round-robin queues + greedy (shows why ordering matters) ─
    label   = "Greedy | round-robin queues | furthest first"
    rr, _   = build_round_robin_queues(orders, totes)
    seq_rr  = greedy_tote_sequence(orders, totes, rr, verbose=False)
    res_rr  = simulate(seq_rr, orders, totes, rr,
                       row_strategy='furthest_first', verbose=False)
    all_results[label]   = res_rr
    all_sequences[label] = seq_rr
    all_queues[label]    = rr
    print_results(label, seq_rr, res_rr, rr)
    write_input_csv(res_rr['event_log'], output_dir + 'input_greedy_roundrobin_furthest.csv')

    # ── Comparison ────────────────────────────────────────────────────────
    compare_table(all_results)

    # ── Verbose walkthrough of best finite method ─────────────────────────
    finite = {k: v for k, v in all_results.items()
              if v['makespan'] != float('inf')}
    if finite:
        best = min(finite, key=lambda k: finite[k]['makespan'])
        print(f"\n\n{'='*68}")
        print(f"  BEST: {best}")
        print(f"{'='*68}")
        simulate(all_sequences[best], orders, totes, all_queues[best],
                 row_strategy='furthest_first', verbose=True)
    else:
        print("\n⚠ All methods had infinite makespan.")

    import shutil
    shutil.copy('/home/claude/greedy_tote_simulation.py',
                output_dir + 'greedy_tote_simulation.py')

    print(f"""
{'='*68}
  TUNABLE PARAMETERS (update after physical conveyor testing)
{'='*68}
  TIME_PER_SEGMENT      = {TIME_PER_SEGMENT}s  LOAD->conv0, conv0->conv1, etc.
  LOOP_TIME             = {LOOP_TIME}s  full belt circulation
  TOTE_LOAD_TIME        = {TOTE_LOAD_TIME}s   time to place tote on belt
  TIME_PER_ITEM_SPACING = {TIME_PER_ITEM_SPACING}s   gap between items placed on belt
""")

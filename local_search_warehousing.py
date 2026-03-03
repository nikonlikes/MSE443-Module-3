"""
Local Search for Warehousing Conveyor Scheduling
------------------------------------------------

This module implements a simple local search heuristic for the conveyor
system described in the MSE 433 Module 3 materials.

Decisions:
  1. Order assignment to conveyors and sequence on each conveyor
  2. Sequence of totes
  3. Sequence of items within each tote row (here: fixed strategy 'furthest_first')

The search operates on:
  - belt_assignment: list of lists of order IDs, one list per conveyor
  - tote_sequence  : list of tote IDs (integers)

Given these, compute_makespan() simulates the system and returns the
resulting makespan (time when the last order finishes).

Two neighborhoods are used:
  - Tote swap: swap two totes in the tote_sequence
  - Order move: move a single order to a different conveyor (any position)

First-improvement local search:
  - Iterate through neighbors in a fixed order
  - As soon as a strictly better neighbor is found, move to it
  - Stop when no improving neighbor exists or max_iterations is reached

Only the Python standard library is used.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterable, Optional

import csv
import copy


# ──────────────────────────────────────────────────────────────────────────────
# Basic model parameters (can be tuned if needed)
# ──────────────────────────────────────────────────────────────────────────────

TIME_PER_SEGMENT = 20.0        # seconds between belt positions
LOOP_TIME = 4 * TIME_PER_SEGMENT
TOTE_LOAD_TIME = 5.0           # seconds to place a tote on the belt
TIME_PER_ITEM_SPACING = 2.0    # seconds between consecutive items on belt
NUM_CONVEYORS = 4
NUM_ITEM_TYPES = 8

ITEM_NAMES = [
    "circle",
    "pentagon",
    "trapezoid",
    "triangle",
    "star",
    "moon",
    "heart",
    "cross",
]


# ──────────────────────────────────────────────────────────────────────────────
# Data structures and loading helpers
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Order:
    """Represents a customer order."""

    items: List[Tuple[int, int]]  # list of (item_type, quantity)
    total_items: int


@dataclass
class ToteEntry:
    """Represents items for a single order inside a tote."""

    order: int
    item_type: int
    quantity: int


OrdersDict = Dict[int, Order]
TotesDict = Dict[int, List[ToteEntry]]
BeltAssignment = List[List[int]]  # conveyor index -> list of order IDs
ToteSequence = List[int]          # list of tote IDs


def _read_csv_raw(path: str) -> List[List[Optional[float]]]:
    """
    Read a CSV file as a grid of floats/None.

    Each non-empty cell is parsed as float, empty cells become None.
    """
    rows: List[List[Optional[float]]] = []
    with open(path, newline="") as f:
        reader = csv.reader(f)
        for raw_row in reader:
            # skip completely empty lines
            if not any(cell.strip() for cell in raw_row):
                continue
            row: List[Optional[float]] = []
            for cell in raw_row:
                cell = cell.strip()
                if cell == "":
                    row.append(None)
                else:
                    row.append(float(cell))
            rows.append(row)
    return rows


def load_orders_and_totes(
    order_itemtypes_path: str,
    order_quantities_path: str,
    orders_totes_path: str,
) -> Tuple[OrdersDict, TotesDict]:
    """
    Build orders and totes dictionaries from the three CSV files.

    This matches the structure used in greedy_tote_simulation.py.
    """
    it_data = _read_csv_raw(order_itemtypes_path)
    qt_data = _read_csv_raw(order_quantities_path)
    tt_data = _read_csv_raw(orders_totes_path)

    orders: OrdersDict = {}
    totes: TotesDict = {}

    for order_idx, (it_row, qt_row, tt_row) in enumerate(
        zip(it_data, qt_data, tt_data), start=1
    ):
        orders[order_idx] = Order(items=[], total_items=0)
        for it, qt, tt in zip(it_row, qt_row, tt_row):
            if it is None or qt is None or tt is None:
                continue
            tote_id = int(tt)
            item_type = int(it)
            quantity = int(qt)

            orders[order_idx].items.append((item_type, quantity))
            orders[order_idx].total_items += quantity

            if tote_id not in totes:
                totes[tote_id] = []
            totes[tote_id].append(
                ToteEntry(order=order_idx, item_type=item_type, quantity=quantity)
            )

    return orders, totes


# ──────────────────────────────────────────────────────────────────────────────
# Item row ordering within a tote (Decision 3)
# ──────────────────────────────────────────────────────────────────────────────

def _order_rows_within_tote(
    conv_rows: Dict[int, List[int]],
    strategy: str = "furthest_first",
) -> List[Tuple[int, List[int]]]:
    """
    Decide the order in which rows (conveyors) are loaded for a tote.

    strategy = 'furthest_first' puts higher conveyor indices first so that
    items travelling the longest distance are placed earlier, making
    arrivals more synchronized across conveyors.
    """
    rows = list(conv_rows.items())
    if strategy == "furthest_first":
        rows.sort(key=lambda x: x[0], reverse=True)
    else:
        rows.sort(key=lambda x: x[0])
    return rows


# ──────────────────────────────────────────────────────────────────────────────
# Core simulation and makespan computation
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class SimulationResult:
    order_completion_times: Dict[int, float]
    makespan: float
    sum_completion_times: float
    avg_completion_time: float
    event_log: List[Dict[str, object]]  # entries: {'conv', 'items', 'tote'}


def _simulate_schedule(
    tote_sequence: ToteSequence,
    orders: OrdersDict,
    totes: TotesDict,
    belt_assignment: BeltAssignment,
    row_strategy: str = "furthest_first",
) -> SimulationResult:
    """
    Simulate the circular belt system for a given schedule.

    All four conveyors are active in parallel: a single tote can supply items
    to multiple conveyors (and thus multiple orders) in one load. The belt
    forms a loop LOAD -> conv0 -> conv1 -> conv2 -> conv3 -> LOAD; items
    placed at time T arrive at conveyor ci at T + TIME_PER_SEGMENT * (ci + 1).

    Parameters
    ----------
    tote_sequence:
        Order in which totes are loaded on the belt.
    orders, totes:
        Problem data as returned by load_orders_and_totes().
    belt_assignment:
        List of length NUM_CONVEYORS, each element is the sequence of order
        IDs to be processed by that conveyor.

    Returns
    -------
    SimulationResult with per-order completion times, makespan and an
    event_log suitable for writing to an input CSV for the conveyor.
    """
    # remaining[order_id][item_type] = quantity still needed
    remaining: Dict[int, Dict[int, int]] = {}
    for oid, order in orders.items():
        remaining[oid] = {}
        for item_type, qty in order.items:
            remaining[oid][item_type] = qty

    # Cache order -> conveyor index for fast lookup
    order_to_conv: Dict[int, int] = {}
    for ci, q in enumerate(belt_assignment):
        for oid in q:
            order_to_conv[oid] = ci

    order_completion_times: Dict[int, float] = {}
    current_time = 0.0
    event_log: List[Dict[str, object]] = []

    for tote_id in tote_sequence:
        # Time when we can start putting items from this tote on the belt
        tote_load_start = current_time + TOTE_LOAD_TIME

        # Build rows of items per conveyor for this tote (items stay on loop).
        # One tote can feed all 4 conveyors; conv_rows can have keys 0,1,2,3.
        conv_rows: Dict[int, List[int]] = {}
        entries_by_conv: Dict[int, List[ToteEntry]] = {}
        for entry in totes.get(tote_id, []):
            oid = entry.order
            item_type = entry.item_type
            qty = entry.quantity

            ci = order_to_conv.get(oid)
            if ci is None:
                continue

            if ci not in conv_rows:
                conv_rows[ci] = [0] * NUM_ITEM_TYPES
            conv_rows[ci][item_type] += qty

            if ci not in entries_by_conv:
                entries_by_conv[ci] = []
            entries_by_conv[ci].append(entry)

        # If tote carries no items for known orders, time still advances
        if not conv_rows:
            current_time = tote_load_start
            continue

        # Decide order of rows within this tote (Decision 3)
        ordered_rows = _order_rows_within_tote(conv_rows, strategy=row_strategy)

        # Log rows for potential CSV generation
        for ci, counts in ordered_rows:
            event_log.append({"conv": ci, "items": counts, "tote": tote_id})

        # Place items from each row on the belt and compute arrival times per conveyor
        belt_cursor = tote_load_start
        clearance_t = tote_load_start
        arrival_times_by_conv: Dict[int, List[float]] = {}

        for ci, counts in ordered_rows:
            total_items = sum(counts)
            if total_items == 0:
                continue
            if ci not in arrival_times_by_conv:
                arrival_times_by_conv[ci] = []
            for k in range(total_items):
                place_t = belt_cursor + k * TIME_PER_ITEM_SPACING
                arrival_t = place_t + TIME_PER_SEGMENT * (ci + 1)
                arrival_times_by_conv[ci].append(arrival_t)
                if arrival_t > clearance_t:
                    clearance_t = arrival_t
            belt_cursor += total_items * TIME_PER_ITEM_SPACING

        # Deliver items to their respective orders in FIFO order on each conveyor.
        # Items that arrive while an order is not active simply "wait" on the loop
        # until the order uses them; for makespan, we only need their arrival time.
        for ci, times in arrival_times_by_conv.items():
            if ci not in entries_by_conv:
                continue
            time_iter = iter(times)
            for entry in entries_by_conv[ci]:
                oid = entry.order
                item_type = entry.item_type
                qty = entry.quantity
                if oid not in remaining or item_type not in remaining[oid]:
                    # Unknown or unnecessary item type; just advance time iterator
                    for _ in range(qty):
                        try:
                            next(time_iter)
                        except StopIteration:
                            break
                    continue
                for _ in range(qty):
                    try:
                        t = next(time_iter)
                    except StopIteration:
                        break
                    remaining[oid][item_type] -= 1
                    if oid not in order_completion_times and all(
                        v <= 0 for v in remaining[oid].values()
                    ):
                        order_completion_times[oid] = t

        current_time = clearance_t

    # Any order that never finished gets infinite completion time
    for oid in orders:
        if oid not in order_completion_times:
            order_completion_times[oid] = float("inf")

    makespan = max(order_completion_times.values())
    finite_times = [t for t in order_completion_times.values() if t != float("inf")]
    sum_completion = sum(finite_times)
    avg_completion = (
        sum_completion / len(finite_times) if finite_times else float("inf")
    )

    return SimulationResult(
        order_completion_times=order_completion_times,
        makespan=makespan,
        sum_completion_times=sum_completion,
        avg_completion_time=avg_completion,
        event_log=event_log,
    )


def compute_makespan(
    orders: OrdersDict,
    totes: TotesDict,
    belt_assignment: BeltAssignment,
    tote_sequence: ToteSequence,
) -> float:
    """
    Compute the makespan for a given schedule.

    Parameters
    ----------
    orders, totes:
        Problem data.
    belt_assignment:
        List of length NUM_CONVEYORS; each sub-list is the queue of orders
        for that conveyor (order IDs).
    tote_sequence:
        List of tote IDs in the order they are processed.

    Returns
    -------
    float
        Makespan (completion time of the last order). If an order never
        completes under this schedule, the makespan will be +inf.

    Time complexity
    ---------------
    Let T = number of totes and I = total number of item entries across
    all totes. The simulation runs in O(T + I) time because each tote and
    each item entry is processed a constant number of times.
    """
    result = _simulate_schedule(
        tote_sequence=tote_sequence,
        orders=orders,
        totes=totes,
        belt_assignment=belt_assignment,
    )
    return result.makespan


# ──────────────────────────────────────────────────────────────────────────────
# Neighborhood definitions
# ──────────────────────────────────────────────────────────────────────────────

def _tote_swap_neighbors(tote_sequence: ToteSequence) -> Iterable[ToteSequence]:
    """
    Generate neighbors by swapping any pair of totes in the sequence.

    Number of neighbors is O(T^2) where T is the number of totes.
    """
    n = len(tote_sequence)
    for i in range(n - 1):
        for j in range(i + 1, n):
            neighbor = tote_sequence.copy()
            neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
            yield neighbor


def _order_move_neighbors(
    belt_assignment: BeltAssignment,
) -> Iterable[BeltAssignment]:
    """
    Generate neighbors by moving a single order to another conveyor,
    while enforcing that all conveyors remain in use.

    For every conveyor c_from and order position k within it, consider
    moving that order to every possible position on every other conveyor.
    Neighbors that leave some conveyor empty are discarded.

    If there are O orders in total and C conveyors, the number of
    neighbors is O(O^2 * C) in the worst case (still small for
    the MSE 433 instance).
    """
    # For deterministic behavior we iterate conveyors and positions in order
    for c_from, queue in enumerate(belt_assignment):
        for pos, oid in enumerate(queue):
            for c_to in range(NUM_CONVEYORS):
                if c_to == c_from:
                    continue
                dest_queue = belt_assignment[c_to]
                for insert_pos in range(len(dest_queue) + 1):
                    if c_from == c_to and insert_pos == pos:
                        continue
                    new_assignment: BeltAssignment = [
                        list(q) for q in belt_assignment
                    ]
                    moved_oid = new_assignment[c_from].pop(pos)
                    new_assignment[c_to].insert(insert_pos, moved_oid)
                    # Enforce using all conveyors: no queue may be empty
                    if any(len(q) == 0 for q in new_assignment):
                        continue
                    yield new_assignment


# ──────────────────────────────────────────────────────────────────────────────
# Local search
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class LocalSearchResult:
    best_belt_assignment: BeltAssignment
    best_tote_sequence: ToteSequence
    best_makespan: float
    best_event_log: List[Dict[str, object]]


def local_search(
    orders: OrdersDict,
    totes: TotesDict,
    initial_belt_assignment: BeltAssignment,
    initial_tote_sequence: ToteSequence,
    max_iterations: int = 200,
) -> LocalSearchResult:
    """
    Run first-improvement local search.

    Search order per iteration:
      1. Try all tote-swap neighbors
      2. If none improve, try all order-move neighbors

    As soon as a better neighbor is found, move to it and start a new
    iteration. If no better neighbor exists for either neighborhood,
    or max_iterations iterations are reached, the search stops.

    Returns the best schedule found (assignment, tote sequence and event log).
    """
    current_belt = copy.deepcopy(initial_belt_assignment)
    current_totes = list(initial_tote_sequence)
    current_result = _simulate_schedule(
        tote_sequence=current_totes,
        orders=orders,
        totes=totes,
        belt_assignment=current_belt,
    )
    current_makespan = current_result.makespan

    for _ in range(max_iterations):
        improved = False

        # 1) Tote swap neighborhood
        for neighbor_totes in _tote_swap_neighbors(current_totes):
            ms = compute_makespan(orders, totes, current_belt, neighbor_totes)
            if ms < current_makespan:
                current_totes = neighbor_totes
                current_makespan = ms
                current_result = _simulate_schedule(
                    tote_sequence=current_totes,
                    orders=orders,
                    totes=totes,
                    belt_assignment=current_belt,
                )
                improved = True
                break  # first improvement

        if improved:
            continue

        # 2) Order move neighborhood
        for neighbor_belt in _order_move_neighbors(current_belt):
            ms = compute_makespan(orders, totes, neighbor_belt, current_totes)
            if ms < current_makespan:
                current_belt = neighbor_belt
                current_makespan = ms
                current_result = _simulate_schedule(
                    tote_sequence=current_totes,
                    orders=orders,
                    totes=totes,
                    belt_assignment=current_belt,
                )
                improved = True
                break  # first improvement

        if not improved:
            break  # local optimum reached

    return LocalSearchResult(
        best_belt_assignment=current_belt,
        best_tote_sequence=current_totes,
        best_makespan=current_makespan,
        best_event_log=current_result.event_log,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Convenience: write conveyor CSV compatible with existing baseline
# ──────────────────────────────────────────────────────────────────────────────

def write_conveyor_input_csv(
    event_log: List[Dict[str, object]],
    output_path: str,
) -> None:
    """
    Write an input CSV for the physical conveyor (same format as
    input_baseline_constraint_furthest.csv).

    Each row corresponds to one "row" of items placed from a tote.
    Columns: conv_num,circle,pentagon,trapezoid,triangle,star,moon,heart,cross
    """
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["conv_num"] + ITEM_NAMES)
        for event in event_log:
            counts = list(event["items"])
            writer.writerow([event["conv"]] + counts)


def write_tote_sequence_csv(
    tote_sequence: ToteSequence,
    output_path: str,
) -> None:
    """
    Write the tote loading sequence (optimal tote order) to a CSV file.

    Columns:
      step_index : 0-based index in the loading sequence
      tote_id    : ID of the tote loaded at that step
    """
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step_index", "tote_id"])
        for idx, tote_id in enumerate(tote_sequence):
            writer.writerow([idx, tote_id])


def write_tote_item_sequence_csv(
    tote_sequence: ToteSequence,
    event_log: List[Dict[str, object]],
    output_path: str,
) -> None:
    """
    Write one row per tote in loading order, with item counts aggregated
    across all conveyors for that tote (unique tote_id per row).

    Columns:
      step_index : index of the tote in the loading sequence
      tote_id    : ID of the tote (unique per row)
      circle,...,cross : total item counts from this tote (all conveyors summed)
    """
    # Aggregate event_log by tote_id: sum item counts for each tote
    tote_items: Dict[int, List[int]] = {}
    for event in event_log:
        tote_id = int(event["tote"])
        counts = list(event["items"])
        if tote_id not in tote_items:
            tote_items[tote_id] = [0] * NUM_ITEM_TYPES
        for i in range(NUM_ITEM_TYPES):
            tote_items[tote_id][i] += counts[i]

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step_index", "tote_id"] + ITEM_NAMES)
        for step_idx, tote_id in enumerate(tote_sequence):
            counts = tote_items.get(tote_id, [0] * NUM_ITEM_TYPES)
            writer.writerow([step_idx, tote_id] + counts)


# ──────────────────────────────────────────────────────────────────────────────
# Example usage with small mock data
# ──────────────────────────────────────────────────────────────────────────────

def _build_small_mock_instance() -> Tuple[OrdersDict, TotesDict, BeltAssignment, ToteSequence]:
    """
    Build a tiny mock instance (2 conveyors, 3 orders, 2 totes) purely for
    demonstration of the API, independent of the CSV files.
    """
    global NUM_CONVEYORS
    old_num_conv = NUM_CONVEYORS
    NUM_CONVEYORS = 2  # override for the small example

    orders: OrdersDict = {
        1: Order(items=[(0, 2)], total_items=2),      # needs 2 circles
        2: Order(items=[(0, 1), (1, 1)], total_items=2),  # 1 circle, 1 pentagon
        3: Order(items=[(1, 2)], total_items=2),      # 2 pentagons
    }

    totes: TotesDict = {
        1: [
            ToteEntry(order=1, item_type=0, quantity=2),
            ToteEntry(order=2, item_type=0, quantity=1),
        ],
        2: [
            ToteEntry(order=2, item_type=1, quantity=1),
            ToteEntry(order=3, item_type=1, quantity=2),
        ],
    }

    # Initial belt assignment: orders 1,2 on conv 0; order 3 on conv 1
    belt_assignment: BeltAssignment = [
        [1, 2],
        [3],
    ]

    tote_sequence: ToteSequence = [1, 2]

    # Restore original number of conveyors for the rest of the module
    NUM_CONVEYORS = old_num_conv

    return orders, totes, belt_assignment, tote_sequence


def _example_usage() -> None:
    """
    Run a small end-to-end example using the mock instance.

    This does NOT use the CSV files; it is only for demonstration of how to
    call compute_makespan() and local_search().
    """
    orders, totes, belt_assignment, tote_sequence = _build_small_mock_instance()

    # For the mock example we temporarily set NUM_CONVEYORS = 2
    original_num_conv = NUM_CONVEYORS
    NUM_CONVEYORS = 2

    base_ms = compute_makespan(orders, totes, belt_assignment, tote_sequence)
    print(f"Initial makespan (mock example): {base_ms:.1f} s")

    ls_result = local_search(
        orders=orders,
        totes=totes,
        initial_belt_assignment=belt_assignment,
        initial_tote_sequence=tote_sequence,
        max_iterations=50,
    )

    print(f"Improved makespan (mock example): {ls_result.best_makespan:.1f} s")
    print("Best belt assignment (per conveyor):")
    for ci, q in enumerate(ls_result.best_belt_assignment):
        print(f"  Conveyor {ci}: {q}")
    print(f"Best tote sequence: {ls_result.best_tote_sequence}")

    NUM_CONVEYORS = original_num_conv


if __name__ == "__main__":
    # Example on mock data
    # _example_usage()

    # Uncomment the block below to run on your real CSV files.
    # Adjust paths as needed for your environment.
    #
    orders, totes = load_orders_and_totes(
        order_itemtypes_path="ranDataGen/order_itemtypes.csv",
        order_quantities_path="ranDataGen/order_quantities.csv",
        orders_totes_path="ranDataGen/orders_totes.csv",
    )
    
    # Simple initial solution: round-robin assignment and sorted totes
    all_orders = sorted(orders.keys())
    belt_assignment: BeltAssignment = [[] for _ in range(NUM_CONVEYORS)]
    for idx, oid in enumerate(all_orders):
        belt_assignment[idx % NUM_CONVEYORS].append(oid)
    
    tote_sequence: ToteSequence = sorted(totes.keys())
    
    ls_result = local_search(
        orders=orders,
        totes=totes,
        initial_belt_assignment=belt_assignment,
        initial_tote_sequence=tote_sequence,
        max_iterations=500,
    )

    print(f"Best makespan (CSV instance): {ls_result.best_makespan:.1f} s")

    # Re-simulate best schedule to get per-order completion times
    best_sim = _simulate_schedule(
        tote_sequence=ls_result.best_tote_sequence,
        orders=orders,
        totes=totes,
        belt_assignment=ls_result.best_belt_assignment,
    )

    print("Per-order completion times (CSV instance):")
    for oid in sorted(best_sim.order_completion_times.keys()):
        t = best_sim.order_completion_times[oid]
        if t == float("inf"):
            print(f"  Order {oid}: never completed (∞)")
        else:
            print(f"  Order {oid}: {t:.1f} s")

    write_conveyor_input_csv(
        event_log=ls_result.best_event_log,
        output_path="local_search_best_schedule.csv",
    )

    write_tote_sequence_csv(
        tote_sequence=ls_result.best_tote_sequence,
        output_path="local_search_best_tote_sequence.csv",
    )

    write_tote_item_sequence_csv(
        tote_sequence=ls_result.best_tote_sequence,
        event_log=ls_result.best_event_log,
        output_path="local_search_best_tote_item_sequence.csv",
    )

